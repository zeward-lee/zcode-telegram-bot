"""Session 同步协调器。

在 AppServerClient 之上,封装两个方向的同步:
- bot→session:send_and_stream 流式拿到 assistant 回复,直接推给用户(体验好)
- TUI→bot:后台轮询 session/events,发现 TUI 在同 session 的活动,回调推送

架构约束:app-server 是单客户端/进程模型,bot 自己的 app-server 与 TUI 的不共享
实时事件总线。所以 bot 自己的 send 能流式(TUI 同理),但发现 TUI 的活动靠轮询。
两边共享同一个 SQLite,数据层面天然同步;轮询只是补足"实时通知"这一层。
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import AsyncIterator, Callable, Optional

from app_server_client import (
    AppServerClient,
    AppServerError,
    AppServerUnavailableError,
    SessionUnavailableError,
    StreamEvent,
)

logger = logging.getLogger("zcode-tg-bot.sync")


@dataclass
class TurnResult:
    """一次 send_and_stream 的聚合结果(供 bot 渲染最终消息)。"""

    response: str
    total_tokens: int
    error: Optional[str] = None  # turn.failed 时填


class SessionSync:
    """全局单例协调器:持有唯一的 AppServerClient,管理所有 session 的轮询。

    用法:
        sync = SessionSync(client, poll_interval=1.5)
        await sync.start()
        # 发消息(流式)
        async for ev in sync.send_stream(sid, text): ...
        # 关联 session 开始轮询 TUI 更新
        sync.watch(sid, last_seq, on_new_events)
        # 停止轮询
        sync.unwatch(sid)
    """

    def __init__(
        self,
        client: AppServerClient,
        poll_interval: float = 1.5,
    ) -> None:
        self.client = client
        self.poll_interval = poll_interval
        # session_id → watcher 列表(同 session 多个 key 共享时,各自回调不覆盖)
        self._watchers: dict[str, list[_Watcher]] = {}
        self._poll_task: Optional[asyncio.Task] = None
        # app-server 崩溃时暂停轮询(set 时 _poll_loop 停下,clear 时恢复)
        self._paused = asyncio.Event()
        self._crash_logged = False  # 避免每 1.5s 刷同一条 warning

    def pause(self) -> None:
        """app-server 不可用时暂停轮询(避免撞墙刷日志)。"""
        if not self._paused.is_set():
            logger.warning("app-server 不可用,轮询暂停")
        self._paused.set()

    def resume(self) -> None:
        """app-server 恢复后恢复轮询。"""
        if self._paused.is_set():
            logger.info("app-server 恢复,轮询继续")
        self._paused.clear()
        self._crash_logged = False

    async def start(self) -> None:
        """启动后台轮询协程。幂等。"""
        if self._poll_task and not self._poll_task.done():
            return
        await self.client.start()
        self._poll_task = asyncio.ensure_future(self._poll_loop())
        logger.info("SessionSync 启动(poll_interval=%.1fs)", self.poll_interval)

    async def stop(self) -> None:
        """停止轮询并关闭 app-server。"""
        self._watchers.clear()
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None
        await self.client.close()

    # ---------- bot → session(流式)----------

    async def send_stream(
        self,
        session_id: str,
        content: str,
        turn_timeout: float = 600.0,
    ) -> AsyncIterator[StreamEvent]:
        """发消息并流式 yield session/event(属于本 turn)。

        调用方据此增量渲染(收到 model.streaming 更新文本,turn.completed 收尾)。

        期间会静音该 session 的所有轮询 watcher,避免 bot 自己发的消息
        被轮询当成"TUI 活动"重复推送(回环)。结束后把水位跳到最新 seq。
        """
        watchers = self._watchers.get(session_id, [])
        for w in watchers:
            w.muted = True  # 静音:发送期间不推送
        last_seq = 0
        try:
            async for ev in self.client.send_and_stream(
                session_id, content, turn_timeout=turn_timeout
            ):
                last_seq = max(last_seq, ev.seq)
                yield ev
        finally:
            # 恢复监听,但把水位跳到自己产生的事件之后(避免回环)
            for w in watchers:
                w.last_seq = max(w.last_seq, last_seq)
                w.muted = False

    async def send_collect(
        self,
        session_id: str,
        content: str,
        turn_timeout: float = 600.0,
    ) -> TurnResult:
        """发消息并聚合为最终结果(不流式,等 turn 完成)。

        适合简单场景;需要流式体验用 send_stream。
        """
        response_parts: list[str] = []
        total_tokens = 0
        error: Optional[str] = None
        async for ev in self.send_stream(session_id, content, turn_timeout=turn_timeout):
            if ev.type == "model.streaming":
                delta = ev.payload.get("delta")
                if isinstance(delta, str):
                    response_parts.append(delta)
            elif ev.type == "turn.completed":
                # 以 turn.completed.response 为准(完整、权威)
                response_parts = [ev.payload.get("response", "") or "".join(response_parts)]
                total_tokens = ev.payload.get("tokenCount", 0)
            elif ev.type == "turn.failed":
                error = ev.payload.get("error", {}).get("message", str(ev.payload))
        return TurnResult(
            response=response_parts[0] if response_parts else "",
            total_tokens=total_tokens,
            error=error,
        )

    # ---------- TUI → bot(轮询)----------

    def watch(
        self,
        session_id: str,
        last_seq: int,
        on_events: Callable[[str, list], None],
    ) -> None:
        """开始监视一个 session:TUI 在该 session 上的新活动会回调 on_events。

        同 session 可被多个 key 共享(群共享),各自 on_events 追加进列表,
        互不覆盖。last_seq 取已有 watcher 的最大值(水位只升不降)。

        Args:
            session_id: 要监视的 session(需已 resume + subscribe 过)
            last_seq: 当前已知的水位 seq(只推送 > last_seq 的事件)
            on_events: 回调 (session_id, new_events);在 loop 线程同步执行
        """
        lst = self._watchers.setdefault(session_id, [])
        # 已有 watcher 的话,水位取最大(避免后注册的把水位拉低)
        seq = last_seq
        for w in lst:
            seq = max(seq, w.last_seq)
        lst.append(_Watcher(session_id=session_id, last_seq=seq, on_events=on_events))
        logger.info("开始监视 session %s (from seq=%d, 共 %d 个 watcher)", session_id, seq, len(lst))

    def unwatch(self, session_id: str) -> None:
        """停止监视某 session 的所有 watcher。"""
        if self._watchers.pop(session_id, None):
            logger.info("停止监视 session %s", session_id)

    def get_last_seq(self, session_id: str) -> Optional[int]:
        """取某 session 当前的轮询水位(供持久化,取所有 watcher 的最大值)。"""
        lst = self._watchers.get(session_id)
        if not lst:
            return None
        return max(w.last_seq for w in lst)

    async def _poll_loop(self) -> None:
        """后台轮询:对每个 watcher 调 get_events,有新事件就回调。"""
        while True:
            try:
                # app-server 崩溃时暂停,等恢复
                if self._paused.is_set():
                    await self._paused.wait()
                await asyncio.sleep(self.poll_interval)
                # 展平所有 watcher(session_id 可被多 key 共享,各自独立轮询)
                flat = [w for ws in self._watchers.values() for w in ws]
                for watcher in flat:
                    await self._poll_one(watcher)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("轮询循环异常(继续)")
                await asyncio.sleep(self.poll_interval)

    async def _poll_one(self, watcher: "_Watcher") -> None:
        """轮询单个 session 的新事件。"""
        if watcher.muted:
            return  # bot 正在发消息,跳过(避免回环)
        try:
            result = await self.client.get_events(
                watcher.session_id, after_seq=watcher.last_seq
            )
        except SessionUnavailableError:
            # session 被关闭了,停止监视
            logger.warning("轮询的 session %s 已不可用,停止监视", watcher.session_id)
            self._watchers.pop(watcher.session_id, None)
            return
        except AppServerUnavailableError:
            # app-server 进程崩了:暂停轮询,保留 watcher 等恢复(不 pop)
            self.pause()
            return
        except AppServerError as e:
            logger.warning("轮询 session %s 出错: %s", watcher.session_id, e)
            return

        events = result.get("events", []) if isinstance(result, dict) else []
        if not events:
            return
        # 更新水位
        new_max = max((e.get("seq", watcher.last_seq) for e in events), default=watcher.last_seq)
        watcher.last_seq = max(watcher.last_seq, new_max)
        # 只回调有意义的事件(过滤掉 session.updated 这类纯内部状态变化)
        meaningful = [e for e in events if _is_meaningful_for_user(e)]
        if meaningful:
            try:
                watcher.on_events(watcher.session_id, meaningful)
            except Exception:
                logger.exception("on_events 回调异常")


@dataclass
class _Watcher:
    """单个 session 的轮询状态。"""

    session_id: str
    last_seq: int
    on_events: Callable[[str, list], None]
    muted: bool = False  # bot 自己 send 期间静音,避免回环


# 用户关心的 TUI 活动事件类型(session.updated / state.updated 是内部噪音,过滤掉)
_MEANINGFUL_TYPES = {
    "turn.started",
    "turn.completed",
    "turn.failed",
    "message.upserted",
    "message.removed",
    "tool.updated",
    "model.streaming",
    "session.closed",
}


def _is_meaningful_for_user(event: dict) -> bool:
    """判断一个事件是否值得推送给用户(过滤内部噪音)。"""
    return event.get("type") in _MEANINGFUL_TYPES
