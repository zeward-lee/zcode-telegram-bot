"""ZCode app-server JSON-RPC 客户端。

封装 `zcode app-server` 子进程(stdio),通过 ZCode Protocol 与之通信。
提供 session 的 create / list / resume / subscribe / send / events 等能力。

协议要点(实测确认):
- 传输层:NDJSON(每行一个 JSON 对象),无 Content-Length 头。
- 请求帧:{"id", "method", "params?"}            —— 注意:不带 "jsonrpc" 字段
- 通知帧:{"method", "params?"}                   (服务端→客户端,无 id)
- 响应帧:{"id", "result"} 或 {"id", "error":{code,message,data?}}
- session/send 是 fire-and-forget(result 为 null),实际结果通过 session/event 通知异步推送。
- 错误码:-32700 ParseError / -32600 InvalidRequest / -32601 MethodNotFound /
         -32602 InvalidParams / -32004 sessionUnavailable / -32010 prompt already running。

这个客户端是同步子进程 + asyncio 包装:stdout 由后台 reader 线程喂入,
请求/响应按 id 配对,通知路由到 handler。所有公开方法都是 async。
"""
from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import threading
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Optional

from zcode_client import find_zcode_cli

logger = logging.getLogger("zcode-tg-bot.appserver")

# 默认 deliveryKind:语义为"远程可回放客户端",seq 从 0 单调递增,
# 配合持久化的 afterSeq 实现断线续传、不重复推送。
DEFAULT_DELIVERY_KIND = "web-remote-replayable"


class AppServerError(RuntimeError):
    """app-server 协议层错误(对应 JSON-RPC error)。"""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.data = data


class AppServerUnavailableError(AppServerError):
    """app-server 子进程不可用(未启动 / 已退出)。"""


class SessionUnavailableError(AppServerError):
    """session 不存在或已关闭(-32004)。"""


class PromptAlreadyRunningError(AppServerError):
    """该 session 已有一个 prompt 在跑(-32010)。"""


@dataclass
class SessionInfo:
    """session/list 返回的单条 session 摘要。"""

    session_id: str
    title: str
    directory: str
    updated_at: int  # ms epoch
    task_type: str = "interactive"
    raw: dict = field(default_factory=dict)


@dataclass
class StreamEvent:
    """从 session/event 通知解析出的一个流式事件,供上层渲染。"""

    type: str  # turn.started / model.streaming / tool.updated / message.upserted / turn.completed / turn.failed / ...
    seq: int
    payload: dict
    turn_id: Optional[str] = None


class AppServerClient:
    """到 `zcode app-server` 的异步客户端。

    单实例持有一个 app-server 子进程;所有 session 操作复用同一连接。
    生命周期:start() → 多次调用 → close()。

    Args:
        cwd: app-server 工作目录
        cli_path: zcode CLI 路径(不传则自动定位)
        request_timeout: 单个请求等待响应的超时秒数
    """

    def __init__(
        self,
        cwd: str = ".",
        cli_path: Optional[str] = None,
        request_timeout: float = 60.0,
    ) -> None:
        self.cwd = cwd
        self.cli_path = cli_path or find_zcode_cli()
        self.request_timeout = request_timeout

        self._proc: Optional[subprocess.Popen] = None
        self._reader_thread: Optional[threading.Thread] = None
        # asyncio 侧的状态(在 loop 线程里访问)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._next_id: int = 0
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._event_handlers: list[Callable[[dict], None]] = []
        self._stdout_q: asyncio.Queue[Optional[str]] = asyncio.Queue()
        self._started = False
        self._lock = asyncio.Lock()  # 串行化 start/close
        # permission 审批去重(reannounce 重发时用 requestId 去重)
        # _perm_pending: requestId → 最新 rid(reannounce 重发时更新,审批时用最新 rid 回应)
        self._perm_pending: dict[str, Any] = {}
        # _perm_cache: requestId → 已缓存的 decision 结果(审批完成后,后续重发直接回)
        self._perm_cache: dict[str, dict] = {}

    # ---------- 生命周期 ----------

    async def start(self) -> None:
        """启动 app-server 子进程并开始读 stdout。幂等。"""
        async with self._lock:
            if self._started and self._proc and self._proc.poll() is None:
                return
            self._loop = asyncio.get_running_loop()
            args = self._build_args()
            logger.info("启动 app-server: %s (cwd=%s)", " ".join(args[:2]) + " ...", self.cwd)
            # Windows 下 node.exe 是控制台程序,不设 CREATE_NO_WINDOW 会弹黑窗
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            self._proc = subprocess.Popen(
                args,
                cwd=self.cwd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # 行缓冲
                creationflags=creationflags,
            )
            self._started = True
            # 后台线程:逐行读 stdout,塞进 asyncio.Queue
            self._reader_thread = threading.Thread(
                target=self._stdout_reader, name="appserver-stdout", daemon=True
            )
            self._reader_thread.start()
            # 后台线程:drain stderr 到日志(诊断用)
            threading.Thread(
                target=self._stderr_drain, name="appserver-stderr", daemon=True
            ).start()
            # 启动 loop 侧的行分发协程
            asyncio.ensure_future(self._dispatch_loop())

    def _build_args(self) -> list[str]:
        cli = self.cli_path
        if cli.endswith(".cjs"):
            return ["node", cli, "app-server"]
        return [cli, "app-server"]

    def _stdout_reader(self) -> None:
        """后台线程:逐行读 stdout,通过 loop.call_soon_threadsafe 投递。"""
        assert self._proc and self._proc.stdout
        loop = self._loop
        try:
            for line in iter(self._proc.stdout.readline, ""):
                line = line.rstrip("\n")
                if not line:
                    continue
                if loop and loop.is_closed():
                    break
                # 跨线程安全投递到 asyncio.Queue
                if loop:
                    loop.call_soon_threadsafe(self._stdout_q.put_nowait, line)
        except Exception:
            logger.exception("app-server stdout reader 异常")
        finally:
            # 投递 sentinel 通知分发协程退出
            if loop and not loop.is_closed():
                loop.call_soon_threadsafe(self._stdout_q.put_nowait, None)

    def _stderr_drain(self) -> None:
        """后台线程:把 stderr 写到日志(避免管道堵塞)。"""
        assert self._proc and self._proc.stderr
        try:
            for line in iter(self._proc.stderr.readline, ""):
                line = line.rstrip()
                if line:
                    logger.debug("app-server stderr: %s", line)
        except Exception:
            pass

    async def _dispatch_loop(self) -> None:
        """asyncio 协程:从队列取行,按 id 配对响应 / 路由通知。"""
        while True:
            line = await self._stdout_q.get()
            if line is None:  # sentinel:子进程 stdout 关闭
                # 唤醒所有 pending future 为不可用错误
                for fut in self._pending.values():
                    if not fut.done():
                        fut.set_exception(
                            AppServerUnavailableError("app-server 进程已退出")
                        )
                self._pending.clear()
                self._started = False
                break
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("app-server 非 JSON 行: %s", line[:200])
                continue
            self._handle_message(msg)

    def _handle_message(self, msg: dict) -> None:
        """分发单条消息:响应按 id 匹配 future,通知交给 handlers。"""
        # 响应(有 id):可能是我们的请求结果,也可能是服务端反向请求
        if "id" in msg:
            rid = msg["id"]
            # 我们的请求的响应
            fut = self._pending.pop(rid, None)
            if fut is not None and not fut.done():
                if "error" in msg:
                    fut.set_exception(self._make_error(msg["error"]))
                elif "result" in msg:
                    fut.set_result(msg["result"])
                else:
                    fut.set_exception(AppServerError(-32603, "响应无 result/error"))
                return
            # 没匹配到 pending:可能是服务端发的反向请求(permission/userInput)
            # 走通知处理(handler 内部负责回响应)
            if "method" in msg:
                self._dispatch_notification(msg)
                return
            logger.debug("收到未匹配的响应: id=%s", rid)
            return

        # 通知(无 id)
        if "method" in msg:
            self._dispatch_notification(msg)
            return

        logger.warning("无法识别的 app-server 消息: %s", str(msg)[:200])

    def _dispatch_notification(self, msg: dict) -> None:
        """把通知(含反向请求)分发给注册的 handler。"""
        for handler in list(self._event_handlers):
            try:
                handler(msg)
            except Exception:
                logger.exception("event handler 异常")

    @staticmethod
    def _make_error(err: dict) -> AppServerError:
        """把 JSON-RPC error 对象映射到具体异常类。"""
        code = err.get("code", -1)
        message = err.get("message", "unknown error")
        data = err.get("data")
        if code == -32004:
            return SessionUnavailableError(code, message, data)
        if code == -32010:
            return PromptAlreadyRunningError(code, message, data)
        return AppServerError(code, message, data)

    def on_message(self, handler: Callable[[dict], None]) -> None:
        """注册一个通知/反向请求 handler。handler 同步执行(在 loop 线程)。"""
        self._event_handlers.append(handler)

    async def close(self) -> None:
        """优雅关闭:关闭 stdin、终止子进程。"""
        async with self._lock:
            proc = self._proc
            if not proc:
                return
            try:
                if proc.stdin:
                    proc.stdin.close()
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
            finally:
                self._proc = None
                self._started = False

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    # ---------- 原始请求 ----------

    async def request(
        self, method: str, params: Optional[dict] = None, timeout: Optional[float] = None
    ) -> Any:
        """发一个 JSON-RPC 请求并等待响应(result 或抛 AppServerError)。

        不带 "jsonrpc" 字段(ZCode 协议要求)。
        """
        if not self.is_running:
            raise AppServerUnavailableError("app-server 未启动")
        self._next_id += 1
        rid = self._next_id
        msg: dict = {"id": rid, "method": method}
        if params is not None:
            msg["params"] = params
        fut: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[rid] = fut
        line = json.dumps(msg, ensure_ascii=False)
        try:
            assert self._proc and self._proc.stdin
            self._proc.stdin.write(line + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            self._pending.pop(rid, None)
            raise AppServerUnavailableError(f"写入 app-server 失败: {e}") from e
        try:
            return await asyncio.wait_for(fut, timeout=timeout or self.request_timeout)
        except asyncio.TimeoutError:
            self._pending.pop(rid, None)
            raise

    async def notify(self, method: str, params: Optional[dict] = None) -> None:
        """发一个通知(无 id,不等响应)。"""
        if not self.is_running:
            raise AppServerUnavailableError("app-server 未启动")
        msg: dict = {"method": method}
        if params is not None:
            msg["params"] = params
        line = json.dumps(msg, ensure_ascii=False)
        assert self._proc and self._proc.stdin
        try:
            self._proc.stdin.write(line + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            raise AppServerUnavailableError(f"写入 app-server 失败: {e}") from e

    async def respond(self, request_id: Any, result: Any) -> None:
        """响应服务端的反向请求(如 permission.requested)。"""
        msg = {"id": request_id, "result": result}
        line = json.dumps(msg, ensure_ascii=False)
        assert self._proc and self._proc.stdin
        self._proc.stdin.write(line + "\n")
        self._proc.stdin.flush()

    # ---------- session 高层 API ----------

    async def create_session(
        self,
        workspace_path: str,
        workspace_key: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> str:
        """创建新 session,返回 sessionId。

        Args:
            workspace_path: 工作目录
            workspace_key: 工作区标识(默认同 workspace_path)
            mode: 权限模式(build/edit/plan/yolo)
        """
        params: dict = {
            "workspace": {
                "workspacePath": workspace_path,
                "workspaceKey": workspace_key or workspace_path,
            }
        }
        if mode:
            params["mode"] = mode
        result = await self.request("session/create", params)
        session = result.get("session", {})
        sid = session.get("sessionId")
        if not sid:
            raise AppServerError(-32603, f"session/create 未返回 sessionId: {result}")
        logger.info("已创建 session %s (workspace=%s)", sid, workspace_path)
        return sid

    async def list_sessions(
        self, workspace_path: Optional[str] = None, limit: int = 20
    ) -> list[SessionInfo]:
        """列出 session。不传 workspace 则列所有。"""
        params: dict = {"limit": limit}
        if workspace_path:
            params["workspace"] = {
                "workspacePath": workspace_path,
                "workspaceKey": workspace_path,
            }
        result = await self.request("session/list", params)
        sessions_raw = result.get("sessions", [])
        infos: list[SessionInfo] = []
        for s in sessions_raw:
            infos.append(
                SessionInfo(
                    session_id=s.get("sessionId", ""),
                    title=s.get("title", "") or "(无标题)",
                    directory=s.get("directory") or s.get("workspace", {}).get("workspacePath", ""),
                    updated_at=s.get("updatedAt", 0),
                    task_type=s.get("sessionKind", "interactive"),
                    raw=s,
                )
            )
        return infos

    async def resume_session(self, session_id: str) -> dict:
        """resume 一个已有 session,返回快照(含 messages 历史)。

        若旧 session 历史模型已失效(restoreWarning),自动用当前 workspace
        可用模型重新应用 runtime,清除警告后再 resume。这样 bot 在 headless
        模式下也能接续旧的 TUI/desktop session。
        """
        snap = await self.request("session/resume", {"sessionId": session_id})
        # 检查是否有 restoreWarning(projection.lastError)
        proj = snap.get("projection", {}) if isinstance(snap, dict) else {}
        last_err = proj.get("lastError")
        if last_err and last_err.get("type") == "ZCODE_RUNTIME_MODEL_UNAVAILABLE":
            logger.info(
                "session %s 触发 restoreWarning,尝试用当前可用模型修复", session_id
            )
            runtime_model = await self._build_current_runtime_model()
            if runtime_model:
                snap = await self.request(
                    "session/resume",
                    {"sessionId": session_id, "runtimeModel": runtime_model},
                )
                proj2 = snap.get("projection", {}) if isinstance(snap, dict) else {}
                if not proj2.get("lastError"):
                    logger.info("✅ restoreWarning 已清除(session=%s)", session_id)
                else:
                    logger.warning(
                        "restoreWarning 修复后仍存在: %s", proj2.get("lastError")
                    )
        return snap

    async def _build_current_runtime_model(self) -> Optional[dict]:
        """从 workspace catalog + cli config 构造当前可用的 runtimeModel。

        用于绕过旧 session 的 restoreWarning(模型 runtime revision 不匹配)。
        返回 {revision, generatedAt, model, provider} 或 None(失败时)。
        """
        try:
            cat = await self._read_model_catalog()
            if not cat or not cat.get("available"):
                return None
            avail = cat["available"][0]
            ref = avail["ref"]
            prov_id = ref["providerId"]
            # 从 catalog 取 provider 骨架
            prov = None
            for p in cat.get("providers", []):
                if p.get("providerId") == prov_id:
                    prov = dict(p)
                    break
            if not prov:
                return None
            # 补 baseURL + apiKey(catalog 里被省略,从 cli config 读)
            self._enrich_provider_from_config(prov, prov_id)
            return {
                "revision": str(cat.get("revision", 0)),  # revision 要 string
                "generatedAt": 0,
                "model": {"providerId": prov_id, "modelId": ref["modelId"]},
                "provider": prov,
            }
        except Exception:
            logger.exception("构造 runtimeModel 失败")
            return None

    async def _read_model_catalog(self) -> dict:
        """读当前 workspace 的模型目录(workspace/readState)。"""
        # workspace 用 app-server 的 cwd(即 self.cwd)
        ws = self.cwd
        result = await self.request(
            "workspace/readState",
            {"workspace": {"workspacePath": ws, "workspaceKey": ws}},
        )
        return result.get("modelCatalog", {}) if isinstance(result, dict) else {}

    @staticmethod
    def _enrich_provider_from_config(prov: dict, provider_id: str) -> None:
        """从 ~/.zcode/cli/config.json 给 provider 补 baseURL / apiKey / kind。"""
        import os
        config_path = os.path.expanduser("~/.zcode/cli/config.json")
        try:
            with open(config_path, encoding="utf-8") as f:
                cfg = json.load(f)
            cfg_prov = cfg.get("provider", {}).get(provider_id, {})
            opts = cfg_prov.get("options", {})
            if opts.get("baseURL"):
                prov["baseURL"] = opts["baseURL"]
            if opts.get("apiKey"):
                # apiKey schema 是 discriminatedUnion:{source:"inline", value:"..."}
                prov["apiKey"] = {"source": "inline", "value": opts["apiKey"]}
            if cfg_prov.get("kind"):
                prov["kind"] = cfg_prov["kind"]
        except (OSError, json.JSONDecodeError):
            logger.warning("读取 cli config 失败,provider 可能不完整")

    async def subscribe(
        self,
        session_id: str,
        delivery_kind: str = DEFAULT_DELIVERY_KIND,
        after_seq: int = 0,
        include_snapshot: bool = False,
    ) -> dict:
        """订阅 session 的事件流,返回 {eventSeq, events, sessionId}。

        之后该 session 的活动会通过 session/event 通知推送(经 on_message handler)。
        after_seq 用于断线续传:只取 > afterSeq 的事件。
        """
        params = {
            "sessionId": session_id,
            "deliveryKind": delivery_kind,
            "afterSeq": after_seq,
            "includeSnapshot": include_snapshot,
        }
        return await self.request("session/subscribe", params)

    async def send_message(
        self, session_id: str, content: str, expected_revision: Optional[int] = None
    ) -> Optional[dict]:
        """向 session 发一条用户消息(fire-and-forget)。

        result 为 null;实际结果通过 session/event 通知异步返回。
        若该 session 已有 prompt 在跑,抛 PromptAlreadyRunningError(-32010)。

        expected_revision: 乐观锁,传入已知 stateRevision;不匹配则报错(避免覆盖)。
        """
        params: dict = {"sessionId": session_id, "content": content}
        if expected_revision is not None:
            params["expectedRevision"] = expected_revision
        return await self.request("session/send", params)

    async def get_events(
        self,
        session_id: str,
        after_seq: int,
        limit: Optional[int] = None,
    ) -> dict:
        """拉取 afterSeq 之后的事件(轮询 TUI 侧更新用)。

        注意:session/events 的 params schema 为 {sessionId, afterSeq?, limit?}
        —— 不含 deliveryKind(与 subscribe 不同)。

        返回结构(按协议):含 events 列表与最新 seq。
        """
        params: dict = {"sessionId": session_id, "afterSeq": after_seq}
        if limit is not None:
            params["limit"] = limit
        return await self.request("session/events", params)

    async def get_messages(
        self, session_id: str, after_message_id: Optional[str] = None, limit: int = 50
    ) -> dict:
        """取结构化消息历史(含 role + parts 文本)。"""
        params: dict = {"sessionId": session_id, "limit": limit}
        if after_message_id:
            params["afterMessageId"] = after_message_id
        return await self.request("session/messages", params)

    # ---------- 流式 send 辅助 ----------

    async def send_and_stream(
        self,
        session_id: str,
        content: str,
        turn_timeout: float = 600.0,
    ) -> AsyncIterator[StreamEvent]:
        """发消息并 yield 该 turn 的所有 session/event,直到 turn.completed/failed。

        依赖外部已注册的 on_message handler 把通知转成 asyncio 事件;
        这里用一个临时队列收集本 turn 的事件。

        注意:同一个 AppServerClient 的通知是全局分发的,本方法用 turn_id 过滤
        属于本次 send 的事件。turn_id 在第一个 turn.started 事件里确定。
        """
        # 临时队列:handler 把本 session 的 session/event 推进来
        ev_q: asyncio.Queue[Optional[StreamEvent]] = asyncio.Queue()
        my_turn_id: list[Optional[str]] = [None]
        done = asyncio.Event()

        def handler(msg: dict) -> None:
            # handler 在 loop 线程内同步执行(_dispatch_loop → _dispatch_notification)
            if msg.get("method") != "session/event":
                # 反向请求:permission 转事件投队列(让上层决定),userInput 自动应答
                if msg.get("method") == "interaction/requestPermission":
                    self._handle_permission_request(msg, ev_q)
                else:
                    self._maybe_auto_respond(msg)
                return
            params = msg.get("params", {})
            if params.get("sessionId") != session_id:
                return
            ev = StreamEvent(
                type=params.get("type", ""),
                seq=params.get("seq", 0),
                payload=params.get("payload", {}),
                turn_id=params.get("turnId"),
            )
            # 绑定 turn_id(第一个 turn.started)
            if ev.type == "turn.started" and my_turn_id[0] is None:
                my_turn_id[0] = ev.turn_id
            # 只投递属于本 turn 的事件(turn_id 匹配,或尚未绑定 turn_id 的 started)
            if my_turn_id[0] is None or ev.turn_id == my_turn_id[0]:
                ev_q.put_nowait(ev)
                if ev.type in ("turn.completed", "turn.failed"):
                    done.set()

        self._event_handlers.append(handler)
        try:
            await self.send_message(session_id, content)
            # 收事件直到 turn 结束或超时
            while not done.is_set():
                try:
                    ev = await asyncio.wait_for(ev_q.get(), timeout=turn_timeout)
                except asyncio.TimeoutError:
                    logger.warning("send_and_stream 超时(%.0fs)", turn_timeout)
                    break
                if ev is None:
                    break
                yield ev
                if ev.type in ("turn.completed", "turn.failed"):
                    break
        finally:
            if handler in self._event_handlers:
                self._event_handlers.remove(handler)

    def _handle_permission_request(self, msg: dict, ev_q: "asyncio.Queue") -> None:
        """处理 interaction/requestPermission 反向请求。

        ZCode 会因 reannounce 机制重发(同 requestId,新 rid),用 requestId 去重:
        - 首次见 requestId → 投 permission.requested 事件到队列(让上层 await 用户决策)
        - 重发(同 requestId)→ 更新 pending rid,不重复投事件(避免刷屏)
        - 已缓存结果(用户已审批)→ 直接用最新 rid 回应缓存结果

        注意:本方法在 loop 线程同步执行,不能 await。审批结果由上层 respond 异步回。
        """
        rid = msg.get("id")
        params = msg.get("params", {})
        request_id = params.get("requestId", "")
        if not rid or not request_id:
            return

        # 已有缓存结果 → 直接回应(用户已审批过这个 requestId)
        if request_id in self._perm_cache:
            cached = self._perm_cache[request_id]
            asyncio.ensure_future(self.respond(rid, cached))
            logger.info("permission 重发(requestId=%s)→ 回缓存结果 %s", request_id, cached.get("decision"))
            return

        # 首次见 → 投事件,记 pending
        is_new = request_id not in self._perm_pending
        self._perm_pending[request_id] = rid  # 更新为最新 rid(重发时覆盖)
        if is_new:
            ev = StreamEvent(
                type="permission.requested",
                seq=0,
                payload={
                    "rid": rid,  # 注意:上层审批后应查 _perm_pending 拿最新 rid
                    "request_id": request_id,
                    "tool_name": params.get("toolName", ""),
                    "input": params.get("input", {}),
                    "risk_level": params.get("riskLevel", ""),
                    "reason": params.get("reason", ""),
                    "tool_call_id": params.get("toolCallId", ""),
                },
            )
            ev_q.put_nowait(ev)
            logger.info(
                "permission 请求(requestId=%s tool=%s)→ 转发待审批",
                request_id, params.get("toolName"),
            )
        else:
            logger.debug("permission 重发(requestId=%s)→ 更新 rid,不重复弹窗", request_id)

    def remember_perm_decision(self, request_id: str, decision: str) -> str:
        """记录审批结果到缓存,并返回最新 rid(用于回应 ZCode)。

        由上层在收到用户决策后调用:缓存结果 + 取最新 rid。
        之后 reannounce 重发会直接用缓存回应。
        """
        result = {"decision": decision}
        self._perm_cache[request_id] = result
        rid = self._perm_pending.pop(request_id, "")
        return rid

    def clear_perm_state(self) -> None:
        """清理 permission 缓存(turn 结束时调用,避免内存堆积)。"""
        self._perm_pending.clear()
        self._perm_cache.clear()

    def _maybe_auto_respond(self, msg: dict) -> None:
        """对服务端反向请求做默认应答,避免会话卡死。

        yolo 模式下一般不会触发 permission;此处兜底:
        - permission.requested → allow
        - userInput.requested   → 空字符串(跳过)
        其余忽略。
        """
        method = msg.get("method", "")
        rid = msg.get("id")
        if rid is None:
            return
        if method == "interaction/requestPermission":
            logger.info("自动应答 permission.requested → allow")
            asyncio.ensure_future(self.respond(rid, {"decision": "allow"}))
        elif method == "interaction/requestUserInput":
            logger.info("自动应答 userInput.requested → skip")
            asyncio.ensure_future(self.respond(rid, {"response": ""}))
        # 其余反向请求(requestProviderRuntimeHeaders 等)忽略
