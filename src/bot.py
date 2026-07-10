"""Telegram Bot 主逻辑。

收到用户消息 → 查/建 sessionId → 调 ZCode → 推回 response。

两种后端:
- app-server 模式(默认,USE_APP_SERVER=1):通过 SessionSync 与 `zcode app-server`
  通信,流式输出 assistant 回复;后台轮询发现 TUI 在同 session 的活动并推送。
- --prompt 模式(回退,USE_APP_SERVER=0):走 ZCodeClient(--prompt --json)。
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Optional

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from app_server_client import (
    AppServerClient,
    AppServerError,
    PromptAlreadyRunningError,
    SessionUnavailableError,
    StreamEvent,
)
from config import Config
from session_store import SessionStore
from session_sync import SessionSync
from zcode_client import ZCodeClient, ZCodeError

logger = logging.getLogger("zcode-tg-bot")


class ZCodeTelegramBot:
    """串联 Telegram <-> ZCode 的 bot。"""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.store = SessionStore(config.db_path)
        # app-server 同步层(默认)
        self.sync: Optional[SessionSync] = None
        # 旧 --prompt 回退
        self.zcode = ZCodeClient(
            working_dir=config.approved_directory,
            timeout=config.zcode_timeout,
            cli_path=config.zcode_cli_path or None,
        )
        self._zcode_lock = asyncio.Lock()
        # session key → (chat_id, thread_id):轮询发现 TUI 活动时,知道往哪推
        self._push_targets: dict[str, tuple[int, Optional[int]]] = {}
        # session_id → asyncio.Lock:群组共享 session 时,串行化同 session 的 prompt(排队)
        self._session_locks: dict[str, asyncio.Lock] = {}
        # 缓存 bot 自己的 user id + username(用于检测"回复 bot"和"@本bot")
        self._bot_user_id: Optional[int] = None
        self._bot_username: str = ""
        # permission 审批:request_id → asyncio.Future(用户点按钮时 set_result)
        self._perm_futures: dict[str, asyncio.Future] = {}

    # ---------- 应用装配 ----------

    def build_application(self) -> Application:
        app = (
            Application.builder()
            .token(self.config.telegram_token)
            .post_init(self.post_init)
            .post_shutdown(self.on_shutdown)
            .build()
        )
        app.add_handler(CommandHandler("start", self.cmd_start))
        app.add_handler(CommandHandler("new", self.cmd_new))
        app.add_handler(CommandHandler("id", self.cmd_id))
        app.add_handler(CommandHandler("sessions", self.cmd_sessions))
        app.add_handler(CommandHandler("sync", self.cmd_sync))
        app.add_handler(CallbackQueryHandler(self.on_session_pick, pattern="^sync:"))
        app.add_handler(CallbackQueryHandler(self.on_permission_decision, pattern="^perm:"))
        app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.on_message)
        )
        return app

    async def post_init(self, application: Application) -> None:
        """Application 启动后初始化 app-server 同步层。"""
        if self.config.use_app_server:
            client = AppServerClient(cwd=self.config.workspace_path)
            self.sync = SessionSync(client, poll_interval=self.config.poll_interval)
            await self.sync.start()
            logger.info("✅ app-server 同步层已就绪")
        else:
            logger.info("ℹ️ USE_APP_SERVER=0,使用 --prompt 回退模式")

    async def on_shutdown(self, application: Application) -> None:
        if self.sync:
            await self.sync.stop()

    # ---------- 鉴权 ----------

    def _chat_type(self, update: Update) -> str:
        """返回 chat 类型:private / group / supergroup / channel。"""
        chat = update.effective_chat
        return chat.type if chat else "private"

    def _is_group(self, update: Update) -> bool:
        """是否群组消息(group / supergroup)。"""
        return self._chat_type(update) in ("group", "supergroup")

    def _is_allowed(self, update: Update) -> bool:
        """鉴权(双轨制)。

        - 私聊:发言者 user_id 在 allowed_users 白名单
        - 群组:群 chat_id 在 allowed_chats 白名单(群内任何人可用)
        """
        if self._is_group(update):
            chat_id = update.effective_chat.id
            return chat_id in self.config.allowed_chats
        # 私聊
        user = update.effective_user
        return bool(user and user.id in self.config.allowed_users)

    def _session_key(self, update: Update) -> str:
        """session 归属的 key。

        论坛群的话题:用 "chat_id:thread_id"(每话题独立 session);
        普通群(没开话题):用 chat_id(群内共享);
        私聊:用 user_id(每用户独立)。
        """
        chat = update.effective_chat
        if chat and self._is_group(update):
            chat_id = chat.id
            # 论坛群:按话题粒度分 session
            if getattr(chat, "is_forum", False):
                msg = update.effective_message
                # 话题内消息有 message_thread_id;General 顶层消息可能为 None → 归到 General(thread_id=1)
                thread_id = getattr(msg, "message_thread_id", None) if msg else None
                if not thread_id:
                    thread_id = 1  # General 话题
                return f"{chat_id}:{thread_id}"
            return str(chat_id)
        return str(update.effective_user.id)

    def _remember_push_target(self, update: Update) -> None:
        """从 update 记录推送目标 (chat_id, thread_id),供轮询推送用。

        论坛群按话题记 thread_id(General 顶层归 1);普通群/私聊 thread_id=None。
        """
        msg = update.effective_message
        chat = update.effective_chat
        if not msg or not chat or not msg.chat_id:
            return
        is_group = self._is_group(update)
        thread_id = getattr(msg, "message_thread_id", None)
        if is_group and getattr(chat, "is_forum", False):
            if not thread_id:
                thread_id = 1  # General 话题
        else:
            thread_id = None
        key = self._session_key(update)
        self._push_targets[key] = (msg.chat_id, thread_id)

    def _get_session_lock(self, session_id: str) -> asyncio.Lock:
        """取/建 per-session 的异步锁(群组共享 session 时排队用)。"""
        if session_id not in self._session_locks:
            self._session_locks[session_id] = asyncio.Lock()
        return self._session_locks[session_id]

    # ---------- 命令 ----------

    async def cmd_start(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        backend = "app-server(流式 + 双向同步)" if self.sync else "--prompt(回退)"
        await update.message.reply_text(
            "👋 你好!我是 ZCode Telegram Bot。\n\n"
            "直接发消息给我,我会转发给 ZCode Agent 执行并返回结果。\n\n"
            f"后端:{backend}\n\n"
            "命令:\n"
            "/new — 开启新会话\n"
            "/sessions — 列出本地 TUI session,选一个关联(双向同步)\n"
            "/sync <sessionId> — 直接关联指定 session\n"
            "/id — 查看你的 Telegram user id"
        )

    async def cmd_new(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_allowed(update):
            return
        key = self._session_key(update)
        if self.sync:
            try:
                sid = await self.sync.client.create_session(
                    self.config.workspace_path, mode=self.config.session_mode
                )
                await self.sync.client.subscribe(sid)
                self.store.set(key, sid)
                self._remember_push_target(update)
                self._setup_watch(key, sid)
                scope = "群组共享" if self._is_group(update) else "个人"
                await update.message.reply_text(
                    f"🔄 已开启新会话({scope}):`{sid[:24]}…`\n下一条消息将从头开始。"
                )
            except AppServerError as e:
                await update.message.reply_text(f"❌ 创建会话失败: {e}")
        else:
            self.store.reset(key)
            await update.message.reply_text("🔄 已开启新会话。下一条消息将从头开始。")

    async def cmd_id(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        user = update.effective_user
        if user:
            await update.message.reply_text(
                f"你的 Telegram user id: `{user.id}`", parse_mode=ParseMode.MARKDOWN
            )

    async def cmd_sessions(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """列出本地 session,内联键盘选择关联。

        直接读 SQLite(高效、含消息数和精确时间),避免逐个 resume 的开销。
        """
        if not self._is_allowed(update):
            return
        rows = _query_local_sessions()
        if not rows:
            await update.message.reply_text("📭 没有可关联的 session。用 /new 新建一个?")
            return
        # 内联键盘:标题 · 精确时间 · 消息数
        keyboard = []
        for r in rows[:8]:  # Telegram 限制按钮数
            label = f"{r['time']} · {r['title'][:18]} · {r['msg_count']}条"
            keyboard.append(
                [InlineKeyboardButton(label, callback_data=f"sync:{r['session_id']}")]
            )
        await update.message.reply_text(
            "选择要关联的 session(关联后双向同步):\n格式:时间 · 标题 · 消息数",
            reply_markup=InlineKeyboardMarkup(keyboard),
        )

    async def cmd_sync(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """直接关联指定 session:/sync <sessionId>。"""
        if not self._is_allowed(update):
            return
        if not self.sync:
            await update.message.reply_text("❌ 当前为 --prompt 回退模式,不支持此命令。")
            return
        if not ctx.args:
            await update.message.reply_text("用法:`/sync <sessionId>`", parse_mode=ParseMode.MARKDOWN)
            return
        sid = ctx.args[0].strip()
        if not sid.startswith("sess_"):
            await update.message.reply_text("❌ sessionId 应以 `sess_` 开头。")
            return
        self._remember_push_target(update)
        await self._do_sync(update, self._session_key(update), sid)

    async def on_session_pick(
        self, update: Update, ctx: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """内联键盘选择 session 后的回调。"""
        query = update.callback_query
        await query.answer()
        # callback_query 用 update 整体鉴权(chat 在 query.message.chat)
        if not self._is_allowed(update):
            await query.edit_message_text("🔒 无权操作。")
            return
        sid = query.data.split("sync:", 1)[1]
        await query.edit_message_text(f"⏳ 关联 {sid[:20]}… 中")
        self._remember_push_target(update)
        await self._do_sync(query, self._session_key(update), sid, edit=True)

    async def _do_sync(
        self, update, key: int, sid: str, edit: bool = False
    ) -> None:
        """关联指定 session:resume + subscribe + 建 watcher。

        key 是 session 归属键(群组=chat_id,私聊=user_id)。
        """
        assert self.sync
        try:
            await self.sync.client.resume_session(sid)
            sub = await self.sync.client.subscribe(sid)
            last_seq = sub.get("eventSeq", 0)
            self.store.set(key, sid)
            self.store.set_last_seq(key, last_seq)
            self._setup_watch(key, sid)
            msg = (
                f"✅ 已关联 `{sid[:24]}…`\n"
                f"现在 TUI 和 bot 双向同步该 session。"
            )
        except SessionUnavailableError:
            msg = f"❌ session 不存在或已关闭: `{sid[:24]}…`"
        except AppServerError as e:
            msg = f"❌ 关联失败: {e}"
        if edit:
            await update.edit_message_text(msg, parse_mode=ParseMode.MARKDOWN)
        else:
            await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)

    def _setup_watch(self, key: str, sid: str) -> None:
        """为一个 session 建立 TUI→bot 轮询 watcher。

        key 是 session 归属键(论坛话题="chat_id:thread_id";普通群=chat_id;私聊=user_id)。
        """
        if not self.sync:
            return
        last_seq = self.store.get_last_seq(key)

        def on_events(session_id: str, events: list) -> None:
            # 在 loop 线程触发推送
            asyncio.ensure_future(self._push_tui_updates(key, session_id, events))

        self.sync.watch(sid, last_seq, on_events)

    def _push_target(self, key: str) -> Optional[tuple[int, Optional[int]]]:
        """从 session key 拿推送目标 (chat_id, thread_id)。

        论坛话题/普通群/私聊 都从 _push_targets 查;查不到返回 None。
        """
        return self._push_targets.get(key)

    async def _push_tui_updates(
        self, key: str, session_id: str, events: list
    ) -> None:
        """轮询发现 TUI 新活动时,推送到对应会话(群话题或私聊)。"""
        target = self._push_target(key)
        if not target:
            return
        chat_id, thread_id = target
        # 汇总:把有意义的事件格式化成一条消息
        lines = ["📥 [TUI 端有新活动]"]
        for e in events:
            t = e.get("type")
            p = e.get("payload", {})
            if t == "turn.started":
                lines.append("▶️ 开始新一轮")
            elif t == "turn.completed":
                resp = p.get("response", "")
                tokens = p.get("tokenCount", 0)
                lines.append(f"✅ 完成:{_truncate(resp, 300)}\n_💎 {tokens} tokens_")
            elif t == "turn.failed":
                lines.append(f"❌ 失败:{p.get('error', {}).get('message', '?')}")
            elif t == "tool.updated":
                kind = p.get("kind", "")
                tool = p.get("toolName", "工具")
                if kind in ("started", "scheduled"):
                    lines.append(f"🔧 调用 {tool}")
            elif t == "message.upserted":
                content = p.get("content", "")
                role = p.get("type", "")
                if content:
                    lines.append(f"💬 [{role}] {_truncate(content, 200)}")
        try:
            await ctx_safe_send(chat_id, "\n".join(lines), message_thread_id=thread_id)
        except Exception:
            logger.exception("推送 TUI 更新失败")

    # ---------- 核心消息处理 ----------

    async def on_message(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        user = update.effective_user
        msg = update.message
        if not user or not msg or not msg.text:
            return

        is_group = self._is_group(update)

        # 群组:所有非命令文本都触发(无需 @bot)
        # 过滤其他 bot 发的消息,避免 bot 间互触发
        if is_group:
            if user.is_bot:
                return
            logger.info("群消息触发: user=%s", user.id)

        # 鉴权(双轨)
        if not self._is_allowed(update):
            if not is_group:
                # 私聊:提示无权
                await msg.reply_text(
                    f"🔒 无权访问。你的 user id 是 {user.id},请联系管理员添加。"
                )
            # 群组:静默忽略(不在群里公开拒绝,避免刷屏/泄露)
            return

        # 清洗 prompt:剥掉 @botname 前缀(群组场景)
        prompt = await self._strip_mention(msg, ctx)
        prompt = prompt.strip()
        if not prompt:
            return

        # 记录推送目标(轮询发现 TUI 活动时,知道往哪个 chat/话题推)
        self._remember_push_target(update)

        key = self._session_key(update)
        logger.info("user=%s chat=%s group=%s prompt=%r", user.id, msg.chat_id, is_group, prompt[:80])

        if self.sync:
            await self._handle_app_server(update, key, prompt)
        else:
            await self._handle_prompt_mode(update, key, prompt)

    async def _is_group_trigger(
        self, update: Update, ctx: ContextTypes.DEFAULT_TYPE
    ) -> tuple[bool, str]:
        """群组里是否应该触发:@bot 提及 或 回复 bot 的消息。

        返回 (是否触发, 原因)。bot 若是群管理员会收到所有消息,靠此过滤。
        """
        msg = update.message
        # 1. 回复 bot 自己的消息(reply_to 指向 bot 发的消息)
        if msg.reply_to_message:
            bot_id = await self._get_bot_id(ctx)
            replied = msg.reply_to_message.from_user
            if replied and replied.id == bot_id:
                return True, "reply_to_bot"
        # 2. @本 bot 提及(entities 里的 MENTION 文本必须是 @自己的 username)
        #    不能只看"有没有 MENTION",否则 @别的 bot 也会误触发
        if msg.entities and msg.text:
            from telegram.constants import MessageEntityType
            bot_username = await self._get_bot_username(ctx)
            for ent in msg.entities:
                if ent.type == MessageEntityType.MENTION:
                    mention_text = msg.text[ent.offset:ent.offset + ent.length]
                    # mention_text 形如 "@ZCode_RJDJ_bot",不区分大小写比对
                    if mention_text.lower() == f"@{bot_username.lower()}":
                        return True, "mention"
        return False, "none"

    async def _get_bot_id(self, ctx: ContextTypes.DEFAULT_TYPE) -> int:
        """获取 bot 自己的 user id(缓存)。用于检测"回复 bot 的消息"。"""
        await self._ensure_bot_info(ctx)
        return self._bot_user_id

    async def _get_bot_username(self, ctx: ContextTypes.DEFAULT_TYPE) -> str:
        """获取 bot 自己的 username(缓存,无 @ 前缀)。用于检测 @本bot 提及。"""
        await self._ensure_bot_info(ctx)
        return self._bot_username

    async def _ensure_bot_info(self, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """一次性获取并缓存 bot 的 id + username。"""
        if self._bot_user_id is None:
            me = await ctx.bot.get_me()
            self._bot_user_id = me.id
            self._bot_username = me.username or ""

    async def _strip_mention(self, msg, ctx: ContextTypes.DEFAULT_TYPE) -> str:
        """剥掉消息开头的 @botname 前缀,返回干净的 prompt 文本。"""
        text = msg.text or ""
        # 只在有 mention entity 时处理
        from telegram.constants import MessageEntityType
        if not msg.entities:
            return text
        for ent in msg.entities:
            if ent.type == MessageEntityType.MENTION:
                mention = msg.text[ent.offset:ent.offset + ent.length]  # 如 @zcode_bot
                # 去掉开头的 mention(可能带或不带尾随空格)
                text = re.sub(rf'^{re.escape(mention)}\s*', '', text)
                break
        return text

    async def _handle_app_server(
        self, update: Update, key: int, prompt: str
    ) -> None:
        """app-server 模式:流式输出。群组共享 session 时排队。"""
        assert self.sync
        session_id = self.store.get(key)
        # 没有 session 先建一个
        if not session_id:
            try:
                session_id = await self.sync.client.create_session(
                    self.config.workspace_path, mode=self.config.session_mode
                )
                await self.sync.client.subscribe(session_id)
                self.store.set(key, session_id)
                self._setup_watch(key, session_id)
            except AppServerError as e:
                await update.message.reply_text(f"❌ 创建会话失败: {e}")
                return

        # 群组共享 session:排队(同一 session 串行执行)
        lock = self._get_session_lock(session_id)
        if lock.locked():
            # 已有人在跑,提示排队
            await update.message.reply_text("⏳ 前面有任务在跑,排队中…")
        async with lock:
            await self._run_app_server_turn(update, session_id, prompt)

    async def _run_app_server_turn(
        self, update: Update, session_id: str, prompt: str
    ) -> None:
        """单个 turn 的流式执行(在 session 锁内)。"""
        assert self.sync
        # 先发"思考中"占位
        placeholder = await update.message.reply_text("⏳ 思考中...")
        chat_id = placeholder.chat_id

        accumulated = ""
        last_edit = 0.0
        total_tokens = 0
        error = None
        EDIT_MIN_INTERVAL = 2.0  # Telegram 同条消息 edit 限流

        try:
            async for ev in self.sync.send_stream(
                session_id, prompt, turn_timeout=self.config.app_server_turn_timeout
            ):
                if ev.type == "model.streaming" and ev.payload.get("kind") in (
                    "text_delta",
                    "text_start",
                ):
                    delta = ev.payload.get("delta") or ""
                    if delta:
                        accumulated += delta
                        # 节流 edit
                        now = time.time()
                        if now - last_edit >= EDIT_MIN_INTERVAL and accumulated:
                            try:
                                await ctx_safe_edit(
                                    chat_id, placeholder.message_id,
                                    f"⏳ {accumulated}",
                                )
                                last_edit = now
                            except Exception:
                                pass
                elif ev.type == "turn.completed":
                    # 权威最终文本
                    final = ev.payload.get("response", "")
                    if final:
                        accumulated = final
                    total_tokens = ev.payload.get("tokenCount", 0)
                elif ev.type == "turn.failed":
                    error = ev.payload.get("error", {}).get("message", "未知错误")
                elif ev.type == "permission.requested":
                    # build 模式:ZCode 请求审批写操作/命令 → 转发到 Telegram
                    await self._handle_permission_event(update, chat_id, placeholder, ev.payload)
                # tool.updated 静默(避免刷屏);可在此追加展示
        except PromptAlreadyRunningError:
            await ctx_safe_edit(
                chat_id, placeholder.message_id,
                "⏳ 该 session 已有任务在跑(TUI 或上一条),请稍后再试。",
            )
            return
        except SessionUnavailableError:
            await ctx_safe_edit(
                chat_id, placeholder.message_id,
                "❌ session 已失效,请用 /new 或 /sessions 重新关联。",
            )
            self.store.reset(self._session_key(update))
            return
        except asyncio.TimeoutError:
            await ctx_safe_edit(chat_id, placeholder.message_id, "❌ 等待回复超时。")
            return
        except AppServerError as e:
            await ctx_safe_edit(chat_id, placeholder.message_id, f"❌ {e}")
            return
        finally:
            # 清理本 turn 的 permission 审批状态(避免内存堆积)
            self.sync.client.clear_perm_state()
            # 清理未决的 future(取消等待)
            for fut in self._perm_futures.values():
                if not fut.done():
                    fut.cancel()
            self._perm_futures.clear()

        # 最终输出:删除占位,发完整结果(可能需要拆分)
        try:
            await ctx_safe_delete(chat_id, placeholder.message_id)
        except Exception:
            pass

        if error:
            await update.message.reply_text(f"❌ {error}")
            return
        if not accumulated.strip():
            accumulated = "(ZCode 返回空内容)"
        footer = f"\n\n_💎 {total_tokens} tokens_" if total_tokens else ""
        await self._send_chunked(update, accumulated, footer)

    # ---------- permission 审批转发(build 模式)----------

    async def _handle_permission_event(
        self, update: Update, chat_id: int, placeholder, payload: dict
    ) -> None:
        """收到 ZCode 的 permission.requested 事件 → 发 Telegram 审批按钮,等用户决策。

        审批按钮用 send_message 发送,在论坛群里必须带 message_thread_id,
        否则会落到 General 话题而不是当前话题。
        """
        assert self.sync
        request_id = payload.get("request_id", "")
        tool_name = payload.get("tool_name", "工具")
        risk = payload.get("risk_level", "")
        risk_label = {"low": "低风险", "medium": "中风险", "high": "高风险", "critical": "极高风险"}.get(risk, risk)
        input_data = payload.get("input", {})

        # 从 input 提取关键信息(文件路径 / 命令内容)
        detail = _format_perm_input(tool_name, input_data)

        # 建 future(等用户点按钮)
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        self._perm_futures[request_id] = fut

        # 发审批消息(带按钮)
        keyboard = [
            [
                InlineKeyboardButton("✅ 批准", callback_data=f"perm:allow:{request_id}"),
                InlineKeyboardButton("❌ 拒绝", callback_data=f"perm:deny:{request_id}"),
            ]
        ]
        msg_text = (
            f"🔐 需要审批\n"
            f"工具:{tool_name}({risk_label})\n"
            f"{detail}"
        )
        app = _bot_ref.get("app")
        if app and app.bot:
            # 论坛群:审批消息要落到当前话题,否则掉到 General
            thread_id = None
            chat = update.effective_chat
            msg = update.effective_message
            if chat and getattr(chat, "is_forum", False) and msg:
                thread_id = getattr(msg, "message_thread_id", None) or 1
            await app.bot.send_message(
                chat_id=chat_id, text=msg_text, message_thread_id=thread_id,
                reply_markup=InlineKeyboardMarkup(keyboard)
            )

        # 等用户决策(不超时,符合需求)
        decision = await fut
        # 缓存结果 + 取最新 rid(reannounce 可能更新过)
        rid = self.sync.client.remember_perm_decision(request_id, decision)
        # 回应 ZCode
        result = {"decision": decision}
        if decision == "deny":
            result["reason"] = "用户在 Telegram 拒绝了该操作"
        await self.sync.client.respond(rid, result)
        logger.info("permission 审批完成(requestId=%s → %s)", request_id, decision)

    async def on_permission_decision(
        self, update: Update, ctx: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """审批按钮回调:用户点了批准/拒绝。"""
        query = update.callback_query
        await query.answer()
        if not self._is_allowed(update):
            await query.edit_message_text("🔒 无权操作。")
            return
        # 解析 callback_data: perm:allow:<request_id> / perm:deny:<request_id>
        parts = query.data.split(":", 2)
        if len(parts) != 3:
            return
        decision = parts[1]  # "allow" / "deny"
        request_id = parts[2]

        fut = self._perm_futures.get(request_id)
        if fut and not fut.done():
            fut.set_result(decision)
            label = "✅ 已批准" if decision == "allow" else "❌ 已拒绝"
            await query.edit_message_text(f"{query.message.text}\n\n→ {label}")
        else:
            await query.edit_message_text("⚠️ 该审批已处理或已过期。")

    async def _handle_prompt_mode(
        self, update: Update, key: int, prompt: str
    ) -> None:
        """旧 --prompt 回退模式(USE_APP_SERVER=0)。"""
        session_id = self.store.get(key)
        processing = await update.message.reply_text("⏳ ZCode 思考中...")
        zcode_task = asyncio.ensure_future(self._call_zcode(prompt, session_id))
        heartbeat = asyncio.ensure_future(self._heartbeat(processing))
        try:
            result = await zcode_task
        except ZCodeError as e:
            heartbeat.cancel()
            await processing.edit_text(f"❌ 执行出错:\n```\n{e}\n```")
            return
        except Exception as e:
            heartbeat.cancel()
            logger.exception("unexpected error")
            await processing.edit_text(f"❌ 意外错误: {e}")
            return
        finally:
            heartbeat.cancel()
        if result.session_id and result.session_id != session_id:
            self.store.set(key, result.session_id)
        try:
            await processing.delete()
        except Exception:
            pass
        footer = f"\n\n_💎 {result.total_tokens} tokens_"
        await self._send_chunked(update, result.response.strip(), footer)

    async def _call_zcode(self, prompt: str, session_id: Optional[str]):
        async with self._zcode_lock:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, lambda: self.zcode.run(prompt, session_id=session_id)
            )

    async def _heartbeat(self, message) -> None:
        t0 = time.time()
        while True:
            await asyncio.sleep(4)
            elapsed = int(time.time() - t0)
            try:
                await message.edit_text(f"⏳ ZCode 思考中...({elapsed}s)")
            except Exception:
                pass

    async def _send_chunked(self, update: Update, text: str, footer: str) -> None:
        """发送结果,自动拆分超长消息。"""
        max_len = self.config.max_message_length
        chunks = _split_message(text, max_len - len(footer))
        for i, chunk in enumerate(chunks):
            suffix = footer if i == len(chunks) - 1 else ""
            try:
                await update.message.reply_text(chunk + suffix)
            except Exception:
                await update.message.reply_text(chunk + suffix)


# ---------- 辅助:绕过 ctx 的 bot 发送(轮询推送时无 ctx)----------

_bot_ref: dict = {}


def set_bot_instance(bot_app) -> None:
    """main 启动时存下 Application,供无 ctx 场景(轮询推送)发消息。"""
    _bot_ref["app"] = bot_app


async def ctx_safe_send(
    chat_id: int, text: str, message_thread_id: Optional[int] = None
) -> None:
    app = _bot_ref.get("app")
    if app and app.bot:
        await app.bot.send_message(
            chat_id=chat_id, text=text, message_thread_id=message_thread_id
        )


async def ctx_safe_edit(chat_id: int, message_id: int, text: str) -> None:
    app = _bot_ref.get("app")
    if app and app.bot:
        await app.bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=text)


async def ctx_safe_delete(chat_id: int, message_id: int) -> None:
    app = _bot_ref.get("app")
    if app and app.bot:
        await app.bot.delete_message(chat_id=chat_id, message_id=message_id)


# ---------- 文本工具 ----------


def _query_local_sessions() -> list[dict]:
    """直接从 SQLite 查本地 session 列表(高效,不走 app-server)。

    返回 [{session_id, title, time, msg_count}],按更新时间倒序。
    用精确时间(时分)区分,因为很多 session 标题会重复(如都叫 "pong")。
    """
    import os
    import sqlite3
    from datetime import datetime

    db_path = os.path.expanduser("~/.zcode/cli/db/db.sqlite")
    if not os.path.isfile(db_path):
        return []
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT s.id AS session_id,
                   s.title AS title,
                   s.time_updated AS updated,
                   (SELECT count(*) FROM message m WHERE m.session_id = s.id) AS msg_count
            FROM session s
            WHERE s.task_type = 'interactive' AND s.time_archived IS NULL
            ORDER BY s.time_updated DESC
            LIMIT 20
            """
        ).fetchall()
        conn.close()
    except sqlite3.Error:
        logger.exception("读取本地 session 列表失败")
        return []
    result = []
    for r in rows:
        ts = datetime.fromtimestamp(r["updated"] / 1000).strftime("%m-%d %H:%M")
        title = r["title"] or "(无标题)"
        result.append({
            "session_id": r["session_id"],
            "title": title,
            "time": ts,
            "msg_count": r["msg_count"],
        })
    return result


def _format_perm_input(tool_name: str, input_data: dict) -> str:
    """把工具入参格式化成审批消息里的可读详情。"""
    if not isinstance(input_data, dict):
        return ""
    if tool_name == "Bash":
        cmd = input_data.get("command", "")
        return f"命令:{_truncate(cmd, 200)}"
    if tool_name in ("Edit", "Write"):
        path = input_data.get("file_path", "")
        if tool_name == "Edit":
            old = input_data.get("old_string", "")
            return f"文件:{path}\n操作:替换 {_truncate(old, 100)}"
        content = input_data.get("content", "")
        return f"文件:{path}\n操作:写入 {_truncate(content, 100)}"
    # 通用:显示前几个 key
    parts = [f"{k}={_truncate(str(v), 60)}" for k, v in list(input_data.items())[:3]]
    return "参数:" + " ".join(parts) if parts else ""


def _truncate(s: str, n: int) -> str:
    s = s.strip()
    return s if len(s) <= n else s[:n] + "…"


def _time_ago(ms_epoch: int) -> str:
    if not ms_epoch:
        return ""
    diff = int(time.time() - ms_epoch / 1000)
    if diff < 60:
        return f"{diff}s前"
    if diff < 3600:
        return f"{diff // 60}m前"
    if diff < 86400:
        return f"{diff // 3600}h前"
    return f"{diff // 86400}d前"


def _split_message(text: str, max_len: int) -> list[str]:
    """把长文本按 max_len 拆成多段(尽量在换行处断)。"""
    if len(text) <= max_len:
        return [text]
    chunks: list[str] = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break
        cut = text.rfind("\n", 0, max_len)
        if cut < max_len // 2:
            cut = max_len
        chunks.append(text[:cut])
        text = text[cut:].lstrip("\n")
    return chunks
