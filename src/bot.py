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
from telegram.constants import ChatAction, ParseMode
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
        # 正在跑的 turn:session_id → asyncio.Task(供 /cancel 取消)
        self._running_turns: dict[str, asyncio.Task] = {}
        # /model 列表缓存:key → [(providerId, modelId, label), ...](callback_data 用索引,避免超 64 字节)
        self._model_lists: dict[str, list[tuple[str, str, str]]] = {}

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
        app.add_handler(CommandHandler("model", self.cmd_model))
        app.add_handler(CommandHandler("stop", self.cmd_stop))
        app.add_handler(CallbackQueryHandler(self.on_session_pick, pattern="^sync:"))
        app.add_handler(CallbackQueryHandler(self.on_permission_decision, pattern="^perm:"))
        app.add_handler(CallbackQueryHandler(self.on_cancel, pattern="^cancel:"))
        app.add_handler(CallbackQueryHandler(self.on_model_pick, pattern="^model:"))
        app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.on_message)
        )
        return app

    async def post_init(self, application: Application) -> None:
        """Application 启动后初始化 app-server 同步层。"""
        if self.config.use_app_server:
            client = AppServerClient(cwd=self.config.workspace_path)
            self.sync = SessionSync(client, poll_interval=self.config.poll_interval)
            # 崩溃自愈钩子:app-server 退出时清 perm/cancel turn/暂停轮询,
            # 重启成功后恢复轮询。
            client.on_unavailable = self._on_appserver_unavailable
            client.on_recovered = self._on_appserver_recovered
            await self.sync.start()
            # 重启后 resume 已存 session 的 watcher(D)
            await self._resume_persisted_sessions()
            logger.info("✅ app-server 同步层已就绪")
        else:
            logger.info("ℹ️ USE_APP_SERVER=0,使用 --prompt 回退模式")

    async def _on_appserver_unavailable(self) -> None:
        """app-server 崩溃时:暂停轮询 + 清 perm + cancel 所有 turn,避免死锁。"""
        if self.sync:
            self.sync.pause()
        # 所有 pending permission 按 deny resolve(解锁卡在 await fut 的 turn)
        for rid in list(self._perm_futures.keys()):
            await self._resolve_perm(rid, "deny", "app-server 重启")
        # cancel 所有正在跑的 turn(清理僵尸 handler,释放 session 锁)
        for sid, task in list(self._running_turns.items()):
            if not task.done():
                task.cancel()

    async def _on_appserver_recovered(self) -> None:
        """app-server 重启成功后:恢复轮询。"""
        if self.sync:
            self.sync.resume()

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
        # 持久化推送目标,重启 resume 后 TUI→bot 推送能立即用
        self.store.set_push_target(key, msg.chat_id, thread_id)

    async def _resume_persisted_sessions(self) -> None:
        """重启后恢复所有已存 session 的 watcher + 推送目标。

        bot 重启后 _watchers/_push_targets 内存全丢,但 sessions.json 还在。
        遍历它 resume+subscribe+建 watcher,让 TUI→bot 推送不哑火。
        失效的 session(session 表里已删/归档)静默清掉 store 这条。
        """
        if not self.sync:
            return
        entries = self.store.iter_all()
        seen_sids: set[str] = set()  # 同 sid 多 key 只 resume/subscribe 一次
        for key, entry in entries:
            sid = entry.get("session_id")
            if not sid or sid in seen_sids:
                continue
            seen_sids.add(sid)
            try:
                last_seq = entry.get("last_seq", 0)
                await self.sync.client.resume_session(sid)
                await self.sync.client.subscribe(sid, after_seq=last_seq)
            except Exception:
                logger.warning("重启恢复 session %s 失败,清理该 key", sid[:24])
                self.store.reset(key)
                continue
            logger.info("已恢复 session %s 的监听", sid[:24])
        # 为每个 key 重建 _push_targets(从持久化的 chat_id/thread_id)
        for key, entry in self.store.iter_all():
            chat_id = entry.get("chat_id")
            if chat_id:
                self._push_targets[key] = (chat_id, entry.get("thread_id"))
            # 重建 watcher(用最新水位)
            sid = entry.get("session_id")
            if sid and sid in seen_sids:
                last_seq = entry.get("last_seq", 0)
                self._setup_watch(key, sid)

    def _get_session_lock(self, session_id: str) -> asyncio.Lock:
        """取/建 per-session 的异步锁(群组共享 session 时排队用)。"""
        if session_id not in self._session_locks:
            self._session_locks[session_id] = asyncio.Lock()
        return self._session_locks[session_id]

    # ---------- 命令 ----------

    async def cmd_start(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_allowed(update):
            return
        backend = "app-server(流式 + 双向同步)" if self.sync else "--prompt(回退)"
        await update.message.reply_text(
            "👋 你好!我是 ZCode Telegram Bot。\n\n"
            "直接发消息给我,我会转发给 ZCode Agent 执行并返回结果。\n\n"
            f"后端:{backend}\n\n"
            "命令:\n"
            "/new — 开启新会话(论坛群会创建新话题)\n"
            "/sessions — 列出本地 TUI session,选一个关联(双向同步)\n"
            "/sync <sessionId> — 直接关联指定 session\n"
            "/model — 查看/切换当前会话模型\n"
            "/stop — 停止当前正在跑的任务\n"
            "/id — 查看你的 Telegram user id"
        )

    async def cmd_new(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_allowed(update):
            return
        chat = update.effective_chat
        is_forum_group = self._is_group(update) and getattr(chat, "is_forum", False)
        # 论坛群:/new 创建新话题 + 新会话绑定(每话题独立会话)
        if self.sync and is_forum_group:
            name = (ctx.args[0] if ctx.args else "") or time.strftime("会话 %m-%d %H:%M")
            try:
                topic = await ctx.bot.create_forum_topic(chat_id=chat.id, name=name)
                new_thread_id = topic.message_thread_id
                sid = await self.sync.client.create_session(
                    self.config.workspace_path, mode=self.config.session_mode
                )
                await self.sync.client.subscribe(sid)
                new_key = f"{chat.id}:{new_thread_id}"
                old_sid = self.store.get(new_key)
                if old_sid and old_sid != sid:
                    self.sync.unwatch(old_sid)
                self.store.set(new_key, sid)
                self._push_targets[new_key] = (chat.id, new_thread_id)
                self.store.set_push_target(new_key, chat.id, new_thread_id)
                self._setup_watch(new_key, sid)
                await ctx.bot.send_message(
                    chat_id=chat.id, message_thread_id=new_thread_id,
                    text=(
                        f"✅ 新话题已创建,新会话就绪:`{sid[:24]}…`\n"
                        f"在这个话题里直接发消息即可(无需 @bot)。"
                    ),
                    parse_mode=ParseMode.MARKDOWN,
                )
            except AppServerError as e:
                await update.message.reply_text(f"❌ 创建会话失败: {e}")
            except Exception as e:
                await update.message.reply_text(f"❌ 创建话题失败: {e}")
            return
        # 非论坛群/私聊:在当前会话里重置
        key = self._session_key(update)
        if self.sync:
            try:
                sid = await self.sync.client.create_session(
                    self.config.workspace_path, mode=self.config.session_mode
                )
                await self.sync.client.subscribe(sid)
                # 切新 session 前 unwatch 旧的,避免 watcher 堆积泄漏
                old_sid = self.store.get(key)
                if old_sid and old_sid != sid:
                    self.sync.unwatch(old_sid)
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
        if not self._is_allowed(update):
            return
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
            # 切新 session 前 unwatch 旧的,避免 watcher 堆积泄漏
            old_sid = self.store.get(key)
            if old_sid and old_sid != sid:
                self.sync.unwatch(old_sid)
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
        # 日志只记 prompt 长度,不打内容(用户消息可能含密钥/敏感信息)
        logger.info("user=%s chat=%s group=%s prompt_len=%d", user.id, msg.chat_id, is_group, len(prompt))

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
            # 把 turn 包成独立 Task,供 /cancel 取消(同 session 串行,同时只有一个)
            task = asyncio.ensure_future(self._run_app_server_turn(update, session_id, prompt))
            self._running_turns[session_id] = task
            try:
                await task
            finally:
                self._running_turns.pop(session_id, None)

    async def _run_app_server_turn(
        self, update: Update, session_id: str, prompt: str
    ) -> None:
        """单个 turn 的流式执行(在 session 锁内)。

        流式输出:边收 delta 边 edit 占位消息显示进度,turn 完成后短文本直接
        把占位 edit 成最终结果(连贯、省一条消息),超长才删占位发多条。
        期间持续发 typing 指示器,避免用户以为卡死。
        """
        assert self.sync
        # 先发"思考中"占位
        placeholder = await update.message.reply_text("⏳ 思考中...")
        chat_id = placeholder.chat_id
        thread_id = _thread_id_of(update)  # 论坛群话题,私聊/普通群为 None

        accumulated = ""
        last_edit = 0.0
        last_typing = 0.0
        total_tokens = 0
        error = None
        edit_interval = self.config.stream_edit_interval
        app = _bot_ref.get("app")
        # 占位消息上挂"取消"按钮(callback_data < 64 字节)
        cancel_markup = InlineKeyboardMarkup(
            [[InlineKeyboardButton("🚫 取消", callback_data=f"cancel:{session_id}")]]
        )

        async def _refresh_typing() -> None:
            """续 typing 指示器(过期 5s,这里 4s 续一次)。"""
            nonlocal last_typing
            now = time.time()
            if app and now - last_typing >= 4.0:
                try:
                    await app.bot.send_chat_action(
                        chat_id=chat_id,
                        action=ChatAction.TYPING,
                        message_thread_id=thread_id,
                    )
                    last_typing = now
                except Exception:
                    pass

        await _refresh_typing()  # 开头发一次

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
                        if now - last_edit >= edit_interval and accumulated:
                            try:
                                await ctx_safe_edit(
                                    chat_id, placeholder.message_id,
                                    _truncate(accumulated, self.config.max_message_length - 8),
                                    reply_markup=cancel_markup,
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
                await _refresh_typing()
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
        except asyncio.CancelledError:
            # 用户 /stop 或点取消:把占位改成"已停止"并释放锁
            try:
                await ctx_safe_edit(
                    chat_id, placeholder.message_id, "🚫 已停止。",
                    reply_markup=InlineKeyboardMarkup([]),
                )
            except Exception:
                pass
            raise
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

        # 最终输出:A1 短文本直接 edit 占位成最终内容(连贯、省一条);
        #         超长才删占位 + 发多条(_send_chunked)
        footer = _build_footer(total_tokens, self.config.workspace_path)
        max_single = self.config.max_message_length
        final_text = accumulated if accumulated.strip() else "(ZCode 返回空内容)"

        if error:
            # 出错:删占位,单独发错误(避免占位上残留进度)
            try:
                await ctx_safe_delete(chat_id, placeholder.message_id)
            except Exception:
                pass
            await update.message.reply_text(f"❌ {error}")
            return

        if len(final_text) + len(footer) <= max_single:
            # 短文本:直接 edit 占位成最终结果(去掉取消按钮)
            try:
                await ctx_safe_edit(
                    chat_id, placeholder.message_id, final_text + footer,
                    reply_markup=InlineKeyboardMarkup([]),
                )
                return
            except Exception:
                # edit 失败(消息被删/权限等)→ fallback 发新消息
                pass
        # 超长或 edit 失败:删占位 + 拆分发多条
        try:
            await ctx_safe_delete(chat_id, placeholder.message_id)
        except Exception:
            pass
        await self._send_chunked(update, final_text, footer)

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
        perm_msg = None
        if app and app.bot:
            # 论坛群:审批消息要落到当前话题,否则掉到 General
            thread_id = _thread_id_of(update)
            perm_msg = await app.bot.send_message(
                chat_id=chat_id, text=msg_text, message_thread_id=thread_id,
                reply_markup=InlineKeyboardMarkup(keyboard)
            )

        # 等用户决策:有超时兜底(防用户不点导致 session 锁死)
        # 超时/取消都按 deny 处理(安全侧偏保守),deny 后 ZCode 会推 turn.completed
        try:
            decision = await asyncio.wait_for(fut, timeout=self.config.perm_timeout)
        except asyncio.TimeoutError:
            logger.warning("permission 审批超时(requestId=%s),按拒绝兜底", request_id)
            await self._resolve_perm(request_id, "deny", "审批超时")
            # 编辑审批消息为"超时已拒绝"(按钮点按的 edit 由 on_permission_decision 处理)
            if perm_msg:
                try:
                    await perm_msg.edit_text(
                        f"{msg_text}\n\n→ ⏱ 超时已拒绝",
                        reply_markup=None,
                    )
                except Exception:
                    pass
            # 让控制流回到 async for:deny 后 ZCode 继续推进 turn
            return
        # 正常路径:缓存结果 + 取最新 rid(reannounce 可能更新过)
        await self._resolve_perm(request_id, decision)
        logger.info("permission 审批完成(requestId=%s → %s)", request_id, decision)

    async def _resolve_perm(
        self, request_id: str, decision: str, reason: str = ""
    ) -> None:
        """统一处理 permission 决策:resolve future + respond ZCode + edit 审批消息。

        用 pop 保证只 resolve 一次(防超时与按钮点按竞争导致重复 respond)。
        被 _handle_permission_event(正常/超时)和 on_cancel(取消整个 turn)共用。
        """
        assert self.sync
        fut = self._perm_futures.pop(request_id, None)
        if fut and not fut.done():
            fut.set_result(decision)
        # 缓存结果 + 取最新 rid(reannounce 可能更新过)
        rid = self.sync.client.remember_perm_decision(request_id, decision)
        result: dict = {"decision": decision}
        if decision == "deny":
            result["reason"] = reason or "用户在 Telegram 拒绝了该操作"
        elif reason:
            result["reason"] = reason
        try:
            await self.sync.client.respond(rid, result)
        except Exception:
            logger.exception("respond permission 失败(requestId=%s)", request_id)

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

    async def on_cancel(
        self, update: Update, ctx: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """占位消息上"🚫 取消"按钮回调:取消正在跑的 turn。

        先 stop_session 真正中断 app-server 那边的 turn(否则 session 还被占,
        下条消息撞 -32010),再 resolve pending permission + cancel bot task。
        """
        query = update.callback_query
        await query.answer()
        if not self._is_allowed(update):
            await query.edit_message_text("🔒 无权操作。")
            return
        # 解析 callback_data: cancel:<session_id>
        sid = query.data.split(":", 1)[1] if ":" in query.data else ""
        task = self._running_turns.get(sid)
        if not task or task.done():
            await query.edit_message_text("⚠️ 该任务已结束,无需取消。")
            return
        # 先停 app-server 那边的 turn(否则 session 还被占,下条消息撞 -32010)
        try:
            await self.sync.client.stop_session(sid)
        except Exception:
            logger.warning("stop_session 失败(sessionId=%s),仍 cancel bot 侧 task", sid[:24])
        # 再 resolve pending permission + cancel bot task
        for rid in list(self._perm_futures.keys()):
            await self._resolve_perm(rid, "deny", "用户取消了整个 turn")
        task.cancel()
        logger.info("用户取消 turn(sessionId=%s)", sid)

    async def cmd_stop(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """停止当前话题/会话正在跑的任务(/stop)。

        先 stop_session 真正中断 app-server 那边的 turn(否则 session 还被占),
        再 resolve pending permission + cancel bot task 停渲染释放锁。
        占位消息由 _run_app_server_turn 的 CancelledError 分支 edit 成"已停止"。
        """
        if not self._is_allowed(update):
            return
        key = self._session_key(update)
        sid = self.store.get(key)
        if not sid:
            await update.message.reply_text("❓ 当前没有会话。")
            return
        task = self._running_turns.get(sid)
        if not task or task.done():
            await update.message.reply_text("💤 当前没有在跑的任务。")
            return
        # 先停 app-server 那边的 turn(否则 session 还被占,下条消息撞 -32010)
        try:
            await self.sync.client.stop_session(sid)
        except Exception:
            logger.warning("stop_session 失败(sessionId=%s),仍 cancel bot 侧 task", sid[:24])
        # 再 resolve pending permission(否则 ZCode 那边 permission 永等)
        for rid in list(self._perm_futures.keys()):
            await self._resolve_perm(rid, "deny", "用户停止了任务")
        task.cancel()
        await update.message.reply_text("🚫 已停止当前任务。")
        logger.info("用户停止 turn(sessionId=%s)", sid)

    async def cmd_model(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """查看/切换当前会话模型:列出可用模型按钮,点选切换。"""
        if not self._is_allowed(update):
            return
        if not self.sync:
            await update.message.reply_text("❌ 当前为 --prompt 回退模式,不支持切模型。")
            return
        key = self._session_key(update)
        sid = self.store.get(key)
        if not sid:
            await update.message.reply_text("❓ 还没有会话,先发条消息或 /new 建一个。")
            return
        # 有 turn 在跑时不让切(避免打断)
        running = self._running_turns.get(sid)
        if running and not running.done():
            await update.message.reply_text("⏳ 当前有任务在跑,完成后再切模型。")
            return
        try:
            models = await self.sync.client.list_available_models()
        except AppServerError as e:
            await update.message.reply_text(f"❌ 读取模型列表失败: {e}")
            return
        if not models:
            await update.message.reply_text("📭 没有可用模型。")
            return
        # 缓存列表(按钮用索引,避免 callback_data 超 64 字节)
        items = []
        for m in models:
            ref = m.get("ref", {})
            pid = ref.get("providerId", "")
            mid = ref.get("modelId", "")
            plabel = m.get("providerLabel", "") or pid[:8]
            mlabel = m.get("label", "") or mid
            # 按钮文案:代理商名/模型名(区分同模型不同 provider)
            display = f"{plabel} · {mlabel}"
            items.append((pid, mid, display))
        self._model_lists[key] = items
        # 渲染键盘
        keyboard = []
        for i, (_pid, _mid, label) in enumerate(items):
            keyboard.append([InlineKeyboardButton(label, callback_data=f"model:{i}")])
        await update.message.reply_text(
            "选择要切换的模型:",
            reply_markup=InlineKeyboardMarkup(keyboard),
        )

    async def on_model_pick(
        self, update: Update, ctx: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """模型列表按钮回调:点选后切换当前会话模型。"""
        query = update.callback_query
        await query.answer()
        if not self._is_allowed(update):
            await query.edit_message_text("🔒 无权操作。")
            return
        # 解析 callback_data: model:<idx>
        idx_str = query.data.split(":", 1)[1] if ":" in query.data else ""
        key = self._session_key(update)
        items = self._model_lists.get(key)
        try:
            idx = int(idx_str)
        except (ValueError, TypeError):
            idx = -1
        if not items or idx < 0 or idx >= len(items):
            await query.edit_message_text("⚠️ 该列表已过期,请重新 /model。")
            return
        provider_id, model_id, label = items[idx]
        sid = self.store.get(key)
        if not sid:
            await query.edit_message_text("⚠️ 当前没有会话,先 /new。")
            return
        # 有 turn 在跑时拒绝
        running = self._running_turns.get(sid)
        if running and not running.done():
            await query.edit_message_text("⏳ 当前有任务在跑,完成后再切。")
            return
        await query.edit_message_text(f"⏳ 切换到 {label}…")
        await self._do_set_model(update, key, sid, provider_id, model_id, label)

    async def _do_set_model(
        self, update, key: str, sid: str,
        provider_id: str, model_id: str, label: str, edit: bool = True
    ) -> None:
        """在 session 锁内切换模型,成功回 ✅,失败回错误。"""
        assert self.sync
        lock = self._get_session_lock(sid)
        msg = ""
        async with lock:
            try:
                snap = await self.sync.client.set_session_model(sid, provider_id, model_id)
                # 从快照确认新模型
                new_model = (snap.get("session", {}).get("model") or {})
                if new_model:
                    label = new_model.get("modelId") or label
                msg = f"✅ 模型已切换:`{label}`"
            except AppServerError as e:
                msg = f"❌ 切换失败: {e}"
            except Exception as e:
                msg = f"❌ 切换失败: {e}"
        # 清掉列表缓存(切完失效)
        self._model_lists.pop(key, None)
        if edit:
            await update.edit_message_text(msg, parse_mode=ParseMode.MARKDOWN)
        else:
            await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)

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
        footer = _build_footer(result.total_tokens, self.config.workspace_path)
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


async def ctx_safe_edit(
    chat_id: int,
    message_id: int,
    text: str,
    reply_markup=None,
) -> None:
    app = _bot_ref.get("app")
    if app and app.bot:
        await app.bot.edit_message_text(
            chat_id=chat_id, message_id=message_id, text=text, reply_markup=reply_markup
        )


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


def _build_footer(total_tokens: int, workspace: str) -> str:
    """构造消息末尾的元信息:token 数 + 工作区目录。"""
    parts = []
    if total_tokens:
        parts.append(f"💎 {total_tokens} tokens")
    if workspace:
        parts.append(f"📁 {workspace}")
    return "\n\n_" + " · ".join(parts) + "_" if parts else ""


def _thread_id_of(update: Update) -> Optional[int]:
    """从 update 拿话题 message_thread_id。

    论坛群的话题内消息有 thread_id(General 顶层为 None → 归 1);
    普通群/私聊返回 None。用于 send_message / send_chat_action 落到正确话题。
    """
    chat = update.effective_chat
    msg = update.effective_message
    if chat and getattr(chat, "is_forum", False) and msg:
        tid = getattr(msg, "message_thread_id", None)
        return tid or 1  # General 话题
    return None


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
