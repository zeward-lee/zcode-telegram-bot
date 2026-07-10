"""配置 —— 从环境变量/.env 加载。

必填项需要在 .env 里设好:
    TELEGRAM_BOT_TOKEN  - @BotFather 拿到的 token
    ALLOWED_USERS       - 允许使用的 Telegram user id(逗号分隔)
    APPROVED_DIRECTORY  - agent 工作目录(沙箱边界)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv() -> None:
    """简易 .env 加载(不引 python-dotenv 依赖)。"""
    env_path = Path(".env")
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = val


_load_dotenv()


def _parse_ids(raw: str) -> set[int]:
    """解析 '123, 456' / '-100123, 789' → {123, 456} / {-100123, 789}。

    支持负数(群组 chat_id 是负数,如 -100123456789)。
    """
    ids = set()
    for part in raw.replace(";", ",").split(","):
        part = part.strip()
        # lstrip("-") 兼容负数群 id
        if part.lstrip("-").isdigit():
            ids.add(int(part))
    return ids


# 向后兼容别名(旧代码可能引用)
_parse_user_ids = _parse_ids


@dataclass
class Config:
    telegram_token: str
    allowed_users: set[int]
    approved_directory: str
    zcode_cli_path: str
    zcode_timeout: int
    max_message_length: int
    # 群组白名单:这些群里的任何人 @bot 都可用(共享 session)
    allowed_chats: set[int]
    # app-server 同步相关
    workspace_path: str  # session/create 用的工作区(默认 = approved_directory)
    poll_interval: float  # TUI→bot 方向的轮询间隔(秒)
    use_app_server: bool  # True=走 app-server(新);False=回退 --prompt(旧)
    app_server_turn_timeout: float  # 单个 turn 最长等待秒数
    session_mode: str  # ZCode 权限模式:build(写操作需审批)/ yolo(全自动)/ plan / edit
    stream_edit_interval: float  # 流式输出时同条消息 edit 的最小间隔(秒,太小易触发 Telegram 429)

    @classmethod
    def from_env(cls) -> "Config":
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        users = _parse_ids(os.environ.get("ALLOWED_USERS", ""))
        approved = os.environ.get("APPROVED_DIRECTORY", os.getcwd())
        chats = _parse_ids(os.environ.get("ALLOWED_CHATS", ""))

        if not token:
            raise SystemExit(
                "❌ 缺少 TELEGRAM_BOT_TOKEN。请在 .env 中设置(从 @BotFather 获取)。"
            )
        if not users and not chats:
            raise SystemExit(
                "❌ 至少需要配置 ALLOWED_USERS(私聊 user id)或 ALLOWED_CHATS(群组 id)之一。\n"
                "   ALLOWED_USERS:私聊用,逗号分隔(私聊 @userinfobot 获取)\n"
                "   ALLOWED_CHATS :群组用,逗号分隔(群 id 为负数,如 -100123)"
            )

        return cls(
            telegram_token=token,
            allowed_users=users,
            approved_directory=approved,
            zcode_cli_path=os.environ.get("ZCODE_CLI_PATH", ""),
            zcode_timeout=int(os.environ.get("ZCODE_TIMEOUT", "600")),
            max_message_length=int(os.environ.get("MAX_MESSAGE_LENGTH", "4096")),
            allowed_chats=chats,
            workspace_path=os.environ.get("WORKSPACE_PATH", approved),
            poll_interval=float(os.environ.get("POLL_INTERVAL", "1.5")),
            use_app_server=os.environ.get("USE_APP_SERVER", "1") not in ("0", "false", "no"),
            app_server_turn_timeout=float(os.environ.get("APP_SERVER_TURN_TIMEOUT", "600")),
            session_mode=os.environ.get("SESSION_MODE", "build"),
            stream_edit_interval=float(os.environ.get("STREAM_EDIT_INTERVAL", "2.0")),
        )

    @property
    def db_path(self) -> str:
        return os.environ.get("SESSION_DB_PATH", "data/sessions.json")
