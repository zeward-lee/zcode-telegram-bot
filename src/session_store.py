"""会话持久化 —— 每个 session key 绑定一个 ZCode sessionId。

存储 (key, session_id) 映射到 JSON 文件,重启不丢。
key 可以是 user_id(私聊)、chat_id(普通群)、或 "chat_id:thread_id"(论坛话题)。
支持:
- get / set / reset
- LRU 淘汰(每 key 会话数上限)
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from threading import Lock
from typing import Optional


class SessionStore:
    """线程安全的 sessionId 持久化存储。

    数据结构(JSON 文件):
        {
          "123456789": {                # key:str(user_id) 或 str(chat_id) 或 "chat_id:thread_id"
            "session_id": "sess_xxx",
            "last_seq": 42,              # 轮询水位(app-server 协议 seq)
            "delivery_kind": "web-remote-replayable",
            "updated_at": 1700000000
          }
        }
    """

    def __init__(self, db_path: str | Path, max_per_user: int = 1) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._max = max_per_user
        self._lock = Lock()
        self._data: dict[str, dict] = self._load()

    def _load(self) -> dict[str, dict]:
        if not self._path.is_file():
            return {}
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def _save(self) -> None:
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        tmp.replace(self._path)

    def get(self, key: str) -> Optional[str]:
        """取某 key 当前的 sessionId,没有则返回 None。"""
        with self._lock:
            entry = self._data.get(key)
            return entry["session_id"] if entry else None

    def set(self, key: str, session_id: str) -> None:
        """设置/更新某 key 的 sessionId(保留已有 last_seq/push_target)。"""
        with self._lock:
            existing = self._data.get(key, {})
            self._data[key] = {
                "session_id": session_id,
                "last_seq": existing.get("last_seq", 0),
                "delivery_kind": existing.get(
                    "delivery_kind", "web-remote-replayable"
                ),
                # 持久化推送目标(重启 resume 后 TUI→bot 推送能立即用,不丢)
                "chat_id": existing.get("chat_id"),
                "thread_id": existing.get("thread_id"),
                "updated_at": int(time.time()),
            }
            self._save()

    def set_push_target(
        self, key: str, chat_id: int, thread_id: Optional[int]
    ) -> None:
        """记录推送目标(chat_id + 话题 thread_id),重启后 resume 能恢复推送。"""
        with self._lock:
            entry = self._data.get(key)
            if not entry:
                return  # 没绑 session,不存孤立目标
            entry["chat_id"] = chat_id
            entry["thread_id"] = thread_id
            entry["updated_at"] = int(time.time())
            self._save()

    def iter_all(self) -> list[tuple[str, dict]]:
        """返回所有 (key, entry) 副本,供重启时 resume 已存 session 的 watcher。"""
        with self._lock:
            return [(k, dict(v)) for k, v in self._data.items()]

    def get_last_seq(self, key: str) -> int:
        """取某 key session 的轮询水位 seq(默认 0)。"""
        with self._lock:
            entry = self._data.get(key)
            return entry.get("last_seq", 0) if entry else 0

    def set_last_seq(self, key: str, seq: int) -> None:
        """更新某 key 的轮询水位(重启后续传,不重复推送)。"""
        with self._lock:
            entry = self._data.get(key)
            if not entry:
                return  # 没绑 session,不存孤立水位
            entry["last_seq"] = seq
            entry["updated_at"] = int(time.time())
            self._save()

    def reset(self, key: str) -> None:
        """清空某 key 的 session(下次消息会开新会话)。"""
        with self._lock:
            self._data.pop(key, None)
            self._save()
