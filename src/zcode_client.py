"""ZCode CLI 调用层。

封装 `zcode --prompt --json` 的子进程调用,提供:
- run():单次执行(新会话或续接),返回结构化结果
- 自动定位 zcode CLI 路径(App bundle 内或 PATH 中)
- 超时 / 错误处理 / JSON 解析
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ZCode CLI 在 macOS App bundle 内的默认路径
_APP_BUNDLE_CLI = "/Applications/ZCode.app/Contents/Resources/glm/zcode.cjs"


class ZCodeNotFoundError(RuntimeError):
    """找不到 zcode CLI。"""


def find_zcode_cli() -> str:
    """定位 zcode CLI 可执行入口。

    优先级:
    1. 环境变量 ZCODE_CLI_PATH(显式指定)
    2. PATH 中的 `zcode`(用户做过 symlink)
    3. macOS App bundle 内的 zcode.cjs
    """
    env_path = os.environ.get("ZCODE_CLI_PATH")
    if env_path and Path(env_path).is_file():
        return env_path

    in_path = shutil.which("zcode")
    if in_path:
        return in_path

    if Path(_APP_BUNDLE_CLI).is_file():
        return _APP_BUNDLE_CLI

    raise ZCodeNotFoundError(
        "找不到 zcode CLI。请设置环境变量 ZCODE_CLI_PATH 指向 zcode.cjs,"
        "或将 zcode symlink 到 PATH。"
    )


@dataclass
class ZCodeResult:
    """一次 --prompt --json 调用的结构化结果。"""

    response: str
    session_id: str
    usage: dict
    projection: dict
    raw: dict

    @property
    def total_tokens(self) -> int:
        return self.usage.get("totalTokens", 0)

    @property
    def is_error(self) -> bool:
        return bool(self.raw.get("isError", False))


@dataclass
class ZCodeClient:
    """封装对 zcode CLI 的调用。

    Args:
        working_dir: agent 工作目录(默认当前目录)
        timeout: 单次调用超时秒数
        cli_path: zcode CLI 路径(不传则自动定位)
    """

    working_dir: str = "."
    timeout: int = 600
    cli_path: Optional[str] = None

    def __post_init__(self) -> None:
        self.cli_path = self.cli_path or find_zcode_cli()

    def run(
        self,
        prompt: str,
        session_id: Optional[str] = None,
        cwd: Optional[str] = None,
    ) -> ZCodeResult:
        """执行一次 prompt,返回结构化结果。

        Args:
            prompt: 用户指令
            session_id: 若提供则 --resume 续接已有会话
            cwd: 覆盖默认工作目录
        """
        args = self._build_args(prompt, session_id)
        work_dir = cwd or self.working_dir

        try:
            # Windows 下 node.exe 是控制台程序,不设 CREATE_NO_WINDOW 会弹黑窗
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            proc = subprocess.run(
                args,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                creationflags=creationflags,
            )
        except subprocess.TimeoutExpired as e:
            raise ZCodeTimeoutError(
                f"zcode 调用超时({self.timeout}s)。prompt: {prompt[:80]!r}"
            ) from e

        return self._parse_output(proc.returncode, proc.stdout, proc.stderr)

    def _build_args(self, prompt: str, session_id: Optional[str]) -> list[str]:
        """构造 zcode CLI 命令行参数。"""
        # zcode.cjs 是 node 脚本;如果 cli_path 指向 .cjs 需要 node 前缀
        cli = self.cli_path
        args: list[str]
        if cli.endswith(".cjs"):
            args = ["node", cli]
        else:
            args = [cli]

        if session_id:
            args += ["--resume", session_id]
        args += ["--prompt", prompt, "--json", "--no-color"]
        return args

    def _parse_output(
        self, returncode: int, stdout: str, stderr: str
    ) -> ZCodeResult:
        """解析 CLI 输出为 ZCodeResult。

        优先解析 stdout 的 JSON;失败则从 stderr 报错。
        """
        # CLI 即使报错也可能 exit 0(实测 ModelProtocolError 也是 exit 0),
        # 所以优先看 stdout 是否能解析出结构化错误。
        if stdout.strip():
            try:
                data = json.loads(stdout)
                return ZCodeResult(
                    response=data.get("response", ""),
                    session_id=data.get("sessionId", ""),
                    usage=data.get("usage", {}),
                    projection=data.get("projection", {}),
                    raw=data,
                )
            except json.JSONDecodeError:
                pass  # 非 JSON,走 stderr 报错路径

        # stderr 里找 ModelProtocolError 等错误
        if returncode != 0 or stderr.strip():
            raise ZCodeExecutionError(
                f"zcode 执行失败(exit={returncode})。\n"
                f"stderr: {stderr.strip()[:500]}"
            )

        # stdout 既不是 JSON 也不是空,说明 CLI 行为变化了
        raise ZCodeExecutionError(
            f"zcode stdout 不是合法 JSON: {stdout.strip()[:500]}"
        )


class ZCodeError(RuntimeError):
    """ZCode 调用相关错误的基类。"""


class ZCodeTimeoutError(ZCodeError):
    """调用超时。"""


class ZCodeExecutionError(ZCodeError):
    """CLI 执行失败。"""
