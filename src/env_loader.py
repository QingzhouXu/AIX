"""
轻量级 .env 加载器，避免额外依赖。
"""
import os
from typing import Optional


def load_dotenv(dotenv_path: str = ".env", override: bool = False) -> bool:
    """
    从项目根目录加载 .env 文件到环境变量。

    Args:
        dotenv_path: .env 文件路径
        override: 是否覆盖现有环境变量

    Returns:
        是否成功找到并加载 .env 文件
    """
    if not os.path.exists(dotenv_path):
        # 兼容：仓库可能提供了 .env.zhipuai 作为示例配置
        if dotenv_path == ".env" and os.path.exists(".env.zhipuai"):
            dotenv_path = ".env.zhipuai"
        else:
            return False

    with open(dotenv_path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()

            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if not key:
                continue

            if override or key not in os.environ:
                os.environ[key] = value

    return True
