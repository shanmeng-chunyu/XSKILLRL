"""
工具模块初始化
自动导入所有工具并注册到工具注册表
"""

from .base import BaseTool
from .tool_registry import TOOL_REGISTRY, register_tool, get_tool, list_tools, get_tool_info

try:
    from .code_interpreter import CodeInterpreter
except ImportError as exc:
    CodeInterpreter = None
    print(f"Warning: CodeInterpreter not available: {exc}")

try:
    from .web_search import WebSearch
except ImportError as exc:
    WebSearch = None
    print(f"Warning: WebSearch not available: {exc}")

try:
    from .visit import Visit
except ImportError as exc:
    Visit = None
    print(f"Warning: Visit not available: {exc}")

try:
    from .image_search import ImageSearch
except ImportError as exc:
    ImageSearch = None
    print(f"Warning: ImageSearch not available: {exc}")

try:
    from .zoom import ZoomTool
except ImportError as exc:
    ZoomTool = None
    print(f"Warning: ZoomTool not available: {exc}")


__all__ = [
    'BaseTool',
    'TOOL_REGISTRY',
    'register_tool',
    'get_tool',
    'list_tools',
    'get_tool_info',
    'CodeInterpreter',
    'WebSearch',
    'Visit',
    'ImageSearch',
    'ZoomTool',
]

