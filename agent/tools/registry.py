"""工具注册表：统一分发本地/MCP 工具（P1 类型级预留 + 空默认实现）。

图编排只依赖 `ToolRegistry` 接口，禁止内联硬编码 local/mcp 二分分支。
"""

from __future__ import annotations

from typing import Protocol

from agent.tools.models import ToolResult, ToolSpec, UnknownToolError


class Tool(Protocol):
    """工具契约：声明 + 异步执行。"""

    @property
    def spec(self) -> ToolSpec: ...

    async def invoke(self, arguments: dict[str, object]) -> ToolResult: ...


class ToolRegistry(Protocol):
    """统一工具注册表契约（dispatch 的唯一分发入口）。"""

    def register(self, tool: Tool) -> None: ...

    def resolve(self, name: str) -> Tool: ...

    def list_specs(self) -> list[ToolSpec]: ...


class InMemoryToolRegistry:
    """P1 空注册表默认实现：无任何工具，仅支撑依赖注入与图结构。

    WHY 默认注入：`build_agent_graph` 需要一个带类型的 registry 占位；
    真实工具（本地/MCP）在 P2 注册，本类届时无需改动。
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.spec.name] = tool

    def resolve(self, name: str) -> Tool:
        if name not in self._tools:
            raise UnknownToolError(name)
        return self._tools[name]

    def list_specs(self) -> list[ToolSpec]:
        return [tool.spec for tool in self._tools.values()]
