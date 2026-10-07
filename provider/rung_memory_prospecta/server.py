"""The MCP server: the `rung-memory/1` marker and the two hook tools, over streamable HTTP at /mcp."""

from __future__ import annotations

import weakref

import anyio
import mcp.types as types
from mcp.server.lowlevel import Server

from . import __version__
from .contract import (
    RECALL_TOOL,
    RETAIN_TOOL,
    ContractError,
    marker,
    read_recall,
    read_retain,
)
from .provider import Provider

FORGET_TOOL = "memory_forget"  # not in rung-memory/1: an ordinary agent tool

_OBJECT = {"type": "object"}

SEARCH_TOOL = "memory_search"
GATHER_TOOL = "memory_gather"

_DEPTH = {"type": "string", "enum": ["standard", "deep"]}

TOOLS = [
    types.Tool(
        name=RECALL_TOOL,
        description="rung hook (rung only): recall records for a scope, under a budget.",
        inputSchema={
            "type": "object",
            "properties": {
                "scope": {"type": "string"},
                "prompt": {"type": "string"},
                "context": {"type": "array", "items": {"type": "string"}},
                "budget": _OBJECT,
            },
            "required": ["scope"],
        },
    ),
    types.Tool(
        name=RETAIN_TOOL,
        description="rung hook (rung only): retain one observation in a scope.",
        inputSchema={
            "type": "object",
            "properties": {"scope": {"type": "string"}, "observation": _OBJECT},
            "required": ["scope", "observation"],
        },
    ),
    types.Tool(
        name=SEARCH_TOOL,
        description=(
            "Search long-term memory. `depth` is 'standard' (default) or 'deep'. Use 'deep' for "
            "discovery questions ('everything about X', 'all the ...', 'every time ...', lists and "
            "sets): it is more complete but costs about 3x. Use 'standard' otherwise."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "depth": {**_DEPTH, "default": "standard"},
            },
            "required": ["query"],
        },
    ),
    types.Tool(
        name=GATHER_TOOL,
        description=(
            "Gather a complete, cited answer to a SET question about one entity ('all the nicknames "
            "of X', 'every time X ...', 'list everything about X'): reads every note tied to the "
            "entity and returns a merged, cited list. Slower and costlier than memory_search; use "
            "it only for set/discovery questions. `depth` defaults to 'deep'."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "entity": {"type": "string"},
                "aliases": {"type": "array", "items": {"type": "string"}},
                "depth": {**_DEPTH, "default": "deep"},
            },
            "required": ["question", "entity"],
        },
    ),
    types.Tool(
        name=FORGET_TOOL,
        description=(
            "Forget one record from long-term memory, by the id a recalled record carries. "
            "Deletes it for good."
        ),
        inputSchema={
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
        },
    ),
]


class ProviderServer(Server):
    def __init__(self, provider: Provider) -> None:
        super().__init__("rung-memory-prospecta", version=__version__)
        self.provider = provider
        self.bound: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()  # session -> scope

    def create_initialization_options(self, notification_options=None, experimental_capabilities=None):
        caps = dict(experimental_capabilities or {})
        caps.update(marker(self.provider.settings.budget))
        return super().create_initialization_options(notification_options, caps)


def build(provider: Provider) -> ProviderServer:
    server = ProviderServer(provider)

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return TOOLS

    @server.call_tool(validate_input=False)
    async def call_tool(name: str, arguments: dict):
        session = server.request_context.session
        try:
            if name == RECALL_TOOL:
                call = read_recall(arguments, provider.settings.budget)
                server.bound[session] = call.scope
                return await anyio.to_thread.run_sync(provider.recall, call)
            if name == RETAIN_TOOL:
                call = read_retain(arguments)
                server.bound[session] = call.scope
                return await anyio.to_thread.run_sync(provider.retain, call)
            if name in (SEARCH_TOOL, GATHER_TOOL):
                scope = server.bound.get(session)
                if scope is None:
                    return _error(f"{name}: no scope: the session has not recalled yet")
                if name == SEARCH_TOOL:
                    if not isinstance(arguments.get("query"), str):
                        return _error("memory_search: `query` must be a string")
                    return await anyio.to_thread.run_sync(
                        provider.search, scope, arguments["query"], arguments.get("depth") or "standard"
                    )
                if not all(isinstance(arguments.get(k), str) for k in ("question", "entity")):
                    return _error("memory_gather: `question` and `entity` must be strings")
                return await anyio.to_thread.run_sync(
                    provider.gather,
                    scope,
                    arguments["question"],
                    arguments["entity"],
                    arguments.get("aliases"),
                    arguments.get("depth") or "deep",
                )
            if name == FORGET_TOOL:
                # Agent tools get no scope from rung: use the scope of this session's latest hook.
                scope = server.bound.get(session)
                if scope is None or not isinstance(arguments.get("id"), str):
                    return _error("memory_forget: needs a recalled record id and a session that has recalled")
                out = await anyio.to_thread.run_sync(provider.forget, scope, arguments["id"])
                return out
            return _error(f"unknown tool {name}")
        except ContractError as e:
            return _error(f"{name}: {e}")
        except Exception as e:  # a failed hook is an error result, never a crashed server
            return _error(f"{name}: {type(e).__name__}: {e}")

    return server


def _error(text: str) -> types.CallToolResult:
    return types.CallToolResult(content=[types.TextContent(type="text", text=text)], isError=True)


def http_app(provider: Provider):
    import contextlib

    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from mcp.server.transport_security import TransportSecuritySettings
    from starlette.applications import Starlette
    from starlette.routing import Route

    security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=["127.0.0.1:*", "localhost:*", "[::1]:*"],
        allowed_origins=["http://127.0.0.1:*", "http://localhost:*"],
    )
    manager = StreamableHTTPSessionManager(
        app=build(provider), json_response=True, security_settings=security
    )

    class Handle:
        async def __call__(self, scope, receive, send):
            await manager.handle_request(scope, receive, send)

    @contextlib.asynccontextmanager
    async def lifespan(_app):
        async with manager.run():
            yield

    return Starlette(
        routes=[Route("/mcp", endpoint=Handle(), methods=["GET", "POST", "DELETE"])], lifespan=lifespan
    )
