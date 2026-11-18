# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""API planner composed from Pydantic AI and Harness capabilities.

The SDK owns model requests, tool scheduling, retries, queued-message delivery,
and cancellation. Harness owns compaction; clai owns terminal interaction.
RPent's runner saves the returned transcript through its existing output flow.
"""

import asyncio
import json
import threading
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager, nullcontext
from contextvars import Context
from functools import partial
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from jsonschema import Draft202012Validator
from pydantic_ai import (
    Agent,
    BinaryContent,
    CallToolsNode,
    CancellationToken,
    EnqueuedMessagesEvent,
    FunctionToolset,
    ModelRequestNode,
    ModelRetry,
    ModelSettings,
    PartEndEvent,
    RunCancelled,
    RunContext,
    RunUsage,
    StructuredDict,
    TextPart,
    ThinkingPart,
    Tool,
    ToolCallPart,
    ToolOutput,
    ToolReturn,
    UsageLimits,
)
from pydantic_ai.agent import WrapperAgent
from pydantic_ai.capabilities import AbstractCapability, Thinking, on_event
from pydantic_ai.capabilities.abstract import AgentNode, NodeResult, WrapRunHandler
from pydantic_ai.exceptions import (
    UnexpectedModelBehavior,
    UsageLimitExceeded,
    UserError,
)
from pydantic_ai.messages import ModelMessage, UserContent
from pydantic_ai.models import Model, ModelRequestContext
from pydantic_ai.run import AgentRun, AgentRunResult
from pydantic_ai_harness.compaction import SlidingWindowCompaction

from rpent.dashboard.events import DashboardEventSink, TranscriptEvent, UsageEvent
from rpent.dashboard.interaction import DashboardInteractionPort, DashboardMessage
from rpent.dashboard.planner_control import DashboardPlannerControl
from rpent.planner.base import (
    REASONING_EFFORTS,
    Planner,
    PlannerResult,
    _wait_for_cleanup,
    cancel_and_wait,
    execute_tool,
)
from rpent.session import Payload
from rpent.session.env_capture import current_tool_call
from rpent.session.planner_capture import CapturingModel, sdk_payload
from rpent.tools.toolkit import Toolkit, ToolResult
from rpent.utils.logging import get_logger
from rpent.utils.templates import substitute

if TYPE_CHECKING:
    from rpent.utils.mcp_servers import McpServerSpec

logger = get_logger("api")

#: Model-request budget for one scene narration session. This is independent of
#: the main conversation's ``max_turns``.
_NARRATOR_MAX_REQUESTS = 200

_SCENE_NARRATOR_PROMPT = (
    "You narrate a robot's camera view continuously. Repeat this round until "
    "told to stop:\n"
    "1. Call view_env_state(step=-1) or read_image for the latest observation "
    "and look at agentview_high.png.\n"
    "2. Write one Chinese sentence of at most 20 characters about the arm and "
    "the main objects, without Markdown. If the view has not changed since "
    "your last round, describe a different detail.\n"
    "3. Call a text-to-speech tool whose name ends with '_synthesize' once with "
    "exactly that sentence, and wait for it to return before starting the next "
    "round.\n"
    "Start the next round immediately, whether or not the view changed. Stop "
    "and reply with a one-line summary only when view_env_state reports "
    "terminated or truncated as true, or a tool reports that the session is "
    "closed. Do not call any other tool."
)

_START_NARRATOR_DESCRIPTION = (
    "Start the background scene narrator. It repeatedly reads the latest robot "
    "observation and speaks one short Chinese sentence through the external "
    "text-to-speech MCP tool until the episode ends. Call it once when the user "
    "asks for spoken narration; do not wait for it."
)


class _RequestBudgetExhausted(Exception):
    """The conversation has used its configured model-request budget."""


def _is_response_truncation(exc: UnexpectedModelBehavior) -> bool:
    """Match the SDK's token-limit-without-usable-output failure."""
    message = str(exc).lower()
    return "token limit" in message and "before any response was generated" in message


def _mcp_toolset() -> Any:
    """Return the optional pydantic-ai MCP toolset class."""
    try:
        from pydantic_ai.mcp import MCPToolset
    except ImportError as exc:
        raise RuntimeError(
            "attaching external MCP servers to the 'api' planner requires "
            "the pydantic-ai MCP client; install it with "
            '`pip install "pydantic-ai-slim[mcp]"`.'
        ) from exc
    return MCPToolset


def _resolve_narrator_sources(
    toolkit: Toolkit, mcp_servers: tuple["McpServerSpec", ...]
) -> tuple[tuple[str, ...], tuple["McpServerSpec", ...]]:
    """Find the observation tools and synthesis servers a narrator may use.

    Raises:
        ValueError: If no attached planner-layer server lists ``synthesize``.
    """
    from rpent.utils.mcp_servers import ALL_TOOLS

    names = {declaration.name for declaration in toolkit.list_tools()}
    observe = tuple(
        name
        for name in ("view_env_state", "read_image")
        if name in names or name == "read_image"
    )
    speak = tuple(
        server
        for server in mcp_servers
        if server.planner_tools == ALL_TOOLS
        or (server.planner_tools and "synthesize" in server.planner_tools)
    )
    if not speak:
        raise ValueError(
            "the scene_narrator subagent needs an external MCP server listing "
            "synthesize under planner_tools"
        )
    return observe, speak


class ApiAgentLoop(Planner):
    """Implement ``Planner`` with a native Agent and reusable Harness capabilities."""

    def __init__(
        self,
        *,
        model: Model,
        dashboard_events: DashboardEventSink,
        max_tokens: int = 8192,
        reasoning_effort: str = "none",
        no_images: bool = False,
        timeout_s: float = 1200,
        interactive: bool = False,
        mcp_servers: tuple["McpServerSpec", ...] = (),
        narrator_model: Model | None = None,
    ) -> None:
        """Initialize the API planner.

        Args:
            narrator_model: Optional model for a background ``scene_narrator``
                agent. ``None`` disables narration and leaves the main agent
                unchanged.
        """
        if reasoning_effort not in REASONING_EFFORTS:
            raise ValueError(f"unsupported reasoning effort: {reasoning_effort}")
        if max_tokens < 1 or timeout_s <= 0:
            raise ValueError("max_tokens and timeout_s must be positive")
        self.model = model
        self.narrator_model = narrator_model
        self.dashboard_events = dashboard_events
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.no_images = no_images
        self.timeout_s = timeout_s
        self.interactive = interactive
        self.mcp_servers = mcp_servers

    def solve(
        self,
        *,
        system_prompt: str,
        user_message: str,
        toolkit: Toolkit,
        max_turns: int,
        input_queue=None,
        dashboard_interaction: DashboardInteractionPort | None = None,
    ) -> PlannerResult:
        """Run a conversation and return its transcript for RPent to save."""
        if max_turns < 1:
            raise ValueError("max_turns must be positive")
        if input_queue is not None:
            raise ValueError("api does not use RPent's terminal input queue")
        if self.interactive and dashboard_interaction is not None:
            raise ValueError("clai and Dashboard cannot run together")
        return asyncio.run(
            self._solve(
                system_prompt,
                user_message,
                toolkit,
                max_turns,
                dashboard_interaction,
            )
        )

    async def _solve(
        self,
        system_prompt: str,
        user_message: str,
        toolkit: Toolkit,
        max_turns: int,
        interaction: DashboardInteractionPort | None,
    ) -> PlannerResult:
        conversation_id = uuid4().hex
        try:
            state = toolkit.state
        except RuntimeError:
            state = None
        capture = (
            CapturingModel(self.model, state, conversation_id)
            if state is not None and state.get_task() is not None
            else None
        )
        narrator_sources = (
            _resolve_narrator_sources(toolkit, self.mcp_servers)
            if self.narrator_model is not None
            else None
        )
        try:
            # The adapter registers one pydantic-ai tool per Toolkit tool, so a
            # robot tool that reuses a planner-owned name (read_image) fails
            # here. Guard construction for the same reason as the Agent below:
            # it must end as PlannerResult.error, not escape solve().
            adapter = _HarnessToolkit(
                toolkit,
                dashboard_events=self.dashboard_events,
                no_images=self.no_images,
                capture=capture,
            )
            narrator = (
                self._build_scene_narrator(toolkit, adapter, narrator_sources)
                if narrator_sources is not None
                else None
            )
            if narrator is not None:
                adapter.add_tool(
                    Tool(
                        narrator.start,
                        name="start_scene_narrator",
                        description=_START_NARRATOR_DESCRIPTION,
                        takes_ctx=True,
                        sequential=True,
                    )
                )
        except Exception as exc:
            logger.exception("Harness toolkit construction failed")
            return PlannerResult(
                error=f"{type(exc).__name__}: {exc}", stop_reason="planner_error"
            )
        session = _Session(
            adapter, conversation_id, max_turns, interactive=self.interactive
        )
        session.narrator = narrator
        if interaction is not None:
            session.control = DashboardPlannerControl(
                interaction=interaction,
                cancel_active_and_wait=adapter.cancel_active_and_wait,
                resume_calls=toolkit.resume_calls,
                emit_user=session.emit_user,
                emit_initial_user=lambda: session.emit_user(user_message),
                defer_message_ack=True,
                submit_while_busy=True,
            )
        try:
            # Building the toolsets resolves env-backed MCP headers, so keep it
            # inside the try: a bad site config must end as PlannerResult.error,
            # not as an exception that skips RPent's normal failure handling.
            agent = Agent(
                capture if capture is not None else self.model,
                name="rpent_api",
                system_prompt=system_prompt,
                output_type=adapter.output_types,
                toolsets=[adapter, *self._external_toolsets(adapter)],
                # Accepted finish skips sibling tool calls in the same response.
                end_strategy="early",
                # Finish refusals may span attempts; the request budget still applies.
                retries={"output": max_turns},
                model_settings=_build_model_settings(
                    self.model, self.max_tokens, self.reasoning_effort
                ),
                capabilities=[
                    SlidingWindowCompaction(
                        max_messages=80, max_fraction=0.8, keep_messages=40
                    ),
                    Thinking(
                        False
                        if self.reasoning_effort == "none"
                        else self.reasoning_effort
                    ),
                    session,
                ],
            )
            await asyncio.wait_for(
                session.run(_ConversationAgent(agent), user_message),
                timeout=None if self.interactive else self.timeout_s,
            )
        except asyncio.TimeoutError:
            session.stop_reason = "timeout"
            session.error = f"planner timed out after {self.timeout_s:g}s"
        except _RequestBudgetExhausted as exc:
            session.stop_reason = "request_limit"
            session.error = None
            logger.info("%s", exc)
        except Exception as exc:
            session.stop_reason = "planner_error"
            if isinstance(exc, UnexpectedModelBehavior) and _is_response_truncation(
                exc
            ):
                session.error = f"planner response truncated (length): {exc}"
                logger.warning("%s", session.error)
            else:
                session.error = f"{type(exc).__name__}: {exc}"
                logger.exception("Harness planner failed")
        finally:
            session.emit_usage()
        return PlannerResult(
            finish_result=adapter.finish_result,
            messages=adapter.messages,
            stats={
                "total_input_tokens": session.usage.input_tokens,
                "total_output_tokens": session.usage.output_tokens,
                "total_cached_input_tokens": session.usage.cache_read_tokens,
                "turns_used": session.usage.requests,
                "tool_calls": adapter.tool_calls,
            },
            error=session.error,
            stop_reason=session.stop_reason,
        )

    def _external_toolsets(self, recorder: "_HarnessToolkit") -> list[Any]:
        """Build one MCP toolset per planner-layer external server.

        Each server's tools are prefixed with its name (``<server>_<tool>``) so
        two servers cannot collide. Remote tool names are only known after the
        server connects, so a name that still collides with an RPent tool is
        reported by pydantic-ai during the first run step. A spec that names
        its tools is filtered before dispatch, so coordinated tools adopted by
        the robot's Toolkit cannot leak back into the planner layer. The
        pydantic-ai MCP client is an optional import: it is only needed when a
        server is actually attached.
        """
        servers = [
            server for server in self.mcp_servers if server.planner_tools is not None
        ]
        if not servers:
            return []
        MCPToolset = _mcp_toolset()

        from rpent.utils.mcp_servers import ALL_TOOLS, request_headers

        toolsets: list[Any] = []
        for server in servers:
            toolset = MCPToolset(
                server.url,
                id=server.name,
                headers=request_headers(server) or None,
                process_tool_call=recorder.external_tool_hook(server.name),
            )
            if server.planner_tools != ALL_TOOLS:
                allowed = frozenset(server.planner_tools or ())
                toolset = toolset.filtered(
                    lambda _ctx, tool_def, allowed=allowed: tool_def.name in allowed
                )
            toolsets.append(toolset.prefixed(server.name))
        return toolsets

    def _build_scene_narrator(
        self,
        toolkit: Toolkit,
        recorder: "_HarnessToolkit",
        sources: tuple[tuple[str, ...], tuple["McpServerSpec", ...]],
    ) -> "_SceneNarrator":
        """Build the background narrator agent and its filtered toolsets."""
        observe, speak = sources
        assert self.narrator_model is not None
        narrator_toolkit = _HarnessToolkit(
            toolkit,
            dashboard_events=self.dashboard_events,
            no_images=False,
            capture=None,
            sequential=True,
        )
        observe_toolset = narrator_toolkit.filtered(
            lambda _ctx, tool_def: tool_def.name in observe
        )
        agent = Agent(
            self.narrator_model,
            name="scene_narrator",
            system_prompt=_SCENE_NARRATOR_PROMPT,
            output_type=str,
            toolsets=[
                observe_toolset,
                *self._narrator_toolsets(narrator_toolkit, speak),
            ],
            model_settings=_build_model_settings(
                self.narrator_model,
                min(self.max_tokens, 1024),
                "none",
                parallel_tool_calls=False,
            ),
            capabilities=[
                SlidingWindowCompaction(
                    max_messages=40, max_fraction=0.8, keep_messages=20
                ),
            ],
        )
        return _SceneNarrator(agent, recorder, self.dashboard_events)

    def _narrator_toolsets(
        self,
        recorder: "_HarnessToolkit",
        servers: tuple["McpServerSpec", ...],
    ) -> list[Any]:
        """Expose only ``synthesize`` from each server to the narrator."""
        if not servers:
            return []
        MCPToolset = _mcp_toolset()

        from rpent.utils.mcp_servers import request_headers

        toolsets: list[Any] = []
        for server in servers:
            toolset = MCPToolset(
                server.url,
                id=f"{server.name}_narrator",
                headers=request_headers(server) or None,
                process_tool_call=recorder.external_tool_hook(server.name),
            )
            toolset = toolset.filtered(
                lambda _ctx, tool_def: tool_def.name == "synthesize"
            )
            toolsets.append(toolset.prefixed(server.name))
        return toolsets


def _build_model_settings(
    model: Model,
    max_tokens: int,
    reasoning_effort: str | None = None,
    *,
    parallel_tool_calls: bool = True,
) -> ModelSettings:
    """Enable parallel tool requests and preserve Anthropic prompt caching."""
    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
    from pydantic_ai.profiles.openai import OPENAI_REASONING_EFFORT_MAP

    if isinstance(model, AnthropicModel):
        return AnthropicModelSettings(
            max_tokens=max_tokens,
            parallel_tool_calls=parallel_tool_calls,
            anthropic_cache_instructions=True,
            anthropic_cache_tool_definitions=True,
            anthropic_cache_messages=True,
        )
    if isinstance(model, OpenAIChatModel) and reasoning_effort is not None:
        return OpenAIChatModelSettings(
            max_tokens=max_tokens,
            parallel_tool_calls=parallel_tool_calls,
            openai_reasoning_effort=OPENAI_REASONING_EFFORT_MAP.get(
                reasoning_effort, reasoning_effort
            ),
        )
    return ModelSettings(max_tokens=max_tokens, parallel_tool_calls=parallel_tool_calls)


def _external_result_text(result: Any) -> str:
    """Render an external MCP tool result as transcript text.

    Image content is recorded by media type and size: transcript events and
    message logs must never carry the raw bytes, which would defeat
    ``--no-images`` and make the Dashboard replay large payloads.
    """
    if isinstance(result, str):
        return result
    if isinstance(result, BinaryContent):
        return f"[image: {result.media_type}, {len(result.data)} bytes]"
    if isinstance(result, dict):
        return json.dumps(result)
    if isinstance(result, Sequence):
        return "\n".join(_external_result_text(item) for item in result)
    return str(result)


def _omit_external_images(result: Any) -> Any:
    """Replace external MCP image content the way local tools do."""
    if isinstance(result, BinaryContent):
        return "[Image omitted: --no-images is enabled.]"
    if isinstance(result, list):
        return [_omit_external_images(item) for item in result]
    return result


class _ConversationAgent(WrapperAgent):
    """Preserve completed operations across failed and cancelled runs.

    This checkpoint takes precedence over the CLI's last successful history.
    """

    def __init__(self, agent: Agent) -> None:
        super().__init__(agent)
        self.history: list[ModelMessage] | None = None

    @asynccontextmanager
    async def iter(
        self,
        user_prompt: str | Sequence[UserContent] | None = None,
        *,
        message_history: Sequence[ModelMessage] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[AgentRun]:
        run: AgentRun | None = None
        try:
            async with self.wrapped.iter(
                user_prompt,
                message_history=(
                    self.history if self.history is not None else message_history
                ),
                **kwargs,
            ) as run:
                yield run
        finally:
            # Read after the SDK exits so cancellation cleanup is included.
            if run is not None:
                self.history = run.all_messages()


class _HarnessToolkit(FunctionToolset):
    """Expose Toolkit tools and a read-only image artifact reader."""

    def __init__(
        self,
        toolkit: Toolkit,
        *,
        dashboard_events: DashboardEventSink,
        no_images: bool,
        capture: CapturingModel | None = None,
        sequential: bool = False,
    ) -> None:
        super().__init__(id="rpent", sequential=sequential)
        self.toolkit = toolkit
        self.dashboard_events = dashboard_events
        self.no_images = no_images
        self.capture = capture
        self.sequential = sequential
        self.finish_result: dict[str, Any] | None = None
        self.tool_calls = 0
        self.messages: list[dict[str, Any]] = []
        self.stopping = threading.Event()
        self.validators: dict[str, Draft202012Validator] = {}
        self.output_types: list[Any] = [str]
        self.add_tool(Tool(self.read_image, takes_ctx=True, sequential=True))
        for declaration in toolkit.list_tools():
            spec = {
                "name": declaration.name,
                "description": declaration.description,
                "input_schema": substitute(declaration.input_schema),
            }
            name = declaration.name
            self.validators[name] = Draft202012Validator(spec["input_schema"])
            if name == "finish":
                # Eager annotations bind this toolkit's schema to the output
                # function. Keep this module free of postponed annotations.
                FinishArguments = StructuredDict(
                    spec["input_schema"], name="FinishArguments"
                )

                async def finish(
                    ctx: RunContext, arguments: FinishArguments
                ) -> dict[str, Any]:
                    with self.capture.tool(ctx) if self.capture else nullcontext():
                        result = await self.execute("finish", arguments, ctx)
                        if not result.data.get("_finish"):
                            raise ModelRetry(self._text(result))
                        self.finish_result = result.to_dict()
                        if self.capture is not None:
                            self.capture.feedback(ctx, "json", self.finish_result)
                        return self.finish_result

                self.output_types.append(
                    ToolOutput(
                        finish, name="finish", description=spec.get("description")
                    )
                )
            else:
                self.add_tool(
                    Tool.from_schema(
                        partial(self.call, name),
                        name=name,
                        description=spec.get("description"),
                        json_schema=spec["input_schema"],
                        takes_ctx=True,
                        sequential=sequential,
                    )
                )

    async def read_image(
        self, ctx: RunContext, name: str, step: int = -1
    ) -> ToolReturn:
        """Read a saved image by artifact filename and step (-1 selects the latest)."""
        with self.capture.tool(ctx) if self.capture else nullcontext():
            self._record_call("read_image", {"name": name, "step": step}, ctx)
            worker = asyncio.create_task(
                asyncio.to_thread(self._read_image, name, step)
            )
            try:
                result = await asyncio.shield(worker)
            except asyncio.CancelledError:
                await _wait_for_cleanup(worker)
                raise
            self._record_result("read_image", ctx, json.dumps(result.return_value))
            if self.capture is not None:
                self.capture.feedback(ctx, "pydantic_ai.tool_return", result)
            return result

    def _read_image(self, name: str, step: int) -> ToolReturn:
        """Resolve an image in the step store, returning artifact errors to the model."""
        capture = current_tool_call.get()
        if capture is not None:
            capture.start()
        try:
            state = self.toolkit.state
            record = state.get(step)
            path = state.artifact_path(name, step=record.step_idx)
            if name not in record.artifacts or not path.is_file():
                raise FileNotFoundError(
                    f"image artifact {name!r} is not available at step {step}"
                )
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                raise ValueError(f"artifact {name!r} is not an image")
            metadata = {"artifact": name, "step": record.step_idx}
            if self.no_images:
                result = ToolReturn(
                    return_value={
                        **metadata,
                        "notice": "Image omitted: --no-images is enabled.",
                    }
                )
            else:
                image = BinaryContent(
                    data=state.load_bytes(name, step=record.step_idx),
                    media_type=(
                        "image/jpeg"
                        if path.suffix.lower() in {".jpg", ".jpeg"}
                        else "image/png"
                    ),
                )
                result = ToolReturn(return_value=metadata, content=[image])
        except Exception as exc:
            result = ToolReturn(return_value={"error": str(exc)})
        if capture is not None:
            capture.result(sdk_payload("pydantic_ai.tool_return", result))
            capture.finish("failed" if "error" in result.return_value else "completed")
        return result

    async def execute(
        self, name: str, arguments: dict[str, Any], ctx: RunContext
    ) -> ToolResult:
        """Validate and run one physical operation, retaining cancellation ownership."""
        error = next(self.validators[name].iter_errors(arguments), None)
        if error is not None:
            raise ModelRetry(f"Invalid arguments for {name}: {error.message}")
        self._record_call(name, arguments, ctx)
        try:
            result = await execute_tool(self.toolkit, name, arguments)
        except asyncio.CancelledError:
            self.stopping.set()
            raise
        self._record_result(name, ctx, self._text(result))
        if self.stopping.is_set():
            # End inside CallToolsNode so the SDK records an interrupted batch
            # before it can consume queued messages in the next request node.
            ctx.cancel()
            raise asyncio.CancelledError
        return result

    def _record_call(
        self, name: str, arguments: dict[str, Any], ctx: RunContext
    ) -> None:
        if self.stopping.is_set():
            # Toolkit draining can finish before Dashboard sends its token.
            # Mark this as application cancellation so the SDK raises RunCancelled.
            ctx.cancel()
            raise asyncio.CancelledError
        self.dashboard_events.emit(
            TranscriptEvent({"type": "tool_call", "tool": name, "args": arguments})
        )
        self.tool_calls += 1

    def _record_result(self, name: str, ctx: RunContext, text: str) -> None:
        self.messages.append(
            {
                "role": "tool",
                "name": name,
                "tool_call_id": ctx.tool_call_id,
                "content": text,
            }
        )
        self.dashboard_events.emit(
            TranscriptEvent({"type": "tool_result", "tool": name, "result": text})
        )

    def external_tool_hook(self, server: str) -> Callable[..., Any]:
        """Wrap an external MCP tool call so it is recorded like a local one.

        The toolset's ``process_tool_call`` receives the bare tool name; the
        server prefix is restored here to match the name the model invoked.
        """

        async def hook(
            ctx: RunContext,
            call_tool: Callable[..., Any],
            name: str,
            args: dict[str, Any],
        ) -> Any:
            tool = f"{server}_{name}"
            self._record_call(tool, args, ctx)
            try:
                result = await call_tool(name, args)
            except Exception as exc:
                # Keep tool_call/tool_result paired even when the remote fails;
                # pydantic-ai converts the exception into a retry for the model.
                self._record_result(
                    tool, ctx, json.dumps({"error": f"{type(exc).__name__}: {exc}"})
                )
                raise
            self._record_result(tool, ctx, _external_result_text(result))
            return _omit_external_images(result) if self.no_images else result

        return hook

    async def call(self, name: str, ctx: RunContext, /, **arguments: Any) -> ToolReturn:
        """Return native multimodal content for an ordinary tool."""
        with self.capture.tool(ctx) if self.capture else nullcontext():
            result = await self.execute(name, arguments, ctx)
            content: list[str | BinaryContent] = [result.to_text()]
            for image in result.images:
                content.append(
                    "[Image omitted: --no-images is enabled.]"
                    if self.no_images
                    else BinaryContent(data=image, media_type="image/png")
                )
            result = ToolReturn(return_value=content)
            if self.capture is not None:
                self.capture.feedback(ctx, "pydantic_ai.tool_return", result)
            return result

    def cancel_active_and_wait(self) -> None:
        """Stop admitting tools before requesting physical cancellation."""
        self.stopping.set()
        self.toolkit.cancel_active_and_wait()

    @staticmethod
    def _text(result: ToolResult) -> str:
        return result.to_text()


class _SceneNarrator:
    """Run one background narrator agent for the lifetime of a session."""

    def __init__(
        self,
        agent: Agent,
        recorder: "_HarnessToolkit",
        dashboard_events: DashboardEventSink,
    ) -> None:
        self._agent = agent
        self._recorder = recorder
        self._events = dashboard_events
        self._task: asyncio.Task[None] | None = None
        self._prompt = "Start narrating the scene."

    async def start(self, ctx: RunContext) -> str:
        """Start narration once; return immediately so the main agent continues."""
        capture = self._recorder.capture
        with capture.tool(ctx) if capture is not None else nullcontext():
            self._recorder._record_call("start_scene_narrator", {}, ctx)
            if capture is not None:
                capture.start()
            if self._task is not None and not self._task.done():
                text = "The scene narrator is already running."
            else:
                # An empty context keeps the main agent's current_tool_call
                # ContextVar out of the narrator's tool executions.
                self._task = asyncio.create_task(
                    self._run(),
                    name="scene-narrator",
                    context=Context(),
                )
                text = "Scene narrator started; it runs until the episode ends."
            self._recorder._record_result("start_scene_narrator", ctx, text)
            if capture is not None:
                capture.result(
                    Payload(
                        format="rpent.tool_result",
                        data={"data": {"text": text}, "images": []},
                        completeness="complete",
                    )
                )
                capture.feedback(
                    ctx,
                    "pydantic_ai.tool_return",
                    ToolReturn(return_value=text),
                )
                capture.finish("completed")
            return text

    async def stop(self) -> None:
        """Cancel and wait for the background narrator, if it is running."""
        task = self._task
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("scene narrator cleanup failed")

    async def _run(self) -> None:
        self._emit("[scene_narrator] started")
        try:
            await self._agent.run(
                self._prompt,
                usage_limits=UsageLimits(request_limit=_NARRATOR_MAX_REQUESTS),
                event_stream_handler=self._stream,
            )
        except asyncio.CancelledError:
            raise
        except UsageLimitExceeded:
            logger.info(
                "scene narrator reached its %d-request budget",
                _NARRATOR_MAX_REQUESTS,
            )
            self._emit("[scene_narrator] request budget reached")
        except Exception:
            logger.exception("scene narrator failed")
            self._emit("[scene_narrator] failed; see logs")
        finally:
            self._emit("[scene_narrator] stopped")

    async def _stream(self, _ctx: RunContext, stream: AsyncIterator[Any]) -> None:
        async for event in stream:
            if isinstance(event, PartEndEvent) and isinstance(event.part, TextPart):
                text = event.part.content.strip()
                if text:
                    self._emit(f"[scene_narrator] {text}")

    def _emit(self, text: str) -> None:
        self._events.emit(TranscriptEvent({"type": "text", "text": text}))


class _Session(AbstractCapability):
    """Track native runs and bridge Dashboard input and events.

    This capability observes public lifecycle boundaries; it never replaces a
    graph node or mutates message history. The conversation driver starts a new
    native run only for a follow-up after completion or interruption.
    """

    def __init__(
        self,
        toolkit: _HarnessToolkit,
        conversation_id: str,
        max_turns: int,
        *,
        interactive: bool = False,
    ) -> None:
        self.toolkit = toolkit
        self.conversation_id = conversation_id
        self.limits = UsageLimits(request_limit=max_turns)
        self.usage = RunUsage()
        self.interactive = interactive
        self.error: str | None = None
        self.stop_reason = "model_output"
        self.control: DashboardPlannerControl | None = None
        self.inbox: asyncio.Queue[DashboardMessage] = asyncio.Queue()
        self.enqueued: dict[str, DashboardMessage] = {}
        self.run_done = asyncio.Event()
        self.run_done.set()
        self.cancellation: CancellationToken | None = None
        self.completions = 0
        self.narrator: _SceneNarrator | None = None

    async def wrap_run(
        self, ctx: RunContext, *, handler: WrapRunHandler
    ) -> AgentRunResult:
        if self.toolkit.finish_result is not None:
            raise UserError("The task has finished. Use /exit to close the session.")
        self.toolkit.toolkit.resume_calls()
        self.toolkit.stopping.clear()
        if self.interactive and ctx.prompt:
            self.emit_user(ctx.prompt)
        try:
            result = await handler()
            self.error = None
            self.stop_reason = (
                "finish" if self.toolkit.finish_result is not None else "model_output"
            )
            return result
        except _RequestBudgetExhausted:
            # The native CLI catches run exceptions before _solve can see them.
            self.stop_reason = "request_limit"
            self.error = None
            raise
        except RunCancelled:
            self.stop_reason = "cancelled"
            self.error = None
            raise
        except Exception as exc:
            self.stop_reason = "planner_error"
            if isinstance(exc, UnexpectedModelBehavior) and _is_response_truncation(
                exc
            ):
                self.error = f"planner response truncated (length): {exc}"
            else:
                self.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            # clai also keeps its own per-run usage for /usage. Observe it without
            # replacing the SDK's counters, including failed/interrupted runs.
            self.usage.incr(ctx.usage)
            self.emit_usage()
            if self.toolkit.capture is not None:
                self.toolkit.capture.settle(
                    ctx.run_id, "SDK run ended before dispatch."
                )

    async def before_model_request(
        self, ctx: RunContext, request_context: ModelRequestContext
    ) -> ModelRequestContext:
        try:
            self.limits.check_before_request(self.usage + ctx.usage)
        except UsageLimitExceeded:
            # This limit only caps requests. Preserve other SDK/provider failures.
            message = f"Request budget of {self.limits.request_limit} reached."
            if self.interactive:
                message += " Use /exit to close the session."
            raise _RequestBudgetExhausted(message) from None
        if self.toolkit.capture is not None:
            self.toolkit.capture.before_request(ctx)
        return request_context

    async def before_node_run(self, ctx: RunContext, *, node: AgentNode) -> AgentNode:
        if self.toolkit.stopping.is_set():
            # Dashboard drains physical calls before awaiting the SDK interrupt.
            # Do not deliver queued input during that drain.
            ctx.cancel()
            return node
        if not isinstance(node, ModelRequestNode):
            return node
        while not self.inbox.empty():
            submission = self.inbox.get_nowait()
            enqueue_id = ctx.enqueue(submission.text)
            if enqueue_id is not None:
                self.enqueued[enqueue_id] = submission
                self.completions += 1
        return node

    async def after_node_run(
        self, ctx: RunContext, *, node: AgentNode, result: NodeResult
    ) -> NodeResult:
        if isinstance(node, ModelRequestNode) and isinstance(result, CallToolsNode):
            # Retain completed turns even when request history is later compacted.
            content: list[dict[str, Any]] = []
            for part in result.model_response.parts:
                if isinstance(part, TextPart) and part.content:
                    content.append({"type": "text", "text": part.content})
                elif isinstance(part, ThinkingPart) and part.content:
                    content.append({"type": "thinking", "thinking": part.content})
                elif isinstance(part, ToolCallPart):
                    content.append(
                        {
                            "type": "tool_use",
                            "id": part.tool_call_id,
                            "name": part.tool_name,
                            "input": part.args_as_dict(),
                        }
                    )
            self.toolkit.messages.append({"role": "assistant", "content": content})
        if isinstance(node, CallToolsNode) and self.toolkit.capture is not None:
            self.toolkit.capture.settle(
                ctx.run_id,
                "SDK rejected the call or skipped it after final output.",
                ctx.run_step,
            )
        if isinstance(node, (ModelRequestNode, CallToolsNode)):
            self.emit_usage(self.usage + ctx.usage)
        return result

    @on_event(EnqueuedMessagesEvent)
    async def acknowledge_enqueued(
        self, ctx: RunContext, event: EnqueuedMessagesEvent
    ) -> None:
        submission = self.enqueued.pop(event.enqueue_id, None)
        if submission is not None and self.control is not None:
            if self.toolkit.stopping.is_set():
                self.control.message_discarded(submission.message_id)
            else:
                self.control.message_started(submission.message_id, submission.text)

    @on_event(PartEndEvent)
    async def emit_part(self, ctx: RunContext, event: PartEndEvent) -> None:
        part = event.part
        if not isinstance(part, (TextPart, ThinkingPart)) or not part.content:
            return
        kind = "thinking" if isinstance(part, ThinkingPart) else "text"
        self.toolkit.dashboard_events.emit(
            TranscriptEvent({"type": kind, "text": part.content})
        )
        if not self.interactive:
            logger.info("[%s] %s", kind, part.content)

    def emit_user(self, text: str) -> None:
        self.toolkit.messages.append({"role": "user", "content": text})
        self.toolkit.dashboard_events.emit(
            TranscriptEvent({"type": "user", "text": text})
        )

    def emit_usage(self, usage: RunUsage | None = None) -> None:
        if usage is None:
            usage = self.usage
        self.toolkit.dashboard_events.emit(
            UsageEvent(
                inp=usage.input_tokens,
                out=usage.output_tokens,
                tool_calls=self.toolkit.tool_calls,
            )
        )

    async def submit(self, message: DashboardMessage) -> int:
        self.inbox.put_nowait(message)
        return 1

    async def interrupt(self) -> int:
        """Discard unstarted submissions, then cancel and drain the native run."""
        discarded = self.inbox.qsize()
        while not self.inbox.empty():
            submission = self.inbox.get_nowait()
            if self.control is not None:
                self.control.message_discarded(submission.message_id)
        if self.cancellation is not None:
            self.cancellation.cancel()
            await self.run_done.wait()
        # The run accounts for its own completion and SDK-enqueued messages.
        # Only submissions that never entered it need to be subtracted here.
        return discarded

    async def run_cli(self, agent: _ConversationAgent, prompt: str) -> None:
        """Run the preset task, then hand its history to the native CLI."""
        from rich.console import Console

        console = Console()
        console.print("Task context:\n" + prompt, markup=False, highlight=False)
        console.print(
            "Starting the task. You can enter follow-up instructions after it "
            "responds, or use /exit to close.",
            markup=False,
        )
        with console.status("Running task…"):
            result = await agent.run(
                prompt,
                conversation_id=self.conversation_id,
                usage_limits=self.limits,
            )
        console.print(result.output, markup=False, highlight=False)
        await agent.to_cli(
            prog_name="rpent",
            message_history=result.all_messages(),
            usage_limits=self.limits,
        )

    async def run(self, agent: _ConversationAgent, prompt: str) -> None:
        tasks: list[asyncio.Task] = []
        try:
            if self.interactive:
                await self.run_cli(agent, prompt)
                return
            if self.control is None:
                self.emit_user(prompt)
                await self._run_conversation(agent, prompt)
                return
            await self.control.start()
            tasks = [
                asyncio.create_task(self.control.run(self)),
                asyncio.create_task(self._run_conversation(agent, prompt)),
            ]
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                await task
        finally:
            if self.control is not None:
                self.control.end()
            # Stop new physical work before cancelling the SDK task. Toolkit
            # cancellation is cooperative, and must finish before solve returns.
            self.toolkit.stopping.set()
            if self.narrator is not None:
                await self.narrator.stop()
            if self.cancellation is not None:
                self.cancellation.cancel()
            await cancel_and_wait(self.toolkit.cancel_active_and_wait)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _run_conversation(self, agent: _ConversationAgent, prompt: str) -> None:
        """Run the SDK, then wait for Dashboard follow-ups in the same conversation."""
        while True:
            self.cancellation = CancellationToken()
            self.run_done.clear()
            self.completions = 1
            try:
                await agent.run(
                    prompt,
                    usage_limits=self.limits,
                    conversation_id=self.conversation_id,
                    cancellation_token=self.cancellation,
                )
            except RunCancelled:
                self.stop_reason = "cancelled"
                self.error = None
            finally:
                self.cancellation = None
                self.run_done.set()
                if self.control is not None:
                    for pending in self.enqueued.values():
                        self.control.message_discarded(pending.message_id)
                self.enqueued.clear()
            if self.toolkit.finish_result is not None or self.control is None:
                return
            if self.usage.requests >= self.limits.request_limit:
                return
            for _ in range(self.completions):
                # run_done already released interrupt(); wait for its input
                # cleanup before consuming the next conversation submission.
                await self.control.complete(self, wait_for_interrupt=True)
            submission = await self.inbox.get()
            self.control.message_started(submission.message_id, submission.text)
            prompt = submission.text
