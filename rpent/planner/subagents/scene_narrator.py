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

"""Shared scene narration instructions and factories for each planner backend."""

from __future__ import annotations

import asyncio
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from rpent.planner.api_loop import ApiAgentLoop, ApiSubAgent, _HarnessToolkit
    from rpent.planner.codex import CodexSubAgent
    from rpent.tools import Toolkit

NARRATOR_NAME = "scene_narrator"

#: API request budget, independent of the main conversation's ``max_turns``.
_NARRATOR_MAX_REQUESTS = 200
_NARRATOR_READ_INTERVAL_S = 5.0

_DESCRIPTION = (
    "Keeps narrating the robot's camera view aloud until the episode ends. "
    "Launch it once, in the background, when asked to narrate; do not launch "
    "a second one while it runs. If it returns before the task is finished, "
    "launch it again. It cannot move the robot."
)

_PROMPT = (
    "Narrate the robot's camera view until asked to stop. Use only the "
    "observation and speech tools. Repeat these steps:\n"
    "1. Read the latest observation and inspect a camera image, preferably one "
    "showing the robot and its surroundings. If a camera image is listed and "
    "an image reader is available, you can open it by its reported filename. "
    "If the observation indicates that the episode has ended or a tool reports "
    "that the session is closed, stop and reply with a one-line summary.\n"
    "2. Describe what you see in one Chinese sentence of at most 20 characters, "
    "without Markdown. If the view is unchanged, describe a different detail.\n"
    "3. Call a text-to-speech tool whose name ends with '__synthesize' "
    "once with exactly that sentence. Wait for it to return. If its result "
    "includes an observation, describe that image next; otherwise read the "
    "next observation."
)


def select_narrator_tools(
    toolkit: Toolkit, *, include_read_image: bool = False
) -> tuple[str, ...]:
    """Select observations and registered MCP synthesis tools for narration.

    Args:
        toolkit: Toolkit after external MCP discovery and registration.
        include_read_image: Whether the adapter supplies its own image reader.

    Returns:
        Plain tool names to expose through the selected planner adapter.

    Raises:
        ValueError: If observation or synthesis tools are unavailable.
    """
    names = tuple(tool.name for tool in toolkit.list_tools())
    observe = tuple(
        name
        for name in ("view_env_state", "read_image")
        if name in names or (name == "read_image" and include_read_image)
    )
    if not observe:
        raise ValueError("the narrator needs a view_env_state or read_image tool")
    speak = tuple(name for name in names if name.partition("__")[2] == "synthesize")
    if not speak:
        raise ValueError(
            "the narrator needs a registered MCP synthesize tool; "
            "attach a server with tools: [synthesize] using --mcp-config"
        )
    return observe + speak


def build_api_narrator(planner: ApiAgentLoop, recorder: _HarnessToolkit) -> ApiSubAgent:
    """Build an API narrator with independent history and a fresh toolset per run."""
    from pydantic_ai import (
        AbstractToolset,
        Agent,
        BinaryContent,
        ModelRetry,
        RunContext,
        ToolReturn,
    )
    from pydantic_ai.capabilities import Thinking
    from pydantic_ai_harness.compaction import SlidingWindowCompaction

    from rpent.planner.api_loop import (
        ApiSubAgent,
        _build_model_settings,
        _HarnessToolkit,
    )
    from rpent.planner.base import _wait_for_cleanup
    from rpent.session import EnvState
    from rpent.tools import ToolResult
    from rpent.utils.logging import get_logger

    latest: ToolReturn | None = None
    delivered: ToolReturn | None = None
    narration_store: EnvState | None = None
    changed = asyncio.Event()
    logger = get_logger(__name__)

    class NarrationToolkit(_HarnessToolkit):
        async def execute(
            self, name: str, arguments: dict[str, Any], ctx: RunContext
        ) -> ToolResult:
            if not self.toolkit.supports_background_mcp_tool(name):
                return await super().execute(name, arguments, ctx)
            error = next(self.validators[name].iter_errors(arguments), None)
            if error is not None:
                raise ModelRetry(f"Invalid arguments for {name}: {error.message}")
            self._record_call(name, arguments, ctx)
            result = await self.toolkit.execute_background_mcp_tool(name, arguments)
            self._record_result(name, ctx, result.to_text())
            if self.stopping.is_set():
                ctx.cancel()
                raise asyncio.CancelledError
            return result

        async def call(
            self, name: str, ctx: RunContext, /, **arguments: Any
        ) -> ToolReturn:
            nonlocal delivered
            if name.partition("__")[2] == "synthesize":
                result = await super().call(name, ctx, **arguments)
                if latest is not None:
                    while latest is delivered:
                        changed.clear()
                        await changed.wait()
                    delivered = latest
                    if delivered is not None:
                        result.content = [
                            "Latest observation:\n"
                            + ToolResult(data=delivered.return_value).to_text(),
                            *(delivered.content or []),
                        ]
                return result
            if (
                name != "view_env_state"
                or latest is None
                or arguments.get("step", -1)
                not in (-1, None, latest.return_value["step"])
            ):
                return await super().call(name, ctx, **arguments)
            error = next(self.validators[name].iter_errors(arguments), None)
            if error is not None:
                raise ModelRetry(f"Invalid arguments for {name}: {error.message}")
            self._record_call(name, arguments, ctx)
            delivered = latest
            result = deepcopy(latest)
            result.return_value = ToolResult(data=result.return_value).to_text()
            self._record_result(name, ctx, result.return_value)
            return result

        def _read_image(self, name: str, step: int) -> ToolReturn:
            sampled = latest
            if sampled is not None and step in (-1, sampled.return_value["step"]):
                step = sampled.return_value["step"]
                if name == sampled.return_value.get("image"):
                    return ToolReturn(
                        return_value={"artifact": name, "step": step},
                        content=list(sampled.content or []),
                    )
            if step == -1:
                try:
                    record = self.toolkit.state.latest_saved_record()
                except RuntimeError as exc:
                    return ToolReturn(return_value={"error": str(exc)})
                if record is None:
                    return ToolReturn(
                        return_value={"error": "No completed observation is available."}
                    )
                step = record.step_idx
            return super()._read_image(name, step)

    def read_saved_observation() -> ToolReturn | None:
        nonlocal narration_store
        try:
            state = recorder.toolkit.state
        except RuntimeError:
            return None
        frame = recorder.toolkit.narration_frame()
        if frame is not None:
            if narration_store is None:
                narration_store = EnvState(state.output_dir / "narration" / uuid4().hex)
            metadata = {
                "source_sequence": frame["sequence"],
                "captured_at": frame["captured_at"],
                "sampled_at": time.time(),
            }
            with narration_store.record_step(
                state=metadata,
                terminated=frame["terminated"],
                truncated=frame["truncated"],
            ) as step:
                if narration_store.save("camera.jpg", frame["rgb"], step=step) is None:
                    raise OSError("Failed to save the narration camera image")
            return ToolReturn(
                return_value={
                    **metadata,
                    "step": step,
                    "source": "narration",
                    "terminated": frame["terminated"],
                    "truncated": frame["truncated"],
                    "image": "camera.jpg",
                },
                content=[
                    BinaryContent(
                        data=narration_store.load_bytes("camera.jpg", step=step),
                        media_type="image/jpeg",
                    )
                ],
            )
        record = state.latest_saved_record()
        if record is None:
            return None
        metadata = {
            "step": record.step_idx,
            "terminated": record.terminated,
            "truncated": record.truncated,
            "state": record.state,
            "artifacts": sorted(record.artifacts),
            "task_language": record.extras.get("task_language"),
        }
        for name in sorted(record.artifacts):
            if Path(name).suffix.lower() in {".png", ".jpg", ".jpeg"}:
                try:
                    data = state.load_bytes(name, step=record.step_idx)
                except FileNotFoundError:
                    continue
                metadata["image"] = name
                return ToolReturn(
                    return_value=metadata,
                    content=[
                        BinaryContent(
                            data=data,
                            media_type=(
                                "image/png"
                                if Path(name).suffix.lower() == ".png"
                                else "image/jpeg"
                            ),
                        )
                    ],
                )
        if record.terminated or record.truncated:
            return ToolReturn(return_value=metadata)
        return None

    async def read_observation() -> None:
        nonlocal latest
        worker = asyncio.create_task(asyncio.to_thread(read_saved_observation))
        try:
            latest = await asyncio.shield(worker)
        except asyncio.CancelledError:
            await _wait_for_cleanup(worker)
            raise
        except Exception:
            latest = None
            logger.exception("Failed to sample the narration image")
        changed.set()

    async def sample_observations() -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _NARRATOR_READ_INTERVAL_S
        while True:
            await asyncio.sleep(max(0.0, deadline - loop.time()))
            deadline = loop.time() + _NARRATOR_READ_INTERVAL_S
            await read_observation()

    # Define backend subclasses here so importing another role factory does not
    # load this backend's SDK or its planner adapter.
    class ApiSceneNarrator(ApiSubAgent):
        description = (
            "Start the background scene narrator. It repeatedly reads "
            "the latest robot observation and speaks one short Chinese "
            "sentence through the external text-to-speech MCP tool until "
            "the episode ends. Call it once when the user asks for "
            "spoken narration; do not wait for it."
        )

        async def _run(self) -> None:
            nonlocal latest, delivered, narration_store
            latest = None
            delivered = None
            narration_store = None
            changed.clear()
            await read_observation()
            sampling = asyncio.create_task(sample_observations())
            try:
                await super()._run()
            finally:
                sampling.cancel()
                try:
                    await sampling
                except asyncio.CancelledError:
                    pass

    allowed = select_narrator_tools(recorder.toolkit, include_read_image=True)
    agent = Agent(
        planner.subagent_model,
        name=NARRATOR_NAME,
        system_prompt=(
            _PROMPT + "\nCamera images are sampled and saved every five seconds in "
            "the background. view_env_state returns the latest sample. Use its "
            "embedded image directly. After synthesis, the tool supplies the "
            "next available sample; describe it immediately. Source frames can "
            "repeat while the environment is stationary."
        ),
        output_type=str,
        model_settings=_build_model_settings(
            planner.subagent_model,
            min(planner.max_tokens, 1024),
            "none",
            parallel_tool_calls=False,
        ),
        capabilities=[
            SlidingWindowCompaction(max_messages=3, keep_messages=2, receipts=False),
            Thinking(False),
        ],
    )

    @agent.toolset(per_run_step=False)
    def narration_tools(_ctx: RunContext) -> AbstractToolset:
        adapter = NarrationToolkit(
            recorder.toolkit,
            dashboard_events=planner.dashboard_events,
            no_images=False,
            sequential=True,
        )
        return adapter.filtered(lambda _ctx, tool: tool.name in allowed)

    return ApiSceneNarrator(
        agent,
        recorder,
        planner.dashboard_events,
        prompt="Start narrating the scene.",
        request_limit=_NARRATOR_MAX_REQUESTS,
    )


def build_claude_narrator(sdk: Any, *, toolkit: Toolkit, model: str) -> Any:
    """Build a background Claude role limited to observations and speech synthesis."""
    from rpent.planner.base import add_mcp_prefix

    return sdk.AgentDefinition(
        description=_DESCRIPTION,
        prompt=_PROMPT,
        tools=[add_mcp_prefix(name) for name in select_narrator_tools(toolkit)],
        model=model,
        maxTurns=200,
        background=True,
    )


def build_codex_narrator(*, toolkit: Toolkit, model: str | None) -> CodexSubAgent:
    """Build a Codex role using the session's filtered RPent MCP bridge."""
    from rpent.planner.codex import CodexSubAgent

    class CodexSceneNarrator(CodexSubAgent):
        name = NARRATOR_NAME
        description = (
            _DESCRIPTION + " Spawn it with "
            f'agent_type="{NARRATOR_NAME}" and fork_turns="none", and do not wait '
            "for it; continue the task."
        )

        def __init__(self) -> None:
            self.tools = select_narrator_tools(toolkit)
            self.model = model

        def render_config(
            self, mcp_url: str, *, model_provider: str | None = None
        ) -> str:
            """Allow only observation and synthesis tools on the RPent bridge."""
            lines = [
                'sandbox_mode = "read-only"',
                'approval_policy = "never"',
                "tools.web_search = false",
                f"developer_instructions = {json.dumps(_PROMPT)}",
                f"mcp_servers.rpent.url = {json.dumps(mcp_url)}",
                f"mcp_servers.rpent.enabled_tools = {json.dumps(self.tools)}",
                'mcp_servers.rpent.default_tools_approval_mode = "approve"',
            ]
            if self.model is not None:
                lines.append(f"model = {json.dumps(self.model)}")
            if model_provider is not None:
                lines.append(f"model_provider = {json.dumps(model_provider)}")
            return "\n".join(lines) + "\n"

    return CodexSceneNarrator()
