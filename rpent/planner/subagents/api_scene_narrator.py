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

"""Pydantic AI agent for spoken narration of robot observations."""

from __future__ import annotations

from pydantic_ai import Agent
from pydantic_ai_harness.compaction import SlidingWindowCompaction

from rpent.planner.api_loop import (
    ApiAgentLoop,
    ApiSubAgent,
    _build_model_settings,
    _HarnessToolkit,
)
from rpent.tools.toolkit import Toolkit
from rpent.utils.mcp_servers import McpServerSpec

#: Model-request budget for one scene narration session. This is independent of
#: the main conversation's ``max_turns``.
_NARRATOR_MAX_REQUESTS = 200


class ApiSceneNarrator(ApiSubAgent):
    """Observe the scene and narrate it through the configured speech tools."""

    description = (
        "Start the background scene narrator. It repeatedly reads "
        "the latest robot observation and speaks one short Chinese "
        "sentence through the external text-to-speech MCP tool until "
        "the episode ends. Call it once when the user asks for "
        "spoken narration; do not wait for it."
    )

    def __init__(self, planner: ApiAgentLoop, recorder: _HarnessToolkit) -> None:
        observe, speak = self._resolve_narrator_sources(
            recorder.toolkit, planner.mcp_servers
        )
        assert planner.subagent_model is not None
        narrator_toolkit = _HarnessToolkit(
            recorder.toolkit,
            dashboard_events=planner.dashboard_events,
            no_images=False,
            capture=None,
            sequential=True,
        )
        external = planner._external_toolsets(
            narrator_toolkit, servers=speak, id_suffix="_narrator"
        )
        prompt = (
            "You narrate a robot's camera view continuously. Repeat this round "
            "until told to stop:\n"
            "1. Call view_env_state(step=-1) or read_image for the latest "
            "observation and look at agentview_high.png.\n"
            "2. Write one Chinese sentence of at most 20 characters about the "
            "arm and the main objects, without Markdown. If the view has not "
            "changed since your last round, describe a different detail.\n"
            "3. Call a text-to-speech tool whose name ends with '_synthesize' "
            "once with exactly that sentence, and wait for it to return before "
            "starting the next round.\n"
            "Start the next round immediately, whether or not the view changed. "
            "Stop and reply with a one-line summary only when view_env_state "
            "reports terminated or truncated as true, or a tool reports that "
            "the session is closed. Do not call any other tool."
        )
        agent = Agent(
            planner.subagent_model,
            name="scene_narrator",
            system_prompt=prompt,
            output_type=str,
            model_settings=_build_model_settings(
                planner.subagent_model,
                min(planner.max_tokens, 1024),
                "none",
                parallel_tool_calls=False,
            ),
            capabilities=[
                SlidingWindowCompaction(
                    max_messages=40, max_fraction=0.8, keep_messages=20
                ),
            ],
        )
        super().__init__(
            agent,
            recorder,
            planner.dashboard_events,
            prompt="Start narrating the scene.",
            request_limit=_NARRATOR_MAX_REQUESTS,
            tools=[
                (narrator_toolkit, observe),
                *((toolset, (f"{toolset.prefix}_synthesize",)) for toolset in external),
            ],
        )

    @staticmethod
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
            raise ApiSceneNarrator.ConfigurationError(
                "the scene_narrator subagent needs an external MCP server listing "
                "synthesize under planner_tools"
            )
        return observe, speak
