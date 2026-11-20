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

"""Claude Agent SDK definition for spoken scene narration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rpent.planner.base import add_mcp_prefix

if TYPE_CHECKING:
    from rpent.tools import Toolkit
    from rpent.utils.mcp_servers import McpServerSpec


def build_scene_narrator_agent(
    sdk: Any,
    *,
    toolkit: Toolkit,
    servers: tuple[McpServerSpec, ...],
    model: str,
) -> Any:
    """Define a read-only subagent that keeps narrating the camera view aloud.

    The subagent may only read observations and call an external
    ``synthesize`` tool; it never receives robot action tools. It runs in the
    background so the main agent keeps acting while it narrates, starting each
    round after the previous synthesis returns until the episode ends or its
    turn limit is reached.

    Raises:
        ValueError: If the toolkit has no observation tool or no attached
            external MCP server exposes ``synthesize``.
    """
    from rpent.utils.mcp_servers import ALL_TOOLS

    names = {declaration.name for declaration in toolkit.list_tools()}
    observe = [
        add_mcp_prefix(name)
        for name in ("view_env_state", "read_image")
        if name in names
    ]
    if not observe:
        raise ValueError(
            "the scene_narrator subagent needs a view_env_state or read_image tool"
        )
    speak = [
        f"mcp__{server.name}__synthesize"
        for server in servers
        if server.planner_tools == ALL_TOOLS
        or (server.planner_tools and "synthesize" in server.planner_tools)
    ]
    if not speak:
        raise ValueError(
            "the scene_narrator subagent needs an external MCP server listing "
            "synthesize under planner_tools"
        )
    return sdk.AgentDefinition(
        description=(
            "Keeps narrating the robot's camera view aloud until the episode "
            "ends. Launch it once, in the background, when asked to narrate; "
            "do not launch a second one while it runs. If it returns before "
            "the task is finished, launch it again. It cannot move the robot."
        ),
        prompt=(
            "You narrate a robot's camera view continuously. Repeat this round "
            "until told to stop:\n"
            "1. Call view_env_state(step=-1) or read_image for the latest "
            "observation and look at agentview_high.png.\n"
            "2. Write one Chinese sentence of at most 20 characters about the "
            "arm and the main objects, without Markdown. If the view has not "
            "changed since your last round, describe a different detail.\n"
            "3. Call synthesize once with exactly that sentence and wait for it "
            "to return before starting the next round.\n"
            "Start the next round immediately, whether or not the view changed. "
            "Stop and reply with a one-line summary only when view_env_state "
            "reports terminated or truncated as true, or a tool reports that "
            "the session is closed. Do not call any other tool."
        ),
        tools=observe + speak,
        model=model,
        maxTurns=200,
        background=True,
    )
