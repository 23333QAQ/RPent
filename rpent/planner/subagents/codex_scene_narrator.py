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

"""Codex role for spoken narration of robot observations."""

from __future__ import annotations

from rpent.planner.codex import CodexSubAgent, _toml_value
from rpent.tools.toolkit import Toolkit
from rpent.utils.mcp_servers import McpServerSpec


class CodexSceneNarrator(CodexSubAgent):
    """Select the observation and speech tools for one Codex session."""

    name = "scene_narrator"
    description = (
        "Keeps narrating the robot's camera view aloud until the episode ends. "
        "Launch it once, in the background, when asked to narrate; do not launch "
        "a second one while it runs. If it returns before the task is finished, "
        "launch it again. It cannot move the robot. Spawn it with "
        'agent_type="scene_narrator" and fork_turns="none", and do not wait for '
        "it; continue the task."
    )

    def __init__(
        self, toolkit: Toolkit, servers: tuple["McpServerSpec", ...], *, model: str
    ) -> None:
        from rpent.utils.mcp_servers import ALL_TOOLS

        names = {declaration.name for declaration in toolkit.list_tools()}
        self.observe = tuple(
            name for name in ("view_env_state", "read_image") if name in names
        )
        if not self.observe:
            raise ValueError(
                "the scene_narrator subagent needs a view_env_state or read_image tool"
            )
        self.speak = tuple(
            server
            for server in servers
            if server.planner_tools == ALL_TOOLS
            or (server.planner_tools and "synthesize" in server.planner_tools)
        )
        if not self.speak:
            raise ValueError(
                "the scene_narrator subagent needs an external MCP server listing "
                "synthesize under planner_tools"
            )
        self.model = model
        self.servers = servers

    def render_config(self, mcp_url: str, *, model_provider: str | None = None) -> str:
        """Build the role for the session's live MCP endpoint."""
        from rpent.utils.mcp_servers import RESERVED_SERVER_NAME

        prompt = (
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
        )
        lines = [f"model = {_toml_value(self.model)}"]
        if model_provider:
            lines.append(f"model_provider = {_toml_value(model_provider)}")
        lines.extend(
            [
                'sandbox_mode = "read-only"',
                'approval_policy = "never"',
                "tools.web_search = false",
                f"developer_instructions = {_toml_value(prompt)}",
                f"mcp_servers.{RESERVED_SERVER_NAME}.url = {_toml_value(mcp_url)}",
                f"mcp_servers.{RESERVED_SERVER_NAME}.enabled_tools = "
                f"{_toml_value(list(self.observe))}",
            ]
        )
        for server in self.speak:
            lines.extend(
                [
                    f"mcp_servers.{server.name}.url = {_toml_value(server.url)}",
                    f"mcp_servers.{server.name}.enabled_tools = "
                    f"{_toml_value(['synthesize'])}",
                ]
            )
            if server.env_http_headers:
                lines.append(
                    f"mcp_servers.{server.name}.env_http_headers = "
                    f"{_toml_value(server.env_http_headers)}"
                )
            if server.headers:
                lines.append(
                    f"mcp_servers.{server.name}.http_headers = "
                    f"{_toml_value(server.headers)}"
                )
        speak_names = {server.name for server in self.speak}
        for server in self.servers:
            if (
                server.planner_tools is None
                or server.name in speak_names
                or server.name == RESERVED_SERVER_NAME
            ):
                continue
            lines.extend(
                [
                    f"mcp_servers.{server.name}.url = {_toml_value(server.url)}",
                    f"mcp_servers.{server.name}.enabled = false",
                ]
            )
        return "\n".join(lines) + "\n"
