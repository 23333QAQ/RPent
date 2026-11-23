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

"""Concrete subagents and backend-neutral subagent factories."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rpent.planner.api_loop import ApiAgentLoop, ApiSubAgent, _HarnessToolkit
    from rpent.tools import Toolkit
    from rpent.utils.mcp_servers import McpServerSpec


def _subagent_name(name: str | None) -> str:
    """Resolve the legacy omitted name to the only current subagent."""
    return name or "narrator"


def build_api_subagent(
    name: str | None,
    planner: "ApiAgentLoop",
    recorder: "_HarnessToolkit",
) -> "ApiSubAgent":
    """Build an API subagent selected by the planner input."""
    if _subagent_name(name) == "narrator":
        from rpent.planner.subagents.api_scene_narrator import ApiSceneNarrator

        return ApiSceneNarrator(planner, recorder)
    raise ValueError(f"unknown API subagent: {name}")


def build_claude_subagent(
    name: str | None,
    sdk,
    *,
    toolkit: "Toolkit",
    servers: tuple["McpServerSpec", ...],
    model: str,
):
    """Build a Claude Agent SDK subagent selected by the planner input."""
    if _subagent_name(name) == "narrator":
        from rpent.planner.subagents.claude_scene_narrator import (
            build_scene_narrator_agent,
        )

        return "scene_narrator", build_scene_narrator_agent(
            sdk, toolkit=toolkit, servers=servers, model=model
        )
    raise ValueError(f"unknown Claude subagent: {name}")


def build_codex_subagent(
    name: str | None,
    *,
    toolkit: "Toolkit",
    servers: tuple["McpServerSpec", ...],
    model: str | None,
):
    """Build a Codex subagent selected by the planner input."""
    if _subagent_name(name) == "narrator":
        from rpent.planner.subagents.codex_scene_narrator import CodexSceneNarrator

        return CodexSceneNarrator(toolkit, servers, model=model)
    raise ValueError(f"unknown Codex subagent: {name}")
