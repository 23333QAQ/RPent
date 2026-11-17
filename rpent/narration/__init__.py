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

"""Background scene narration controlled by planner tools.

A site config ``narration`` section enables two Toolkit tools for every
robot: ``narration_start`` and ``narration_stop``. Once started, a background
thread describes each new camera frame with its own vision-model requests and
sends the text to an external MCP ``synthesize`` tool. The planner's own
conversation, model, and tool scheduling are unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rpent.narration.config import (
    NarrationConfig,
    load_narration_config,
    resolve_tts_server,
)

if TYPE_CHECKING:
    from rpent.narration.narrator import Narrator
    from rpent.tools.toolkit import Toolkit
    from rpent.utils.mcp_servers import McpServerSpec

__all__ = [
    "NarrationConfig",
    "attach_narration",
    "load_narration_config",
    "resolve_tts_server",
]


def attach_narration(
    site_config: str | Path | None,
    *,
    toolkit: Toolkit,
    mcp_servers: tuple[McpServerSpec, ...],
    output_dir: str | Path,
) -> Narrator | None:
    """Register narration tools on ``toolkit`` for one planner run.

    Call before the planner starts so the tools are listed. Narration itself
    starts only when the planner calls ``narration_start``. Closing the
    toolkit stops any active narration and drains its queue before the
    robot's resources are released.

    Args:
        site_config: Path given by ``--site-config``; without a ``narration``
            section, or with no path, nothing is registered.
        toolkit: The run's Toolkit; its ``state`` supplies camera frames.
        mcp_servers: Merged robot and site MCP servers.
        output_dir: Directory for ``narration.jsonl``.

    Returns:
        The narrator, or ``None`` when narration is not configured.

    Raises:
        ValueError: If the ``narration`` section is invalid or names an
            undeclared MCP server.
    """
    config = load_narration_config(site_config)
    if config is None:
        return None
    from rpent.narration.describer import SceneDescriber
    from rpent.narration.narrator import Narrator
    from rpent.narration.speaker import Speaker
    from rpent.narration.tools import NarrationTools
    from rpent.tools.base import iter_tools

    server = resolve_tts_server(config, mcp_servers)
    narrator = Narrator(
        config,
        state=toolkit.state,
        output_dir=output_dir,
        describer_factory=lambda: SceneDescriber(config),
        synthesizer_factory=lambda: Speaker(
            server, timeout_s=config.synthesis_timeout_s
        ),
    )
    toolkit.add_tools(iter_tools(NarrationTools(narrator)))
    toolkit.on_close(narrator.close)
    return narrator
