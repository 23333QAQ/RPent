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

from collections.abc import Iterator
from contextlib import contextmanager
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
    "load_narration_config",
    "narration_session",
    "resolve_tts_server",
]


@contextmanager
def narration_session(
    config: NarrationConfig | None,
    *,
    toolkit: Toolkit,
    mcp_servers: tuple[McpServerSpec, ...],
    output_dir: str | Path,
) -> Iterator[Narrator | None]:
    """Register narration tools on ``toolkit`` for one planner run.

    Narration itself starts only when the planner calls ``narration_start``.
    On exit any active narration is stopped and its queue drained.

    Args:
        config: The site ``narration`` section, or ``None`` to do nothing.
        toolkit: The run's Toolkit; its ``state`` supplies camera frames.
        mcp_servers: Merged robot and site MCP servers.
        output_dir: Directory for ``narration.jsonl``.

    Raises:
        ValueError: If ``config.tts_server`` is not a declared MCP server.
    """
    if config is None:
        yield None
        return
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
    try:
        yield narrator
    finally:
        narrator.close()
