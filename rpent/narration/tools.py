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

"""Planner tools that start and stop background scene narration."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from rpent.narration.config import MAX_INTERVAL_S, MIN_INTERVAL_S
from rpent.narration.narrator import Narrator
from rpent.tools.base import ToolResult, tool


class NarrationTools:
    """Expose one run's narrator to the planner."""

    def __init__(self, narrator: Narrator) -> None:
        self._narrator = narrator

    @tool(readonly=True)
    def narration_start(
        self,
        interval_s: Annotated[
            float | None, Field(ge=MIN_INTERVAL_S, le=MAX_INTERVAL_S)
        ] = None,
        focus: Annotated[str | None, Field(max_length=200)] = None,
    ) -> ToolResult:
        """Start spoken narration of the camera view in the background. Call once; it returns immediately and keeps describing each new observation frame and saving it as speech while you use other tools. It never moves the robot. Calling it again while narration runs only updates interval_s or focus. Narration stops automatically when the task ends.

        Args:
            interval_s: Seconds between checks for a new frame. Omit to keep the current setting.
            focus: Short hint about what descriptions should emphasize, such as the object being manipulated. An empty string clears the hint.
        """
        return ToolResult(data=self._narrator.start(interval_s=interval_s, focus=focus))

    @tool(readonly=True)
    def narration_stop(self) -> ToolResult:
        """Stop starting new narration. Sentences already described are still converted to speech. Optional: narration also stops when the task ends."""
        return ToolResult(data=self._narrator.stop())
