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

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from rpent.dashboard.events import NullDashboardEventSink
from rpent.planner.base import PlannerResult
from rpent.robots import PromptBundle, RobotSpec, RunConfig
from rpent.session import EnvState
from rpent.tools.toolkit import Toolkit

SITE = """\
mcp_servers:
  {server}:
    url: http://127.0.0.1:1/mcp
narration:
  tts_server: tts
  model: vision
  base_url: https://example.test/v1
  api_key: key
"""


class RobotToolkit(Toolkit):
    def __init__(self, state: EnvState) -> None:
        super().__init__(
            dashboard_events=NullDashboardEventSink(),
            memory=SimpleNamespace(),
            state=state,
        )

    def _register_common_tools(self) -> None:
        pass


def _run(monkeypatch, tmp_path: Path, *, server: str) -> list[set[str]]:
    from rpent.cli import main as cli

    seen: list[set[str]] = []
    toolkit = RobotToolkit(EnvState(tmp_path / "state"))

    class FakePlanner:
        def solve(self, **kwargs: Any) -> PlannerResult:
            seen.append({tool.name for tool in kwargs["toolkit"].list_tools()})
            return PlannerResult(
                finish_result={"status": "success"}, messages=[], stats={}, error=None
            )

    def add_cli_args(parser: Any, use_dashboard: bool) -> None:
        del use_dashboard

    robot_spec = RobotSpec(
        name="testrobot",
        prompts=PromptBundle(system=lambda v: "system", user=lambda v: "task"),
        add_cli_args=add_cli_args,
        parse_config=lambda args: RunConfig(
            recipe_tag="t",
            output_dir=Path(args.output_dir),
            prompt_vars={"memory_dir": args.memory_dir},
            task_desc={},
        ),
        init_runtime=lambda *args: ([], {}),
    )
    site = tmp_path / "site.yaml"
    site.write_text(SITE.format(server=server))
    monkeypatch.setattr(cli, "enumerate_robots", lambda: ("testrobot",))
    monkeypatch.setattr(cli, "get_robot_spec", lambda name: robot_spec)
    monkeypatch.setattr(cli, "build_planner", lambda *a, **k: FakePlanner())
    monkeypatch.setattr(cli, "get_toolkit", lambda *a, **k: toolkit)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rpent",
            "--robot",
            "testrobot",
            "--planner",
            "api",
            "--memory-profile",
            "local",
            "--memory-dir",
            str(tmp_path / "memory"),
            "--output-dir",
            str(tmp_path),
            "--site-config",
            str(site),
        ],
    )
    cli.main()
    return seen


def test_planner_receives_narration_tools_from_site_config(monkeypatch, tmp_path):
    assert _run(monkeypatch, tmp_path, server="tts") == [
        {"narration_start", "narration_stop"}
    ]


def test_undeclared_tts_server_fails_before_the_run(monkeypatch, tmp_path, capsys):
    with pytest.raises(SystemExit):
        _run(monkeypatch, tmp_path, server="lights")

    assert "narration.tts_server 'tts' is not declared" in capsys.readouterr().err
