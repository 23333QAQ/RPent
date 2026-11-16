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

import pytest

from rpent.narration import load_narration_config, resolve_tts_server
from rpent.narration.config import parse_narration_config
from rpent.utils.mcp_servers import McpServerSpec, load_site_mcp_servers

REQUIRED = {
    "tts_server": "tts",
    "model": "vision",
    "base_url": "https://example.test/v1",
    "api_key": "secret",
}


def test_site_config_resolves_env_secrets_beside_mcp_servers(tmp_path, monkeypatch):
    monkeypatch.setenv("NARRATION_KEY", "from-env")
    path = tmp_path / "site.yaml"
    path.write_text(
        "mcp_servers:\n"
        "  tts:\n"
        "    url: http://127.0.0.1:8766/mcp\n"
        "narration:\n"
        "  tts_server: tts\n"
        "  model: vision\n"
        "  base_url: https://example.test/v1\n"
        "  api_key: ${oc.env:NARRATION_KEY}\n"
        "  cameras: [wrist.png]\n"
        "  interval_s: 3\n"
    )

    config = load_narration_config(path)
    servers = load_site_mcp_servers(path)

    assert config.api_key == "from-env"
    assert config.cameras == ("wrist.png",)
    assert config.interval_s == 3.0
    assert "from-env" not in repr(config)
    assert servers[0].planner_tools is None
    assert resolve_tts_server(config, servers) is servers[0]


def test_absent_section_or_path_disables_narration(tmp_path):
    path = tmp_path / "site.yaml"
    path.write_text("mcp_servers: {}\n")

    assert load_narration_config(None) is None
    assert load_narration_config(path) is None


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"model": ""}, "narration.model"),
        ({"base_url": "ftp://x"}, "base_url"),
        ({"interval_s": 0.1}, "interval_s"),
        ({"interval_s": True}, "interval_s"),
        ({"max_chars": 0}, "max_chars"),
        ({"cameras": "agentview.png"}, "cameras"),
        ({"surprise": 1}, "unknown key"),
    ],
)
def test_invalid_fields_name_the_field(override, message):
    with pytest.raises(ValueError, match=message):
        parse_narration_config({**REQUIRED, **override})


def test_missing_required_keys_are_listed():
    with pytest.raises(ValueError, match=r"\['api_key', 'model'\]"):
        parse_narration_config({"tts_server": "tts", "base_url": "https://x"})


def test_undeclared_tts_server_is_rejected():
    config = parse_narration_config(REQUIRED)
    other = McpServerSpec(name="lights", url="http://127.0.0.1:1/mcp")

    with pytest.raises(ValueError, match="'tts' is not declared"):
        resolve_tts_server(config, (other,))
