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

"""Task selection must preserve the requested benchmark identity."""

import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from robots.libero.env_server import make_env


@pytest.fixture
def backend(monkeypatch):
    counts = (2, 0, 4)
    queried = []

    def initial_states(task):
        queried.append(task)
        return list(range(counts[task]))

    suite = SimpleNamespace(n_tasks=len(counts), get_task_init_states=initial_states)
    constructor = Mock(side_effect=lambda **kwargs: kwargs["cfg"])
    for name in ("rlinf", "rlinf.envs", "rlinf.envs.sim", "rlinf.envs.sim.libero"):
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    module = types.ModuleType("rlinf.envs.sim.libero.libero_env")
    module.LiberoEnv = constructor
    monkeypatch.setitem(sys.modules, module.__name__, module)
    utils = types.ModuleType("rlinf.envs.sim.libero.utils")
    utils.benchmark = SimpleNamespace(get_benchmark=lambda _: lambda: suite)
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    return constructor, queried


@pytest.mark.parametrize("task", [-1, -3, 3, 10])
def test_invalid_task_cannot_alias_another_task(backend, task):
    constructor, queried = backend
    with pytest.raises(ValueError, match="task_id.*demo"):
        make_env(task_id=task, seed=0, suite_name="demo")
    assert not queried
    constructor.assert_not_called()


def test_empty_selected_task_is_rejected_before_construction(backend):
    constructor, _ = backend
    with pytest.raises(ValueError, match="demo.*task 1.*no initial states"):
        make_env(task_id=1, seed=0, suite_name="demo")
    constructor.assert_not_called()


@pytest.mark.parametrize(
    ("task", "seed", "reset_id"), [(0, 3, 1), (2, 0, 2), (2, 5, 3), (2, 7, 5)]
)
def test_valid_task_offsets_and_seed_wrapping_are_preserved(
    backend, task, seed, reset_id
):
    constructor, _ = backend
    config = make_env(task_id=task, seed=seed, suite_name="demo")
    assert config.specific_reset_id == reset_id
    assert config.seed == seed
    assert config.task_suite_name == "demo"
    constructor.assert_called_once()


def test_narration_reads_completed_step_during_chunk(
    backend, monkeypatch, make_server_and_client
):
    from robots.libero.env_client import LiberoEnvClient
    from robots.libero.env_server import LiberoEnvFacade, _NarrationFrames

    first_step_done = threading.Event()
    continue_chunk = threading.Event()

    class NativeEnv:
        def __init__(self, **kwargs):
            pass

        def reset(self):
            return {"main_images": np.zeros((1, 2, 2, 3), dtype=np.uint8)}, {}

        def step(self, actions, **kwargs):
            value = int(actions[0, 0])
            return (
                {"main_images": np.full((1, 2, 2, 3), value, dtype=np.uint8)},
                np.zeros(1),
                np.array([value == 1]),
                np.array([False]),
                {},
            )

        def chunk_step(self, actions):
            first = self.step(actions[:, 0], auto_reset=False)
            first_step_done.set()
            assert continue_chunk.wait(5)
            last = self.step(actions[:, 1], auto_reset=False)
            return (
                [first[0], last[0]],
                *(np.stack([first[i], last[i]], axis=1) for i in (1, 2, 3)),
                [{}, {}],
            )

    monkeypatch.setattr(
        sys.modules["rlinf.envs.sim.libero.libero_env"], "LiberoEnv", NativeEnv
    )
    frames = _NarrationFrames()
    env = make_env(0, 0, "demo", narration_frames=frames)
    facade = LiberoEnvFacade(env, meta={}, narration_frames=frames)
    with make_server_and_client(facade, "socket") as rpc, ThreadPoolExecutor() as pool:
        client = LiberoEnvClient(rpc, expected_meta={})
        initial = client.narration_frame()
        chunk = pool.submit(client.chunk_step, np.array([[1], [2]]))
        try:
            assert first_step_done.wait(3)
            during = client.narration_frame()
            assert not chunk.done()
            assert during["sequence"] == initial["sequence"] + 1
            assert during["captured_at"] >= initial["captured_at"]
            np.testing.assert_array_equal(during["rgb"], np.ones((2, 2, 3)))
            assert during["terminated"]
            assert not during["truncated"]
        finally:
            continue_chunk.set()
        chunk.result(timeout=3)
        final = client.narration_frame()
        assert final["sequence"] == during["sequence"] + 1
        assert final["terminated"]
        np.testing.assert_array_equal(final["rgb"], np.full((2, 2, 3), 2))
        client.reset()
        assert not client.narration_frame()["terminated"]
