'''Idle-state motion detection with the hardware modules stubbed out.'''
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("statemachine")
gpio = types.ModuleType("RPi")
gpio.GPIO = types.SimpleNamespace(BCM=0, OUT=0, LOW=0, HIGH=1, setmode=lambda *a: None,
                                  setup=lambda *a: None, output=lambda *a: None,
                                  input=lambda *a: 0, cleanup=lambda: None)
sys.modules.setdefault("RPi", gpio)
sys.modules.setdefault("RPi.GPIO", gpio.GPIO)

from collections import deque  # noqa: E402
from stateidle import IdleState  # noqa: E402
from statetypes import Event, States  # noqa: E402


class FakeData:
    headless = True
    trigger_bc, trigger_br, trigger_bcw, trigger_brh = 0, 0, 100, 100

    def __init__(self):
        self.idle_history = deque()
        self.recorded = []
        self.timeout_timer = SimpleNamespace(start=lambda: None)

    def record_image(self, image, label):
        self.recorded.append(label)


def frame(v, box=False):
    f = np.full((100, 100, 3), v, np.uint8)
    if box:
        f[20:80, 20:80] = 255
    return f


def run(state, data, image, ts):
    return state.run(Event(image, ts), data)


def test_static_scene_stays_idle_and_does_not_sleep():
    state, data = IdleState(name=States.IDLE), FakeData()
    for i in range(30):
        assert run(state, data, frame(10), i / 15) == States.IDLE


def test_movement_triggers_even_when_arriving_between_old_samples():
    state, data = IdleState(name=States.IDLE), FakeData()
    for i in range(20):
        run(state, data, frame(10), i / 15)
    assert run(state, data, frame(10, box=True), 20 / 15) == States.TRIGGERING
    assert data.recorded == ["movement"]


def test_history_is_bounded():
    state, data = IdleState(name=States.IDLE), FakeData()
    for i in range(150):
        run(state, data, frame(10), i / 15)
    assert len(data.idle_history) <= 17
