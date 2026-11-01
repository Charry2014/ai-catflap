'''Movement detection and the idle state, with the hardware modules stubbed out.'''
from types import SimpleNamespace

import numpy as np

import stubs  # noqa: F401  (stubs RPi.GPIO)
from motion import MotionBox, MotionDetector
import stateidle
from statetypes import Event, States


def frame(v, box=False):
    f = np.full((100, 100, 3), v, np.uint8)
    if box:
        f[20:80, 20:80] = 255
    return f


def test_static_scene_has_no_motion():
    m = MotionDetector(0, 0, 100, 100)
    for i in range(30):
        assert m.process(i / 15, frame(10)) == []


def test_movement_found_against_a_second_old_reference():
    m = MotionDetector(0, 0, 100, 100)
    for i in range(20):
        m.process(i / 15, frame(10))
    boxes = m.process(20 / 15, frame(10, box=True))
    assert boxes and max(b.w * b.h for b in boxes) > 2000


def test_boxes_are_in_full_frame_coordinates():
    m = MotionDetector(50, 50, 150, 150)
    base = np.full((200, 200, 3), 10, np.uint8)
    moved = base.copy()
    moved[70:130, 70:130] = 255
    m.process(0.0, base)
    box = m.process(1.2, moved)[0]
    assert 50 <= box.x <= 80 and 50 <= box.y <= 80


def test_history_is_bounded_and_reset_works():
    m = MotionDetector(0, 0, 100, 100)
    for i in range(150):
        m.process(i / 15, frame(10))
    assert len(m._history) <= 17
    m.reset()
    assert len(m._history) == 0


class FakeData:
    headless = True

    def __init__(self):
        self.recorded = []
        self.timeout_timer = SimpleNamespace(start=lambda: None)

    def record_image(self, image, label, detections=None):
        self.recorded.append(label)


def test_idle_triggers_only_on_big_boxes():
    state, data = stateidle.IdleState(name=States.IDLE), FakeData()  # looked up now: other tests reload the module
    small = Event(frame(1), 0, 0, motion=[MotionBox(0, 0, 10, 10, 100)])
    big = Event(frame(1), 0, 1, motion=[MotionBox(0, 0, 60, 60, 3000)])
    assert state.run(small, data) == States.IDLE
    assert state.run(big, data) == States.TRIGGERING
    assert data.recorded == ["movement"]
