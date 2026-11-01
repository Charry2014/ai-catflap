'''State machine driven by events, and the analysis thread feeding it.'''
import os
import queue
import threading
import time
from types import SimpleNamespace

import importlib

import numpy as np
import pytest

import stubs  # noqa: F401
pytest.importorskip("statemachine")

from analysisthread import AnalysisThread  # noqa: E402
from capturethread import CaptureThread  # noqa: E402
from detections import BoundingRect, TFLDetection  # noqa: E402
from framebus import FrameBus  # noqa: E402
from motion import MotionBox, MotionDetector  # noqa: E402
from statetypes import Event, EventKind, GlobalData, States  # noqa: E402

CONFIG = os.path.join(os.path.dirname(__file__), '..', 'src', 'catflap')


def img(v=0):
    return np.full((100, 100, 3), v, np.uint8)


def cat(score=0.8, label="Cat-alone"):
    return [TFLDetection(label, 0, score, BoundingRect(10, 10, 50, 50))]


class FakeAnalysis:
    armed = None

    def set_armed(self, armed):
        self.armed = armed


class FakeClips:
    def __init__(self):
        self.triggers = []

    def trigger(self, ts):
        self.triggers.append(ts)

    def close(self):
        pass


def fresh_state_machine_class():
    '''The state objects are created once per class when the module is imported, and
    python-statemachine binds them to the first machine. Reload so every test starts clean.'''
    import stateidle, statetriggering, statemovementlocked, statemouselocked, stateunlocked, states
    for module in (stateidle, statetriggering, statemovementlocked, statemouselocked, stateunlocked, states):
        importlib.reload(module)
    return states.CatFlapFSM


@pytest.fixture
def fsm(tmp_path):
    args = SimpleNamespace(trigger="0,0,100,100", headless=True, record_path=str(tmp_path),
                           record_overlays=False,
                           label_json=os.path.join(CONFIG, 'labels.json'),
                           trigger_json=os.path.join(CONFIG, 'trigger_config.json'),
                           eval_json=os.path.join(CONFIG, 'eval_config.json'))
    data = GlobalData(args, event=Event(img(), 0.0, 0))
    data.events = queue.Queue()
    data.analysis = FakeAnalysis()
    data.clips = FakeClips()
    machine = fresh_state_machine_class()(data)
    try:
        yield machine, data
    finally:
        machine.exit()
        data.close()


def motion(ts):
    return Event(img(), ts, 1, motion=[MotionBox(0, 0, 60, 60, 3000)], armed=False)


def detected(ts, detections):
    return Event(img(), ts, 2, detections=detections, armed=True)


def test_full_sequence_arms_locks_unlocks_and_times_out(fsm):
    machine, data = fsm
    assert machine.current_state.name == States.IDLE and data.analysis.armed is False

    machine.event_handle(motion(1.0))
    assert machine.current_state.name == States.TRIGGERING
    assert data.analysis.armed is True            # ML model now runs
    assert data.clips.triggers == [1.0]           # clip started from the motion frame

    machine.event_handle(detected(1.1, cat()))
    assert machine.current_state.name == States.MOVEMENT_LOCKED
    assert data.cat_flap_control.locked

    for i in range(6):
        machine.event_handle(detected(1.2 + i / 10, cat()))
    assert machine.current_state.name == States.UNLOCKED
    assert not data.cat_flap_control.locked
    assert len(data.clips.triggers) > 3           # kept alive while active
    assert len(data.annotations._items) >= 7      # results recorded against capture time

    machine.event_handle(Event(None, kind=EventKind.TIMEOUT))
    assert machine.current_state.name == States.IDLE
    assert data.analysis.armed is False


def test_stale_results_are_ignored(fsm):
    machine, data = fsm
    machine.event_handle(detected(0.5, cat()))                  # ML result while idle
    assert machine.current_state.name == States.IDLE
    machine.event_handle(motion(1.0))
    machine.event_handle(motion(1.1))                           # movement result while triggering
    assert machine.current_state.name == States.TRIGGERING


def test_no_cat_goes_back_to_idle(fsm):
    machine, data = fsm
    machine.event_handle(motion(1.0))
    for i in range(4):
        machine.event_handle(detected(1.1 + i / 10, []))         # nothing recognised = background
    assert machine.current_state.name == States.IDLE


def test_timeout_in_idle_is_harmless(fsm):
    machine, data = fsm
    machine.event_handle(Event(None, kind=EventKind.TIMEOUT))
    assert machine.current_state.name == States.IDLE


def test_timer_thread_posts_instead_of_touching_state(fsm):
    machine, data = fsm
    threading.Thread(target=machine._timeout_handle, args=(None,)).start()
    assert data.events.get(timeout=2).kind == EventKind.TIMEOUT


class FakeSource:
    def __init__(self, count, fps):
        self.count, self.fps, self.isopen, self.n = count, fps, True, 0

    def get_image(self):
        if self.n >= self.count:
            self.isopen = False
            return None
        self.n += 1
        return img(10)


class CountingDetector:
    def __init__(self, delay=0.0):
        self.calls, self.delay = 0, delay

    def detect(self, image):
        self.calls += 1
        time.sleep(self.delay)
        yield from cat()


def run_pipeline(source, detector, armed):
    bus, events = FrameBus(), queue.Queue(maxsize=200)
    analysis = AnalysisThread(bus, events, detector, MotionDetector(0, 0, 100, 100))
    analysis.set_armed(armed)
    capture = CaptureThread(source, bus)
    analysis.start()
    capture.start()
    got = []
    while True:
        e = events.get(timeout=5)
        got.append(e)
        if e.kind == EventKind.END:
            break
    return analysis, got


def test_analysis_does_not_run_the_model_while_idle():
    detector = CountingDetector()
    analysis, got = run_pipeline(FakeSource(15, 100), detector, armed=False)
    assert detector.calls == 0
    frames = [e for e in got if e.kind == EventKind.FRAME]
    assert frames and not any(e.armed for e in frames)


def test_slow_model_skips_frames_and_does_not_queue():
    detector = CountingDetector(delay=0.1)            # 10 Hz model, 100 Hz camera
    analysis, got = run_pipeline(FakeSource(60, 100), detector, armed=True)
    assert analysis.skipped > 20
    frames = [e for e in got if e.kind == EventKind.FRAME]
    assert all(e.armed and e.detections for e in frames)
    assert got[-1].kind == EventKind.END and not analysis.failed


def test_detector_error_ends_the_stream_and_is_reported():
    class Broken:
        def detect(self, image):
            raise RuntimeError("boom")
    analysis, got = run_pipeline(FakeSource(5, 100), Broken(), armed=True)
    assert analysis.failed and got[-1].kind == EventKind.END
