'''Tests for the capture -> frame bus -> consumer plumbing (no camera or TFLite needed).
Run with: pytest test/'''
import threading
import time

import numpy as np

from backgroundworker import BackgroundWorker
from capturethread import CaptureThread
from framebus import FrameBus
from image_recorder import ImageRecorder


def img(v=0):
    return np.full((4, 4, 3), v, np.uint8)


class FakeSource:
    def __init__(self, count, fps=None):
        self.count, self.fps, self.isopen, self.n = count, fps, True, 0

    def get_image(self):
        if self.n >= self.count:
            self.isopen = False
            return None
        self.n += 1
        return img(self.n)


def test_bus_ids_and_readonly():
    bus = FrameBus()
    a, b = bus.publish(img()), bus.publish(img())
    assert (a.id, b.id) == (0, 1)
    assert not a.image.flags.writeable
    assert bus.latest() is b


def test_wait_newer_returns_newest_and_skips():
    bus = FrameBus()
    for i in range(5):
        bus.publish(img(i))
    f = bus.wait_newer(-1, timeout=0.1)
    assert f.id == 4
    assert bus.wait_newer(4, timeout=0.05) is None  # timeout, nothing newer


def test_wait_newer_wakes_on_publish_and_on_close():
    bus = FrameBus()
    threading.Timer(0.05, bus.publish, args=(img(),)).start()
    assert bus.wait_newer(-1, timeout=2).id == 0
    threading.Timer(0.05, bus.close).start()
    assert bus.wait_newer(0, timeout=2) is None and bus.closed


def test_ring_buffer_and_since():
    bus = FrameBus(maxlen=3)
    frames = [bus.publish(img(i)) for i in range(5)]
    assert [f.id for f in bus.since(0)] == [2, 3, 4]
    assert [f.id for f in bus.since(frames[3].ts)] == [3, 4]


def test_capture_thread_publishes_all_then_closes():
    bus = FrameBus()
    t = CaptureThread(FakeSource(10), bus)
    t.start()
    t.join(2)
    assert bus.published == 10 and bus.closed


def test_capture_thread_paces_file_sources():
    bus = FrameBus()
    start = time.monotonic()
    t = CaptureThread(FakeSource(6, fps=50), bus)
    t.start()
    t.join(2)
    assert time.monotonic() - start >= 0.08  # 5 intervals of 20 ms, not instant


def test_worker_drop_oldest_keeps_newest():
    gate, seen = threading.Event(), []

    def fn(x):
        gate.wait(2)
        seen.append(x)

    w = BackgroundWorker(fn, "t", maxsize=1, drop_oldest=True)
    w.submit(0)
    time.sleep(0.05)        # worker now holds 0
    w.submit(1)
    w.submit(2)             # drops 1
    gate.set()
    w.close()
    assert seen == [0, 2] and w.dropped == 1


def test_worker_survives_exception_and_never_blocks():
    seen = []

    def fn(x):
        if x == 1:
            raise RuntimeError("boom")
        seen.append(x)

    w = BackgroundWorker(fn, "t", maxsize=8)
    for i in range(4):
        w.submit(i)
    w.close()
    assert seen == [0, 2, 3]


def test_image_recorder_writes_in_background(tmp_path):
    rec = ImageRecorder(str(tmp_path / "new" / "dir"))  # created on demand
    rec(img(), "movement")
    rec(img(), "locking")
    rec.close()
    names = sorted(p.name for p in (tmp_path / "new" / "dir").iterdir())
    assert len(names) == 2 and names[0].startswith("locking") and names[1].startswith("movement")
