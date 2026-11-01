'''Video clips with pre-roll, and overlays that follow the capture time.'''
import time

import cv2 as cv
import numpy as np

from annotations import AnnotationLog, render
from capturethread import CaptureThread
from detections import BoundingRect, TFLDetection
from framebus import FrameBus
from videorecorder import ClipRecorder

FPS = 30


class Source:
    def __init__(self, count):
        self.count, self.fps, self.isopen, self.n = count, FPS, True, 0

    def get_image(self):
        if self.n >= self.count:
            self.isopen = False
            return None
        self.n += 1
        return np.full((120, 160, 3), 100, np.uint8)


def frame_count(path):
    cap = cv.VideoCapture(path)
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    return n


def record(tmp_path, annotations=None, trigger_after=1.0, total=90, **kw):
    bus = FrameBus(maxlen=60)
    rec = ClipRecorder(bus, str(tmp_path), fps=FPS, preroll=0.5, postroll=0.5,
                       annotations=annotations, overlay_delay=0.1, **kw)
    capture = CaptureThread(Source(total), bus)
    capture.start()
    time.sleep(trigger_after)
    rec.trigger(bus.latest().ts)
    capture.join(5)
    rec.close()
    return rec


def test_clip_includes_preroll_and_postroll(tmp_path):
    rec = record(tmp_path)
    assert len(rec.clips) == 1
    n = frame_count(rec.clips[0])
    # 0.5 s before + 0.5 s after the trigger at 30 fps = 30 frames
    assert 24 <= n <= 36, n


def test_trigger_extends_the_clip_and_one_file_results(tmp_path):
    bus = FrameBus(maxlen=60)
    rec = ClipRecorder(bus, str(tmp_path), fps=FPS, preroll=0.2, postroll=0.4)
    capture = CaptureThread(Source(90), bus)
    capture.start()
    for _ in range(5):                       # activity every 0.3 s keeps it going
        time.sleep(0.3)
        rec.trigger(bus.latest().ts)
    capture.join(5)
    rec.close()
    assert len(rec.clips) == 1
    assert frame_count(rec.clips[0]) > 1.5 * FPS


def test_no_trigger_no_file(tmp_path):
    bus = FrameBus()
    rec = ClipRecorder(bus, str(tmp_path), fps=FPS)
    capture = CaptureThread(Source(10), bus)
    capture.start()
    capture.join(5)
    rec.close()
    assert rec.clips == [] and list(tmp_path.iterdir()) == []


def test_clip_with_overlays_runs(tmp_path):
    ann = AnnotationLog()
    bus = FrameBus(maxlen=60)
    rec = ClipRecorder(bus, str(tmp_path), fps=FPS, preroll=0.3, postroll=0.3,
                       annotations=ann, overlay_delay=0.1)
    capture = CaptureThread(Source(60), bus)
    capture.start()
    time.sleep(0.5)
    ts = bus.latest().ts
    ann.add(ts, 0, [TFLDetection("Cat-alone", 0, 0.9, BoundingRect(5, 5, 50, 50))], "triggeringState")
    rec.trigger(ts)
    capture.join(5)
    rec.close()
    assert len(rec.clips) == 1 and frame_count(rec.clips[0]) > 10


def test_nearest_annotation_by_time():
    ann = AnnotationLog()
    ann.add(1.0, 10, [], "a")
    ann.add(2.0, 20, [], "b")
    assert ann.nearest(1.9).frame_id == 20
    assert ann.nearest(1.3).frame_id == 10
    assert ann.nearest(5.0) is None


def test_render_draws_on_a_copy():
    image = np.zeros((100, 100, 3), np.uint8)
    ann = AnnotationLog()
    ann.add(0, 0, [TFLDetection("Cat", 0, 0.5, BoundingRect(10, 10, 40, 40))], "state")
    out = render(image, ann.nearest(0))
    assert out.any() and not image.any()
    assert render(image, None) is image
