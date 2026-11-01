'''The whole pipeline from a video file to recordings, with the ML model faked.'''
import importlib
import os
from types import SimpleNamespace

import cv2 as cv
import numpy as np
import pytest

import stubs  # noqa: F401
pytest.importorskip("statemachine")

CONFIG = os.path.join(os.path.dirname(__file__), '..', 'src', 'catflap')


class FakeDetector:
    def __init__(self, *a, **kw):
        pass

    def detect(self, image):
        from detections import BoundingRect, TFLDetection
        yield TFLDetection("Cat-alone", 0, 0.9, BoundingRect(220, 190, 100, 100))


def make_video(path, frames=60, fps=15):
    writer = cv.VideoWriter(path, cv.VideoWriter_fourcc(*'mp4v'), fps, (640, 480))
    for i in range(frames):
        f = np.full((480, 640, 3), 30, np.uint8)
        if i >= 30:                      # something big appears in the trigger area after 2 s
            f[200:420, 230:420] = 255
        writer.write(f)
    writer.release()


def test_video_file_to_stills_and_clip(tmp_path):
    video = str(tmp_path / "in.mp4")
    make_video(video)
    out = tmp_path / "rec"

    import stateidle, statetriggering, statemovementlocked, statemouselocked, stateunlocked, states
    for m in (stateidle, statetriggering, statemovementlocked, statemouselocked, stateunlocked, states):
        importlib.reload(m)
    import main
    importlib.reload(main)
    main.make_detector = lambda *a, **kw: FakeDetector()

    args = SimpleNamespace(
        stream=video, web=None, trigger="210,180,250,280", headless=True, record_path=str(out),
        record_overlays=True, record_video=True, preroll=1.0, postroll=1.0,
        label_json=os.path.join(CONFIG, 'labels.json'),
        trigger_json=os.path.join(CONFIG, 'trigger_config.json'),
        eval_json=os.path.join(CONFIG, 'eval_config.json'),
        model="x", enable_edgetpu=False, num_threads=1)
    assert main.main_loop(args) == 0

    names = sorted(os.listdir(out))
    assert any(n.startswith("movement") for n in names), names
    assert any(n.startswith("locking") for n in names), names
    clips = [n for n in names if n.endswith(".mp4")]
    assert len(clips) == 1, names
    cap = cv.VideoCapture(str(out / clips[0]))
    n = 0
    while cap.read()[0]:
        n += 1
    assert n > 15      # pre-roll + the event + post-roll
