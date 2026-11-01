'''The ai-edge-litert detector, with the real model.'''
import os
import shutil

import cv2 as cv
import numpy as np
import pytest

import stubs  # noqa: F401
pytest.importorskip("ai_edge_litert")

import litert_detect  # noqa: E402
from litert_detect import LiteRTDetect, read_labelmap  # noqa: E402

MODEL = os.path.join(os.path.dirname(__file__), 'cats.tflite')
LABELS = ['Cat-alone', 'Cat-with-mouse', 'Background', 'Cat-body']


def scene(seed=0, size=(640, 480)):
    rng = np.random.default_rng(seed)
    w, h = size
    im = np.full((h, w, 3), 60 + 25 * (seed % 6), np.uint8)
    cv.ellipse(im, (w // 2, h // 2), (w // 5, h // 6), 20 * seed, 0, 360, (30, 30, 30), -1)
    cv.rectangle(im, (w // 6, h * 5 // 8), (w // 3, h * 9 // 10), (90, 200, 90), -1)
    im = cv.GaussianBlur(im, (0, 0), 2)
    return np.clip(im.astype(int) + rng.normal(0, 5, im.shape), 0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def detector():
    return LiteRTDetect(MODEL, False, 2)


def test_labels_come_from_the_model():
    assert read_labelmap(MODEL) == LABELS
    assert read_labelmap(__file__) == []         # not a model: no labels, no crash


def test_detections_are_few_sorted_and_over_threshold(detector):
    for seed in range(8):
        found = list(detector.detect(scene(seed)))
        assert len(found) <= LiteRTDetect.MAX_RESULTS
        assert found == sorted(found, key=lambda d: -d.score)
        for d in found:
            assert d.score >= LiteRTDetect.SCORE_THRESHOLD
            assert d.label == LABELS[d.index]
            assert isinstance(d.box.origin_x, int) and d.box.width > 0 and d.box.height > 0
    assert len(detector) == len(found)


def test_boxes_follow_the_image_size(detector, monkeypatch):
    monkeypatch.setattr(LiteRTDetect, 'SCORE_THRESHOLD', 0.05)
    base = scene(1)                    # the same picture at two sizes
    small = list(detector.detect(cv.resize(base, (320, 240), interpolation=cv.INTER_AREA)))
    large = list(detector.detect(cv.resize(base, (1280, 960), interpolation=cv.INTER_LINEAR)))
    assert small and large and small[0].label == large[0].label
    for a, b in ((small[0].box.width, large[0].box.width), (small[0].box.height, large[0].box.height)):
        assert abs(a * 4 - b) < 0.15 * b


def test_nothing_below_the_threshold(detector, monkeypatch):
    monkeypatch.setattr(LiteRTDetect, 'SCORE_THRESHOLD', 0.999)
    assert list(detector.detect(scene(2))) == []


def test_scores_and_classes_are_told_apart_by_value(detector):
    expected = [(d.label, round(d.score, 4)) for d in detector.detect(scene(3))]
    detector._scores, detector._classes = detector._classes, detector._scores   # wrong way round
    assert [(d.label, round(d.score, 4)) for d in detector.detect(scene(3))] == expected


def test_labels_json_is_the_fallback(monkeypatch):
    monkeypatch.setattr(litert_detect, 'read_labelmap', lambda f: [])
    d = LiteRTDetect(MODEL, False, 1, labels=['a', 'b', 'c', 'd'])
    found = list(d.detect(scene(1)))
    assert found and found[0].label in 'abcd'


def test_rejects_a_model_that_is_not_a_detector(tmp_path):
    with pytest.raises(Exception):
        LiteRTDetect(str(tmp_path / "missing.tflite"), False, 1)


def test_agrees_with_tflite_support():
    pytest.importorskip("tflite_support")
    from tflite_detect import TFLiteDetect
    from tools import compare_detectors as cmp
    rows = cmp.compare_images([scene(i) for i in range(20)], TFLiteDetect(MODEL, False, 2), LiteRTDetect(MODEL, False, 2))
    same = sum(r['same_top'] for r in rows)
    # Resizing differs slightly between the two, so scores near the 0.3 threshold can flip
    assert same >= 0.8 * len(rows), same
    assert np.median([abs(r['score_diff']) for r in rows if r['same_top'] and r['ref_top']]) < 0.05
