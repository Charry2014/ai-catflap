'''What the system decided about which frame, so recordings can show it.

Detection runs a little behind capture, and not on every frame. The log keys
results by capture time, so whoever writes a frame later (a video clip, a
still) can look up the result that belongs to that moment.'''
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional

from detections import draw_detections


@dataclass(frozen=True)
class Annotation:
    ts: float          # Capture time (monotonic) of the frame that was analysed
    frame_id: int
    detections: tuple
    label: str         # e.g. the state the system was in


class AnnotationLog:
    def __init__(self, maxlen: int = 600) -> None:
        self._items = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def add(self, ts: float, frame_id: int, detections, label: str = '') -> None:
        with self._lock:
            self._items.append(Annotation(ts, frame_id, tuple(detections), label))

    def nearest(self, ts: float, max_gap: float = 0.4) -> Optional[Annotation]:
        '''The annotation closest in time to ts, if within max_gap seconds'''
        with self._lock:
            best = min(self._items, key=lambda a: abs(a.ts - ts), default=None)
        if best is not None and abs(best.ts - ts) <= max_gap:
            return best
        return None


def render(image, annotation: Optional[Annotation]):
    '''A copy of image with the annotation drawn on it, or image itself if there is none'''
    if annotation is None:
        return image
    import cv2 as cv
    out = draw_detections(image.copy(), annotation.detections)
    if annotation.label:
        cv.putText(out, annotation.label, (10, out.shape[0] - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return out
