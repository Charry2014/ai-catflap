'''Video clips with pre-roll, written from a separate thread.'''
import threading
import time
from datetime import datetime
from os import makedirs, path

import cv2 as cv

from annotations import AnnotationLog, render
from base_logger import logger
from framebus import FrameBus


class ClipRecorder:
    '''Records the frames around an event, including the seconds before it.

    The state machine only calls trigger(). The recorder thread reads frames
    straight from the FrameBus ring buffer, so it gets every captured frame no
    matter how slow detection is, and starts the clip `preroll` seconds before
    the trigger. Each trigger pushes the end of the clip to `postroll` seconds
    after it. When annotations are given, each frame is drawn with the detection
    result closest to its capture time. Results arrive late, so frames are
    held back `overlay_delay` seconds before being written.
    '''
    FOURCC = 'mp4v'

    def __init__(self, bus: FrameBus, record_path: str, fps: float = 15.0,
                 preroll: float = 3.0, postroll: float = 5.0, max_length: float = 120.0,
                 annotations: AnnotationLog = None, overlay_delay: float = 0.6,
                 clock=time.monotonic) -> None:
        try:
            makedirs(record_path, exist_ok=True)
        except OSError:
            raise ValueError(f"Cannot record video to {record_path}")
        self._bus = bus
        self._path = record_path
        self._fps = fps
        self._preroll = preroll
        self._postroll = postroll
        self._max_length = max_length
        self._annotations = annotations
        self._delay = overlay_delay if annotations is not None else 0.0
        self._clock = clock

        self._cond = threading.Condition()
        self._active = False
        self._start_ts = 0.0
        self._deadline = 0.0
        self._stop = False
        self.clips = []   # File names written, newest last
        self._thread = threading.Thread(target=self._run, name="clip-recorder", daemon=True)
        self._thread.start()

    @property
    def recording(self) -> bool:
        with self._cond:
            return self._active

    def trigger(self, ts: float) -> None:
        '''Something is happening at (monotonic) time ts: start a clip, or extend the running one.'''
        with self._cond:
            if not self._active:
                self._active = True
                self._start_ts = ts - self._preroll
                self._deadline = ts + self._postroll
                self._cond.notify_all()
            else:
                self._deadline = max(self._deadline, ts + self._postroll)

    def close(self, timeout: float = 15.0) -> None:
        '''Finish the running clip (with what has been captured) and stop.'''
        with self._cond:
            self._stop = True
            self._cond.notify_all()
        self._thread.join(timeout)

    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._active and not self._stop:
                    self._cond.wait(0.5)
                if not self._active:
                    return
                start_ts = self._start_ts
            try:
                self._record_clip(start_ts)
            except Exception:
                logger.exception("Clip recording failed")
                with self._cond:
                    self._active = False

    def _record_clip(self, start_ts: float) -> None:
        writer = None
        cursor = None
        outfile = None
        written = 0
        try:
            while True:
                with self._cond:
                    deadline, stopping = self._deadline, self._stop
                now = self._clock()
                if stopping:
                    deadline, delay = min(deadline, now), 0.0
                else:
                    delay = self._delay
                frames = self._bus.since(start_ts) if cursor is None else self._bus.after(cursor)
                passed_end = False
                for f in frames:
                    if f.ts > now - delay:
                        break               # Not old enough yet, annotations may still arrive
                    if f.ts > deadline or written >= self._max_length * self._fps:
                        passed_end = True
                        break
                    if writer is None:
                        h, w = f.image.shape[:2]
                        stamp = datetime.fromtimestamp(f.wall).strftime("%Y%m%d-%H%M%S")
                        outfile = path.join(self._path, f"{stamp}_catcam.mp4")
                        writer = cv.VideoWriter(outfile, cv.VideoWriter_fourcc(*self.FOURCC), self._fps, (w, h))
                        if not writer.isOpened():
                            logger.error(f"Cannot open video writer for {outfile}")
                            writer = None
                            return
                        logger.info(f"Recording started to {outfile}")
                    image = f.image
                    if self._annotations is not None:
                        image = render(image, self._annotations.nearest(f.ts))
                    writer.write(image)
                    written += 1
                    cursor = f.id
                # Finished once everything up to the deadline is written
                caught_up = now - delay > deadline
                if passed_end or (caught_up and not frames) or stopping or (self._bus.closed and not frames):
                    with self._cond:
                        if self._deadline > deadline and not stopping and not passed_end:
                            continue        # extended while we were finishing
                        self._active = False
                    return
                time.sleep(0.03)
        finally:
            if writer is not None:
                writer.release()
                self.clips.append(outfile)
                logger.info(f"Recording done: {outfile} ({written} frames)")
