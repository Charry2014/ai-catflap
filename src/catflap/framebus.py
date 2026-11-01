'''Frame hand-off between the camera thread and the consumers.

The camera thread publishes timestamped, read-only Frames. Consumers never
block the camera: they ask for the newest frame they have not seen yet and,
if they are slower than the camera, simply skip the ones in between.
'''
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Frame:
    '''One captured image. The image is shared by reference between all
    consumers and is marked read-only, so draw overlays on a copy.'''
    id: int          # Increases by one per captured frame, gaps mean skipped frames
    ts: float        # time.monotonic() at capture, for latency and ordering
    wall: float      # time.time() at capture, for file names and display
    image: np.ndarray


class FrameBus:
    '''Thread safe ring buffer of the most recent frames.'''

    def __init__(self, maxlen: int = 45) -> None:
        self._frames = deque(maxlen=maxlen)
        self._cond = threading.Condition()
        self._next_id = 0
        self._closed = False

    def publish(self, image: np.ndarray) -> Frame:
        image.flags.writeable = False
        with self._cond:
            frame = Frame(self._next_id, time.monotonic(), time.time(), image)
            self._next_id += 1
            self._frames.append(frame)
            self._cond.notify_all()
        return frame

    def close(self) -> None:
        '''No more frames will arrive; wakes all waiting consumers.'''
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    @property
    def closed(self) -> bool:
        with self._cond:
            return self._closed

    @property
    def published(self) -> int:
        with self._cond:
            return self._next_id

    def latest(self) -> Optional[Frame]:
        with self._cond:
            return self._frames[-1] if self._frames else None

    def wait_newer(self, last_id: int, timeout: Optional[float] = None) -> Optional[Frame]:
        '''Block until a frame with an id greater than last_id exists and
        return the newest one. Returns None on timeout, or when the bus is
        closed and there is nothing newer left. Use last_id=-1 for the first call.'''
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while True:
                if self._frames and self._frames[-1].id > last_id:
                    return self._frames[-1]
                if self._closed:
                    return None
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                self._cond.wait(remaining)

    def since(self, ts: float) -> list:
        '''Frames captured at or after monotonic time ts, oldest first.
        This is what pre-roll recording will use.'''
        with self._cond:
            return [f for f in self._frames if f.ts >= ts]

    def after(self, frame_id: int) -> list:
        '''Frames with an id greater than frame_id, oldest first. If the ring
        has already dropped some of them, those are simply missing.'''
        with self._cond:
            return [f for f in self._frames if f.id > frame_id]
