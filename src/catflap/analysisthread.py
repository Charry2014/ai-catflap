'''Turns frames into events for the state machine.

While the flap is idle this looks for movement (cheap). Once the state machine
arms it, every frame goes through the ML model (slow). Either way it works on
the newest frame only, so a slow model skips frames instead of falling behind.
Results are posted to the state machine's event queue, which the main thread
reads, so all state, GPIO and window handling stays on one thread.'''
import queue
import threading
import time

from base_logger import logger
from framebus import FrameBus
from statetypes import Event, EventKind


class AnalysisThread(threading.Thread):

    def __init__(self, bus: FrameBus, events: queue.Queue, detector, motion) -> None:
        super().__init__(name="analysis", daemon=True)
        self._bus = bus
        self._events = events
        self._detector = detector
        self._motion = motion
        self._armed = False
        self._halt = threading.Event()
        self.failed = False
        # Statistics, read from other threads for logging only
        self.processed = 0
        self.skipped = 0
        self.dropped = 0

    def set_armed(self, armed: bool) -> None:
        '''True: run the ML model on every frame. False: watch for movement.'''
        self._armed = armed

    def stop(self) -> None:
        self._halt.set()

    def _post(self, event: Event, must_arrive: bool = False) -> None:
        try:
            if must_arrive:
                self._events.put(event, timeout=5)
            else:
                self._events.put_nowait(event)
        except queue.Full:
            self.dropped += 1

    def run(self) -> None:
        last_id = -1
        mode = False
        try:
            while not self._halt.is_set():
                frame = self._bus.wait_newer(last_id, timeout=0.5)
                if frame is None:
                    if self._bus.closed:
                        break
                    continue
                self.skipped += frame.id - last_id - 1
                last_id = frame.id
                armed = self._armed
                if armed != mode:
                    self._motion.reset()  # Frames from before the change are stale
                    mode = armed
                if armed:
                    detections = list(self._detector.detect(frame.image))
                    event = Event(frame.image, frame.ts, frame.id, detections=detections, armed=True)
                else:
                    event = Event(frame.image, frame.ts, frame.id,
                                  motion=self._motion.process(frame.ts, frame.image), armed=False)
                self.processed += 1
                # Droppable: a missing preview/motion frame is harmless, the next one follows
                self._post(event)
        except Exception:
            logger.exception("Analysis thread failed")
            self.failed = True
        finally:
            self._post(Event(None, kind=EventKind.END), must_arrive=True)
