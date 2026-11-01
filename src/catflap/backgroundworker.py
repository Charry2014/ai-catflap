'''Run slow, non-critical work (disk writes, network) off the main loop.'''
import queue
import threading
from typing import Callable

from base_logger import logger


class BackgroundWorker:
    '''A thread that applies `fn` to every submitted item, in order.

    The queue is bounded so a slow consumer can never grow memory or stall the
    producer. When full, drop_oldest=True discards the oldest queued item
    (right for "latest picture" uses like the website); drop_oldest=False
    discards the new one (right when order matters). Drops are counted and logged.
    '''

    def __init__(self, fn: Callable, name: str, maxsize: int = 1, drop_oldest: bool = True) -> None:
        self._fn = fn
        self._drop_oldest = drop_oldest
        self._queue = queue.Queue(maxsize=maxsize)
        self.dropped = 0
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def submit(self, item) -> bool:
        '''Never blocks. Returns False if something was dropped.'''
        try:
            self._queue.put_nowait(item)
            return True
        except queue.Full:
            pass
        self.dropped += 1
        if self._drop_oldest:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                pass
        if self.dropped == 1 or self.dropped % 100 == 0:
            logger.warning(f"{self._thread.name} worker is behind, dropped {self.dropped} so far")
        return False

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                return
            try:
                self._fn(item)
            except Exception:
                logger.exception(f"{self._thread.name} worker failed")
            finally:
                self._queue.task_done()

    def close(self, timeout: float = 5.0) -> None:
        '''Finish queued items, then stop.'''
        self._queue.put(None)
        self._thread.join(timeout)
