from threading import Timer
from base_logger import logger


class StateTimer():

    def __init__(self, callback, interval=5) -> None:
        '''A timer, inverval in minutes!!
        Creates the timer but does not start it.
        '''
        self._interval = interval * 60
        self._callback = callback
        self._timer = Timer(self._interval, self._internal_callback)
        self._running = False

    def _internal_callback(self) -> None:
        logger.info(f"{self.__class__.__name__} timer timeout.")
        self._running = False
        self._callback(args=None)

    def timer_start(self) -> None:
        '''Starts the timer if it is not running'''
        if self._running == False:
            logger.info(f"{self.__class__.__name__} starting timer.")
            self._timer.start()
            self._running = True

    def timer_restart(self) -> None:
        '''Start the timeout again from the beginning'''
        logger.info(f"{self.__class__.__name__} timer restarted.")
        if self._running == True:
            self._timer.cancel()
        self._timer.start()
    
    def timer_cancel(self):
        logger.info(f"{self.__class__.__name__} timer canceled.")
        self._running = False
        self._timer.cancel()


