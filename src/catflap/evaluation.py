from collections import deque
import numpy as np
from enum import Enum
from base_logger import logger
from base_logger import logger

class CatDetection(int, Enum):
    CAT_ALONE = 0,
    CAT_WITH_MOUSE = 1,
    UNDECIDED = 2

class Stats:
    '''Container class for a queue with a rolling average'''
    def __init__(self, name:str, window=5, initial=0.01):
        '''Create an array, with low initial values'''
        self._name = name
        self._queue = deque([initial] * window, maxlen=window)
        self._prev_ma = 0
        self._delta = 0
        self._moving_average = 0
        self._count = 0
        self._min = initial
        self._max = initial

    def __str__(self):
        return f"{self.__class__.__name__} {self._name} ma:{self._moving_average:.2f} min:{self._min:.2f} max:{self._max:.2f} cnt:{self._count} dta:{self._delta:.2f}"
    
    @property
    def max(self) -> float:
        return self._max

    @property
    def min(self) -> float:
        return self._min

    @property
    def moving_average(self) -> float:
        return self._moving_average

    @property
    def count(self) -> int:
        return self._count

    def push(self, value):
        '''Add a new value to the queue and update the average, min, max, count'''
        self._queue.append(value)
        self._count += 1
        ma = sum(self._queue) / len(self._queue)
        self._delta = ma - self._prev_ma
        self._moving_average = ma
        self._prev_ma = ma
        if value > self._max:
            self._max = value
        if value < self._min:
            self._min = value
        logger.info(f"{__class__.__name__} {self._name} pushing value: {value:.2f} == {self}")


# 
class Evaluation():
    '''Container class for evaluation statistics
    The labels used are:
    {"labels": ["Cat-alone", "Cat-with-mouse", "Background", "Cat-body"]}
    Only Cat-alone and Cat-with-mouse are relevant for this evaluation
    This evaluation will swing as more detections are evaluated, it makes
    a recommendation if the flap should be locked or not, but the state 
    machine makes the final determination.
    '''
    def __init__(self) -> None:
        self._stats = {"Cat-alone": Stats("Cat-alone"), "Cat-with-mouse": Stats("Cat-with-mouse")}
        self._flap_locked = False
        self._count = 0
        self._result = CatDetection.UNDECIDED

    def __str__(self) -> str:
        return f"{self._count}:{self._result}"

    def _evaluate(self):
        '''Run the evaluation on the recorded statistics - very simple here
        Generally we err on the side of caution - any hint of a mouse is taken
        as a positive, but the Cat-alone detection requires several strong
        results before we assume it is true.

        TODO - perhaps using size or shape of the bounding rect will help
        choose detections with higher confidence. A good detection will most
        likely be mostly square, or perhaps taller. A small detection rectangle
        indicates the cat is far away; one that is too wide shows a false result
        As an example - this image test_data/Cat-alone-63-20230805-042718.482_catcam.jpg
        clearly has a mouse, but was falsely classified as Alone because the cat is 
        too far away. In this case about 2m - the bounding rect area would have given a 
        good weighting. In the whole sequence of images the logic below would have 
        kept the flap locked, but the Cat-alone statistics are not enough
        0:Cat-alone    0.62   BoundingRect(origin_x=310, origin_y=161, width=60, height=48)
        '''
        retval = CatDetection.UNDECIDED
        if self._stats["Cat-with-mouse"].count < 4 and self._stats["Cat-alone"].count < 3:
            # wait for a minimum number of results
            pass
        elif self._stats["Cat-with-mouse"].max > 0.45:
            retval = CatDetection.CAT_WITH_MOUSE
        elif self._stats["Cat-alone"].moving_average > 0.5:
            retval = CatDetection.CAT_ALONE

        logger.info(f"{self.__class__.__name__} evaluation result {retval.name}")
        self._result = retval

    def add_record(self, label:str, value:float) -> None:
        '''Add the record to the statistics'''
        if label not in self._stats:
            logger.error(f"Label {label} not used for this evaluation")
            return
        self._count += 1
        self._stats[label].push(value)
        self._evaluate()

    @property
    def result(self) -> CatDetection:
        '''Determines if the evaluation determines that the flap should be
        locked or unlocked. This is just a recommendation. The state machine
        will determine the action to perform.
        '''
        return self._result

def main():
    e = Evaluation()
    e.stats["Cat-alone"].push(0.55)
    e.stats["Cat-alone"].push(0.60)
    e.stats["Cat-alone"].push(0.65)
    pass

if __name__ == "__main__":
    main()
