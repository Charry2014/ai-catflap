from collections import deque
import numpy as np
from enum import Enum
from base_logger import logger


class Stats:
    '''Container class for a queue with a rolling average
        Don't create these yourself, use the StatsFactory'''
    def __init__(self, name:str, window=5, initial=0.01):
        '''Create an array, with low initial values'''
        self._name = name
        self._queue = deque(maxlen=window)
        self._prev_ma = 0
        self._delta = 0
        self._moving_average = 0
        self._count = 0
        self._min = initial
        self._max = initial
        self._threshold = 1

        self._status = None
        self._triggered_result = None

    def __str__(self):
        return f"{self.__class__.__name__} for {self._name} m-avg:{self._moving_average:.2f} min:{self._min:.2f} max:{self._max:.2f} count:{self._count} delta:{self._delta:.2f}"
    
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

    @property
    def status(self):
        return self._status

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
        # evaluate the result....
        if self._moving_average >= self.average_threshold and self._count >= self.min_result_count:
            self._status = self._triggered_result

        logger.debug(f"{__class__.__name__} pushed value: {self._name},{value:.2f} == {self} result {self._status.name}")
        return self._status


class StatsFactory():
    '''Helper class to create Stats objects correctly. The result enum MUST be ordered
        the same way as the labels, and have an UNDECIDED element last'''
    def __init__(self, config:str, result:Enum) -> None:
        self._index = 0
        self._config = config
        self._result_enum = result
        self._default_result = result(len(self._result_enum)-1)
        assert('thresholds' in config)
        self._thresholds = dict([ (l['label'], l['values']) for l in config['thresholds'] ])

    def create(self, label:str) -> Stats:
        '''Creates a Stats object, correctly configured. Only labels that
            have a configuration object are used - the rest are ignored.
            All the attributes set in the evaluation json are turned into
            attributes in the Stats object, as well as the result to return
            when triggered.'''
        stat = None
        if label in self._thresholds:
            stat = Stats(label)
            [setattr(stat, l, self._thresholds[label][l]) for l in self._thresholds[label] ]
            stat._triggered_result = self._result_enum(self._index)
            stat._default_result = self._default_result 
            stat._status = self._default_result

        self._index += 1
        return stat


class Evaluation():
    '''Container class for evaluation statistics
    The labels used are defined in JSON, example:
    {"labels": ["Cat-alone", "Cat-with-mouse", "Background", "Cat-body"]}
    This is the same json used by the TF Lite model
    Evaluation details can be set with the config json - configure the relevant
    labels and threshold levels. See examples provided.
    The result_enum values must correspond to the labels used, and end in UNDECIDED,
    ie. the default value when no determination has been made. It must be an int.
    '''
    def __init__(self, labels:list, config, result_enum: Enum, min=3, delta=0.30) -> None:
        '''_stats is a dictionary { label, Eval() } to evaluate each
        incoming result. 
        The label Undecided is an extra value.
        '''
        labels.append('Undecided')
        factory = StatsFactory(config, result_enum)
        self._stats = dict([(l, factory.create(l)) for l in labels])
        self._count = 0
        self._result = result_enum(len(result_enum)-1)
        self._last_result = self._default_result = result_enum(len(result_enum)-1)

    def __str__(self) -> str:
        return f"{self._count}:{self._result}"

    # Stats for Cat-with-mouse m-avg:0.59 min:0.01 max:0.73 count:43 delta:-0.03 result CAT_WITH_MOUSE
    # Stats for Cat-alone m-avg:0.47 min:0.01 max:0.57 count:26 delta:0.02 result UNDECIDED
    # Stats for Cat-alone m-avg:0.53 min:0.01 max:0.58 count:28 delta:0.04 result CAT_ALONE
    # The score is calculated (m-avg * count * max)
    def __evaluate(self) -> int:
        '''Evaluates the stats results. For any result that is not the default (ie. the basic
            evaluation criteria have been met) they are scored by their overall weight and the
            best one is returned as the result.
            The return value is from the CatDetection Enum'''
        retval = self._default_result
        result_list = []
        for k,v in self._stats.items():
            if v == None or v._status == v._default_result:
                continue
            v._score = v.count * v.max * v._moving_average
            result_list.append(v)
        result_list.sort(key = lambda x: x._score, reverse=True)
        if len(result_list) > 0:
            retval = result_list[0].status
        return retval
 

    def add_record(self, label:str, value:float) -> int:
        '''Add the record to the statistic, if the label is relevant for this evaluation
            The return value is from the CatDetection Enum'''
        retval = self._last_result
        if self._stats[label] != None:
            self._stats[label].push(value)
            self._last_result = retval = self.__evaluate()

        return retval

    @property
    def result(self) -> int:
        '''The result of the determination'''
        return self._result


def main():
    import json
    with open ("./labels.json") as j:
        labels = json.load(j)['labels']
        with open ("./eval_config.json") as e:
            config = json.load(e)

    e = Evaluation(labels, config, CatDetection)
    result = e.add_record("Background", 0.60)
    result = e.add_record("Background", 0.60)
    result = e.add_record("Cat-alone", 0.55)
    result = e.add_record("Background", 0.60)
    result = e.add_record("Cat-alone", 0.60)
    result = e.add_record("Background", 0.60)
    result = e.add_record("Cat-alone", 0.65)
    pass

if __name__ == "__main__":
    main()
