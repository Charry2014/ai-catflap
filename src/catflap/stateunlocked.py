import cv2 as cv
import numpy as np

from statetypes import TState, GlobalData, Event, States, CatDetection
from base_logger import logger


class UnlockedState(TState):
    def __init__(self, *args, **kwargs) -> None:
        super(UnlockedState, self).__init__(*args, **kwargs)

    def on_enter_state(self, event:Event, data:GlobalData) -> None:
        #logger.info(f"Entering {self.__class__.__name__} state")
        # control.cat_flap_unlock()
        #logger.info(f"PUML unlockedState --> flapControl: cat-flap-unlock")
        pass

    def run(self, event:Event, data:GlobalData) -> States:
        '''The state machine will remain unlocked until a new movement is
        detected or a timeout takes it back to idle'''
        retval = States.UNLOCKED

        for d in data.tflite.detect(event.payload):
            eval = data.evaluation.add_record(d.label, d.score)
            logger.debug(f'{self.__class__.__name__} evaluated {d.label} {d.score} results {eval.name}')

            if eval == CatDetection.CAT_ALONE:
                retval = States.UNLOCKED
                break

        # Record or show the detection results
        if data.headless == False:
            new_image = np.zeros_like(event.payload)
            cv.imshow(UnlockedState.window_name, data.tflite.create_overlays(new_image))
            cv.waitKey(30)

        return retval


