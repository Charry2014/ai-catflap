import cv2 as cv
import numpy as np

from tflite_detect import TFLiteDetect
from evaluation import Evaluation
from statetypes import TState, GlobalData, Event, States, CatDetection
from base_logger import logger



class MouseLockedState(TState):
    window_name = 'Detections'

    def __init__(self, *args, **kwargs) -> None:
        super(MouseLockedState, self).__init__(*args, **kwargs)

    def on_enter_state(self, event:Event, data:GlobalData) -> None:
        # logger.info(f"Entering {self.__class__.__name__} state")
        # control.cat_flap_lock()
        # logger.info(f"PUML mouseLockedState --> flapControl: cat-flap-lock")
        pass


    def run(self, event:Event, data:GlobalData) -> States:
        '''The cat has a mouse so the flap is locked. Here we can choose to keep evaluating and
            perhaps unlock, or just wait for the timeout. It is not sure what makes more sense
            so for now we give the benefit of the doubt and keep evaluating.'''
        retval = States.MOUSE_LOCKED

        for d in data.tflite.detect(event.payload):
            eval = data.evaluation.add_record(d.label, d.score)
            logger.debug(f'{self.__class__.__name__} evaluated {d.label} {d.score} results {eval.name}')

            # Decide next state, after each detection result - first result wins
            if eval == CatDetection.CAT_ALONE:
                retval = States.UNLOCKED
                break

        # Record or show the detection results
        if data.headless == False:
            new_image = np.zeros_like(event.payload)
            cv.imshow(MouseLockedState.window_name, data.tflite.create_overlays(new_image))
            cv.waitKey(30)
        # TODO - Record an image with the overlays
        # if(data.args.record_overlays == True):
        #     frame = create_overlays(frame, detections)
        # outfile = make_outfile_name(data.args.record_path, detections[0].label, detections[0].score)
        # cv.imwrite(outfile, frame)

        return retval
