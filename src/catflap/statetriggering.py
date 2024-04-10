import cv2 as cv
import numpy as np

from tflite_detect import TFLiteDetect
from evaluation import Evaluation
from statetypes import TState, GlobalData, Event, States, CatDetection
from base_logger import logger



class TriggeringState(TState):

    '''Movement was detected, now we are looking for a cat'''
    def __init__(self, *args, **kwargs) -> None:
        super(TriggeringState, self).__init__(*args, **kwargs)

    def on_enter_state(self, event:Event, data:GlobalData) -> None:
        '''When we return to idle we unlock the cat flap, so cats can exit from the inside, and
            reset the evaluation class ready for the next event sequence'''
        # logger.info(f"Entering {self.__class__.__name__} state")
        # logger.info(f"PUML idleState --> flapControl: cat-flap-unlock")
        data.tflite = TFLiteDetect(data.args.model, data.args.enable_edgetpu, data.args.num_threads)
        data.evaluation = Evaluation(data._json_labels, data._json_trigger, CatDetection)

    def run(self, event:Event, data:GlobalData) -> States:
        '''There was movement in front of the camera. Now we look for any cat
            detection - even a low certainty - if we get any cat event we move
            on to locking the cat flap, otherwise this is a false trigger and 
            back to idle we go.
        '''
        retval = States.TRIGGERING

        for d in data.tflite.detect(event.payload):
            eval = data.evaluation.add_record(d.label, d.score)
            logger.debug(f'{self.__class__.__name__} detection {d.label} {d.score} results {eval.name}')

        if len(data.tflite) == 0:
            # This image was not recognised, so treat it as a low certainty background
            # This is done here as we want to return to idle if no cat is seen. We don't
            # want to drop too many images and potentially get stuck here too long
            # Yeah, this is a bit of a hack, but it works.
            eval = data.evaluation.add_record('Background', 0.5)
        else:
            # Record or show the detection results
            if data.headless == False:
                new_image = event.payload.copy()
                cv.imshow(data.window_name, data.tflite.create_overlays(new_image))
                cv.waitKey(30)
            # TODO - Record an image with the overlays
            # if(data.args.record_overlays == True):
            #     frame = create_overlays(frame, detections)
            # outfile = make_outfile_name(data.args.record_path, detections[0].label, detections[0].score)
            # cv.imwrite(outfile, frame)

        # Decide next state once after all evaluations are done
        if eval == CatDetection.BACKGROUND:
            data.record_image(event.payload, "toidle")
            retval = States.IDLE
        elif eval != CatDetection.UNDECIDED:
            data.record_image(event.payload, "locking")
            retval = States.MOVEMENT_LOCKED

        return retval

    def on_exit_state(self, event:Event, data:GlobalData) -> None:
        '''Leaving idle state'''
        '''Start the timeout timer on the transition out of idleState only'''
        # self.timeout_timer = StateTimer(self._timeout_handle)
        # self.timeout_timer.timer_start()
        pass

