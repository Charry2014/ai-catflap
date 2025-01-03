import cv2 as cv
import numpy as np

from tflite_detect import TFLiteDetect
from evaluation import Evaluation
from statetypes import TState, GlobalData, Event, States, CatDetection
from base_logger import logger



class MovementLockedState(TState):
    window_name = 'Detections'

    def __init__(self, *args, **kwargs) -> None:
        super(MovementLockedState, self).__init__(*args, **kwargs)

    def on_enter_state(self, event:Event, data:GlobalData) -> None:
        logger.info(f"PUML movementLockedState --> flapControl: cat-flap-lock")
        data.cat_flap_control.lock()
        data.tflite = TFLiteDetect(data.args.model, data.args.enable_edgetpu, data.args.num_threads)
        data.evaluation = Evaluation(data._json_labels, data._json_eval, CatDetection)
        data.timeout_timer.start()


    def run(self, event:Event, data:GlobalData) -> States:
        '''From Movement Locked we can either unlock (no mouse is detected) or go into
        permanent lock when a mouse is detected. We can also stay here until the
        timeout if there is no solid determination of either of these.'''
        retval = States.MOVEMENT_LOCKED
    
        for d in data.tflite.detect(event.payload):
            eval = data.evaluation.add_record(d.label, d.score)
            logger.debug(f'{self.__class__.__name__} evaluated {d.label} {d.score:.2f} results {eval.name}')

            # Decide next state, after each detection result - first result wins
            if eval == CatDetection.CAT_ALONE:
                data.record_image(event.payload, "unlock")
                retval = States.UNLOCKED
                break
            elif eval == CatDetection.CAT_WITH_MOUSE:
                data.record_image(event.payload, "mouselock")
                retval = States.MOUSE_LOCKED
                break

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

        return retval


