import cv2 as cv

from statetypes import TState, GlobalData, Event, States, CatDetection
from base_logger import logger
from detections import draw_detections



class MouseLockedState(TState):

    def __init__(self, *args, **kwargs) -> None:
        super(MouseLockedState, self).__init__(*args, **kwargs)

    def on_enter_state(self, event:Event, data:GlobalData) -> None:
        logger.info("PUML mouseLockedState --> flapControl: cat-flap-lock")
        data.cat_flap_control.lock()
        data.timeout_timer.start()

    def run(self, event:Event, data:GlobalData) -> States:
        '''The cat has a mouse so the flap is locked. Here we can choose to keep evaluating and
            perhaps unlock, or just wait for the timeout. It is not sure what makes more sense
            so for now we give the benefit of the doubt and keep evaluating.'''
        retval = States.MOUSE_LOCKED

        for d in event.detections:
            eval = data.evaluation.add_record(d.label, d.score)
            logger.debug(f'{self.__class__.__name__} evaluated {d.label} {d.score} results {eval.name}')

            # Decide next state, after each detection result - first result wins
            if eval == CatDetection.CAT_ALONE:
                data.record_image(event.payload, "unlock", event.detections)
                retval = States.UNLOCKED
                break

        # Record or show the detection results
        if data.headless == False:
            new_image = event.payload.copy()
            cv.imshow(data.window_name, draw_detections(new_image, event.detections))
            cv.waitKey(1)

        return retval
