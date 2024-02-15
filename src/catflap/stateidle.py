import cv2 as cv
import numpy as np
import time

from statetypes import TState, GlobalData, Event, States
from base_logger import logger


class IdleState(TState):
    window_name = 'Movement Contours'

    '''Idle state is slow monitoring of the camera, looking for movement'''
    def __init__(self, *args, **kwargs) -> None:
        super(IdleState, self).__init__(*args, **kwargs)

    def run(self, event:Event, data:GlobalData) -> States:
        '''
        Detect movement
        No movement? Sleep 1 second. Return States.IDLE
        evaluation.add_record(event.label, event.score)
        '''
        retval = States.IDLE
        frame1, frame2 = data.get_images(2) # Get newest 2 images from queue

        # Motion detection magic, trim the incoming images
        cp1 = frame1[data.trigger_br:data.trigger_brh, data.trigger_bc:data.trigger_bcw]
        cp2 = frame2[data.trigger_br:data.trigger_brh, data.trigger_bc:data.trigger_bcw]

        diff = cv.absdiff(cp1, cp2)
        diff_gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(diff_gray, (5, 5), 0)
        _, thresh = cv.threshold(blur, 20, 255, cv.THRESH_BINARY)
        dilated = cv.dilate(thresh, None, iterations=3)
        contours, _ = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        if data.headless == False:
            new_image = frame2.copy()
            cv.drawContours(new_image, contours, -1, (0, 255, 0), 2)
            cv.imshow(IdleState.window_name, new_image)
            cv.waitKey(30)

        # Check if enough movement is found
        total_area = 0
        for contour in contours:
            area = cv.contourArea(contour)
            total_area += area
        logger.debug(f"{self.__class__.__name__} Movement detection area {total_area}")
        if total_area > 1200:
            retval = States.TRIGGERING
        else:
            time.sleep(1)
        
        return retval

    def on_enter_state(self, event:Event, data:GlobalData) -> None:
        '''When we return to idle we unlock the cat flap, so cats can exit from the inside, and
        reset the evaluation class ready for the next event sequence'''
        # data.cat_flap_control.unlock()
        # logger.info(f"PUML idleState --> flapControl: cat-flap-unlock")
        # data.evaluation = None
        pass
        
    def on_exit_state(self, event:Event, data:GlobalData) -> None:
        '''Leaving idle state'''
        if data.headless == False:
            cv.destroyWindow(IdleState.window_name)


