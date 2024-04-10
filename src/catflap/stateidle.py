import cv2 as cv
import numpy as np
import time

from statetypes import TState, GlobalData, Event, States
from base_logger import logger


class IdleState(TState):

    '''Idle state is slow monitoring of the camera, looking for movement'''
    def __init__(self, *args, **kwargs) -> None:
        super(IdleState, self).__init__(*args, **kwargs)

    def on_enter_state(self, event:Event, data:GlobalData) -> None:
        '''When we return to idle we unlock the cat flap, so cats can exit from the inside'''
        data.cat_flap_control.unlock()
        data.timeout_timer.cancel()

    def run(self, event:Event, data:GlobalData) -> States:
        '''
        Detect movement
        No movement? Sleep 1 second. Return States.IDLE
        evaluation.add_record(event.label, event.score)
        '''
        retval = States.IDLE
        frame1, frame2 = data.get_images(2) # Get newest 2 images from queue

        cp1 = frame1[data.trigger_br:data.trigger_brh, data.trigger_bc:data.trigger_bcw].copy()
        cp2 = frame2[data.trigger_br:data.trigger_brh, data.trigger_bc:data.trigger_bcw].copy()

        diff = cv.absdiff(cp1, cp2)
        diff_gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(diff_gray, (5, 5), 0)
        _, thresh = cv.threshold(blur, 20, 255, cv.THRESH_BINARY)
        dilated = cv.dilate(thresh, None, iterations=3)
        contours, _ = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        if data.headless == False:
            new_image = frame2.copy()
            for contour in contours:
                if cv.contourArea(contour) > 200:  # Filter small contours
                    x, y, w, h = cv.boundingRect(contour)
                    colour = (0, 255, 0)
                    if w * h > 2000:
                        colour = (0, 0, 255)
                    cv.rectangle(new_image, (x+data.trigger_bc, y+data.trigger_br), (x+data.trigger_bc+w, y+data.trigger_br+h), colour, 2)

            cv.imshow(data.window_name, new_image)
            cv.waitKey(30)

        # Check if enough movement is found - is there a big enough rectangle
        total_area = 0
        for contour in contours:
            x, y, w, h = cv.boundingRect(contour)
            if w * h > 2000:
                retval = States.TRIGGERING
                data.record_image(frame2, "movement")
                data.timeout_timer.start()
                break
        else:
            time.sleep(1)
        
        return retval

    def on_exit_state(self, event:Event, data:GlobalData) -> None:
        '''Leaving idle state'''
        pass


