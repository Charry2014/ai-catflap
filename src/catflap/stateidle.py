import cv2 as cv

from statetypes import TState, GlobalData, Event, States
from base_logger import logger


class IdleState(TState):

    '''Idle state is watching the camera for movement. The movement analysis itself
    is done by the analysis thread (motion.py); this state decides what it means.'''
    MIN_AREA = 200         # Smaller contours are not drawn
    TRIGGER_AREA = 2000    # A movement box at least this big triggers

    def __init__(self, *args, **kwargs) -> None:
        super(IdleState, self).__init__(*args, **kwargs)

    def on_enter_state(self, event:Event, data:GlobalData) -> None:
        '''When we return to idle we unlock the cat flap, so cats can exit from the inside'''
        data.cat_flap_control.unlock()
        data.timeout_timer.cancel()

    def run(self, event:Event, data:GlobalData) -> States:
        '''
        Detect movement. event.motion holds the movement boxes found in the trigger area.
        '''
        retval = States.IDLE
        frame = event.payload

        if data.headless == False:
            new_image = frame.copy()
            for box in event.motion:
                if box.area > self.MIN_AREA:  # Filter small contours
                    colour = (0, 255, 0)
                    if box.w * box.h > self.TRIGGER_AREA:
                        colour = (0, 0, 255)
                    cv.rectangle(new_image, (box.x, box.y), (box.x + box.w, box.y + box.h), colour, 2)

            cv.imshow(data.window_name, new_image)
            cv.waitKey(1)

        # Check if enough movement is found - is there a big enough rectangle
        for box in event.motion:
            if box.w * box.h > self.TRIGGER_AREA:
                retval = States.TRIGGERING
                data.record_image(frame, "movement")
                data.timeout_timer.start()
                break

        return retval

    def on_exit_state(self, event:Event, data:GlobalData) -> None:
        '''Leaving idle state'''
        pass
