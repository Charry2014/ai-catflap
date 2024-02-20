from base_logger import logger
import platform
CPU = platform.processor()
from base_logger import logger




# A messy attempt at platform independence
if CPU == "i386":
    # This is the non-ARM case, a desktop system
    pass
else:
    import RPi.GPIO as GPIO

GPIO_PIN = 21


class CatFlapControl():
    '''Control the relay that enables or disables the cat flap.
    When GPIO is LOW then the NC connections on the relay are closed.
    This means that normally the GPIO will be LOW and when it is necessary
    to lock the cat flap then the GPIO will be driven HIGH'''
    def __init__(self) -> None:
        self._locked = False

        if CPU == "i386":
            pass
        else:
            # Set up GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(GPIO_PIN, GPIO.OUT)
            GPIO.output(GPIO_PIN, GPIO.LOW)

    @property
    def locked(self) -> bool:
        return self._locked

    @property
    def gpio_state(self) -> int:
        if CPU == "i386":
            if self._locked == True: return 1 
            else: return 0
        state = GPIO.input(GPIO_PIN)
        return state

    def lock(self) -> None:
        # if self._locked == True and self.gpio_state == 1:
        #     return
        self._locked = True
        logger.debug(f"{self.__class__.__name__} locking cat flap")
        if CPU == "i386":
            pass
        else:
            GPIO.output(GPIO_PIN, GPIO.HIGH)

    def unlock(self) -> None:
        # if self._locked == False and self.gpio_state == 0:
        #     return
        self._locked = False
        logger.debug(f"{self.__class__.__name__} unlocking cat flap")
        if CPU == "i386":
            pass
        else:
            GPIO.output(GPIO_PIN, GPIO.LOW)

    def exit(self) -> None:
        logger.debug(f"{self.__class__.__name__} exiting")
        self.cat_flap_unlock()
        if CPU == "i386":
            pass
        else:
            # Clean up GPIO on program exit
            GPIO.cleanup()
