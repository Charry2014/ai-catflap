import os

from base_logger import logger

# GPIO mode, from the CATFLAP_GPIO environment variable:
#   auto (default) - use the Pi's GPIO if it is there, otherwise simulate (desktop, containers without devices)
#   hardware       - the GPIO must work, refuse to start without it
#   sim            - never touch hardware
GPIO_MODE = os.environ.get("CATFLAP_GPIO", "auto").lower()

GPIO = None
if GPIO_MODE != "sim":
    try:
        import RPi.GPIO as GPIO
    except (ImportError, RuntimeError) as e:    # RuntimeError: "can only be run on a Raspberry Pi"
        if GPIO_MODE == "hardware":
            raise
        logger.info(f"No GPIO available ({e}), cat flap relay is simulated")

GPIO_PIN = 21


class CatFlapControl():
    '''Control the relay that enables or disables the cat flap.
    When GPIO is LOW then the NC connections on the relay are closed.
    This means that normally the GPIO will be LOW and when it is necessary
    to lock the cat flap then the GPIO will be driven HIGH'''
    def __init__(self) -> None:
        self._locked = False
        self._gpio = GPIO

        if self._gpio is not None:
            try:
                self._gpio.setmode(self._gpio.BCM)
                self._gpio.setup(GPIO_PIN, self._gpio.OUT)
                self._gpio.output(GPIO_PIN, self._gpio.LOW)
            except (RuntimeError, OSError) as e:    # e.g. no access to /dev/gpiomem
                if GPIO_MODE == "hardware":
                    raise
                logger.error(f"GPIO setup failed ({e}), cat flap relay is SIMULATED - the flap will not lock")
                self._gpio = None

    @property
    def locked(self) -> bool:
        return self._locked

    @property
    def gpio_state(self) -> int:
        if self._gpio is None:
            return 1 if self._locked else 0
        return self._gpio.input(GPIO_PIN)

    def lock(self) -> None:
        self._locked = True
        logger.debug(f"{self.__class__.__name__} locking cat flap")
        if self._gpio is not None:
            self._gpio.output(GPIO_PIN, self._gpio.HIGH)

    def unlock(self) -> None:
        self._locked = False
        logger.debug(f"{self.__class__.__name__} unlocking cat flap")
        if self._gpio is not None:
            self._gpio.output(GPIO_PIN, self._gpio.LOW)

    def exit(self) -> None:
        logger.debug(f"{self.__class__.__name__} exiting")
        self.unlock()
        if self._gpio is not None:
            # Clean up GPIO on program exit
            self._gpio.cleanup()
