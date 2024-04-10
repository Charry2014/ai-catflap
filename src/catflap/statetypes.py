from abc import ABC, abstractmethod
from statemachine import State
from enum import Enum
from collections import deque
from array import array
import json
import sys
from base_logger import logger

from catflapcontrol import CatFlapControl
import image_recorder


class States(int, Enum):
    IDLE = 1
    TRIGGERING = 2
    UNLOCKED = 3
    MOVEMENT_LOCKED = 4
    MOUSE_LOCKED = 5


class CatDetection(int, Enum):
    CAT_ALONE = 0
    CAT_WITH_MOUSE = 1
    BACKGROUND = 2
    CAT_BODY = 3
    UNDECIDED = 4


class Event():
    '''Basic event wrapper class so state machine stays more generic'''
    def __init__(self, event):
        self._event = event

    def __str__(self) -> str:
        return f"{str(type(self._event))}{self._event.shape}"

    @property
    def payload(self):
        return self._event
        
    

class GlobalData():
    '''This has become a bit of a smorsgasbord of everything - not pretty but functional'''
    def __init__(self, args, event=None) -> None:
        self.args = args
        assert(hasattr(args, 'trigger'))
        assert(hasattr(args, 'label_json'))
        assert(hasattr(args, 'trigger_json'))
        assert(hasattr(args, 'eval_json'))

        # Command line arguments processing
        # Define the trigger area. Images are cropped to this to detect movement
        self._trigger_bc, self._trigger_br, bw, bh = [int(x) for x in args.trigger.split(',')]
        self._trigger_bcw = self._trigger_bc + bw
        self._trigger_brh = self._trigger_br + bh

        # Open config files
        # First, the labels used in the ML models
        try:
            with open (args.label_json) as j:
                self._json_labels = json.load(j)['labels']
        except Exception as e:
            logger.error(f"Error opening JSON {args.label_json} - {e.__class__} - {e}")
            sys.exit(1)
        # Then, the JSON that defines the triggering evaluation
        try:
            with open (args.trigger_json) as j:
                self._json_trigger = json.load(j)
        except Exception as e:
            logger.error(f"Error opening JSON {args.trigger_json} - {e.__class__} - {e}")
            sys.exit(1)
        # Finally, the JSON that defines the cat alone / cat with mouse evaluation
        try:
            with open (args.eval_json) as j:
                self._json_eval = json.load(j)
        except Exception as e:
            logger.error(f"Error opening JSON {args.eval_json} - {e.__class__} - {e}")
            sys.exit(1)

        # Create the event queue
        self._event_queue = deque(maxlen=2)
        if event != None:
            self._event_queue.append(event)

        # Settings
        if hasattr(args, "headless") == True:
            self._headless = args.headless
        else:
            self._headless = False

        # 
        self.cat_flap_control = CatFlapControl()
        self.evaluation = None
        self.timeout_timer = None
        self.tflite = None

        # Create the image recorder
        self._image_recorder = image_recorder.image_recorder(self.args.record_path)


    '''Settings'''
    @property
    def headless(self) -> bool:
        '''headless - ie. show no windows'''
        return self._headless

    '''Trigger area properties'''
    @property
    def trigger_bc(self) -> int:
        return self._trigger_bc
    @property
    def trigger_br(self) -> int:
        return self._trigger_br
    @property
    def trigger_bcw(self) -> int:
        return self._trigger_bcw
    @property
    def trigger_brh(self) -> int:
        return self._trigger_brh

    '''Event queue stuff'''
    def add_event(self, event:Event):
        self._event_queue.append(event)

    def get_images(self, count=1) -> list:
        return [e.payload for e in list(self._event_queue)[(-1 * count):]]

    def record_image(self, image: array, label:str):
        self._image_recorder(image, label)

    @property
    def window_name(self) -> str:
        return "Cat flap image"


class TState(State, ABC):
    @abstractmethod
    def run(self, event:Event) -> States:
        pass

   
    def __cmp__(self, other):
        return self.name == other.name
    
    # Necessary when __cmp__ or __eq__ is defined
    # in order to make this class usable as a
    # dictionary key:
    def __hash__(self):
        return hash(self.name)




