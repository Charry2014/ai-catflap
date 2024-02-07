
from statemachine import StateMachine, State
from enum import Enum
import sys
from base_logger import logger
from detections import TFLDetections, TFLDetection, BoundingRect
from evaluation import Evaluation, CatDetection
from catflapcontrol import CatFlapControl
from statetimer import StateTimer

class States(int, Enum):
    IDLE = 1
    TRIGGERING = 2
    UNLOCKED = 3
    MOVEMENT_LOCKED = 4
    MOUSE_LOCKED = 5


from abc import ABC, abstractmethod


class Event():
    '''Basic event wrapper class so state machine stays more generic'''
    def __init__(self, d):
        self._label = d.label
        pass

    def __str__(self) -> str:
        return self._label

    @property
    def detail(self):
        return self._label
    pass

class GlobalData():
    def __init__(self) -> None:
        self.cat_flap_control = CatFlapControl()
        self.evaluation = None
        self.timeout_timer = None


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


class Idle(TState):
    # name = "idleState"

    '''Idle state is slow monitoring of the camera, looking for movement'''
    def __init__(self, *args, **kwargs) -> None:
        super(Idle, self).__init__(*args, **kwargs)

    def run(self, event:Event, data:GlobalData) -> States:
        '''
        Detect movement
        No movement? Sleep 1 second. Return States.IDLE
        evaluation.add_record(event.label, event.score)
        '''
        # sys.sleep(1)
        # return States.IDLE
        return States.TRIGGERING

    def on_enter_state(self, event:Event, data:GlobalData) -> None:
        '''When we return to idle we unlock the cat flap, so cats can exit from the inside, and
        reset the evaluation class ready for the next event sequence'''
        # data.cat_flap_control.cat_flap_unlock()
        # logger.info(f"PUML idleState --> flapControl: cat-flap-unlock")
        # data.evaluation = None
        pass
        
    def on_exit_state(self, event:Event, data:GlobalData) -> None:
        '''Leaving idle state - create a newly initialised statistics class'''
        logger.debug(f"Exiting {self.__class__.__name__} state")


class Triggering(TState):
    # name = "triggeringState"

    '''Movement was detected, now we are looking for a cat'''
    def __init__(self, *args, **kwargs) -> None:
        super(Triggering, self).__init__(*args, **kwargs)

    def run(self, event:Event, data:GlobalData) -> States:
        '''
        Detect movement
        No movement? Sleep 1 second. Return States.IDLE
        evaluation.add_record(event.label, event.score)
        '''
        return States.MOVEMENT_LOCKED

    def on_enter_state(self, event:Event, data:GlobalData) -> None:
        '''When we return to idle we unlock the cat flap, so cats can exit from the inside, and
        reset the evaluation class ready for the next event sequence'''
        # logger.info(f"Entering {self.__class__.__name__} state")
        # data.cat_flap_control.cat_flap_unlock()
        # logger.info(f"PUML idleState --> flapControl: cat-flap-unlock")
        pass

    def on_exit_state(self, event:Event, data:GlobalData) -> None:
        '''Leaving idle state - create a newly initialised statistics class'''
        # logger.debug(f"Exiting {self.__class__.__name__} state")
        '''Start the timeout timer on the transition out of idleState only'''
        # self.timeout_timer = StateTimer(self._timeout_handle)
        # self.timeout_timer.timer_start()
        pass


class MovementLocked(TState):
    def __init__(self, *args, **kwargs) -> None:
        super(MovementLocked, self).__init__(*args, **kwargs)

    def run(self, event:Event, data:GlobalData) -> States:
        '''From Movement Locked we can either unlock (no mouse is there) or go into
        permanent lock when a mouse is detected.'''
        retval = States.MOVEMENT_LOCKED
        # evaluation.add_record(event.label, event.score)
        # if evaluation.result == CatDetection.CAT_WITH_MOUSE:
        #     retval = States.MOUSE_LOCKED
        # elif evaluation.result == CatDetection.CAT_ALONE:
        #     retval = States.UNLOCKED
        return retval

    def on_enter_state(self, event:Event, data:GlobalData) -> None:
        #logger.info(f"Entering {self.__class__.__name__} state")
        # control.cat_flap_lock()
        logger.info(f"Doing something important here")
        pass


class Unlocked(TState):
    def __init__(self, *args, **kwargs) -> None:
        super(Unlocked, self).__init__(*args, **kwargs)

    def run(self, event:Event, data:GlobalData) -> States:
        '''The state machine will remain unlocked until a new movement is
        detected or a timeout takes it back to idle'''
        retval = States.UNLOCKED
        # evaluation.add_record(event.label, event.score)
        # if evaluation.result == CatDetection.CAT_WITH_MOUSE:
        #     retval = States.MOUSE_LOCKED
        return retval


    def on_enter_state(self, event:Event, data:GlobalData) -> None:
        #logger.info(f"Entering {self.__class__.__name__} state")
        # control.cat_flap_unlock()
        #logger.info(f"PUML unlockedState --> flapControl: cat-flap-unlock")
        pass

class MouseLocked(TState):
    def __init__(self, *args, **kwargs) -> None:
        super(MouseLocked, self).__init__(*args, **kwargs)

    def run(self, event:Event, data:GlobalData) -> States:
        retval = States.MOUSE_LOCKED
        # evaluation.add_record(event.label, event.score)
        # if evaluation.result == CatDetection.CAT_ALONE:
        #     retval = States.MOVEMENT_LOCKED
        return retval

    def on_enter_state(self, event:Event, data:GlobalData) -> None:
        # logger.info(f"Entering {self.__class__.__name__} state")
        # control.cat_flap_lock()
        # logger.info(f"PUML mouseLockedState --> flapControl: cat-flap-lock")
        pass

    def on_exit_state(self, event:Event, data:GlobalData) -> None:
        # logger.info(f"Exiting {self.__class__.__name__} state")
        pass

class CatFlapFSM(StateMachine):

    '''Define the states'''
    idleState = Idle(name=States.IDLE, initial=True)
    triggeringState = Triggering(name=States.TRIGGERING)
    unlockedState = Unlocked(name=States.UNLOCKED)
    movementLockedState = MovementLocked(name=States.MOVEMENT_LOCKED)
    mouseLockedState = MouseLocked(name=States.MOUSE_LOCKED)
    '''State transitions. All same-state transitions are internal, which means they
        don't trigger the _exit and _enter actions if the from and too states are 
        the same.'''
    triggering = idleState.to(triggeringState) | triggeringState.to.itself(internal=True)
    movementLock = triggeringState.to(movementLockedState) | movementLockedState.to.itself(internal=True)
    unlock = movementLockedState.to(unlockedState) | unlockedState.to.itself(internal=True)
    returnIdle = movementLockedState.to(idleState) | mouseLockedState.to(idleState) |  unlockedState.to(idleState) | idleState.to.itself(internal=True)
    mouseLock = movementLockedState.to(mouseLockedState) | unlockedState.to(mouseLockedState) | mouseLockedState.to.itself(internal=True)


    def __init__(self) -> None:
        # Must come before parent init as this calls the Idle.on_enter_state method
        self._data = GlobalData()
        StateMachine.__init__(self, CatFlapFSM.idleState)
        
        logger.info("PUML nullState -> idleState: Startup")

        '''Lookups from States ID into the state variables, and the transitions'''
        self.states = {
            States.IDLE: self.idleState,
            States.TRIGGERING : self.triggeringState,
            States.UNLOCKED: self.unlockedState,
            States.MOVEMENT_LOCKED: self.movementLockedState,
            States.MOUSE_LOCKED: self.mouseLockedState
        }
        self.transitions = {
            States.IDLE : self.returnIdle,
            States.TRIGGERING: self.triggering,
            States.UNLOCKED : self.unlock,
            States.MOVEMENT_LOCKED : self.movementLock,
            States.MOUSE_LOCKED : self.mouseLock
            }


    def event_handle(self, event:Event):
        '''Handle detection events from the cat AI system
            This is the first step in handling a new event from outside'''
        logger.debug(f"Handling incoming detection event {event.detail} in state {self.current_state.id}")
        # if self.evaluation == None:
        #     logger.debug(f"Reset the evaluation statistics to default values")
        #     self.evaluation = Evaluation()

        # Run the event handler for the current state with the incoming event and
        # the global data
        self._data.new_state = self.states[self.current_state.run(event, self._data)]

        # if self.current_state == self.idleState and new_state != self.idleState:
        #     '''Start the timeout timer on the transition out of idleState only'''
        #     # self.timeout_timer = StateTimer(self._timeout_handle)
        #     # self.timeout_timer.timer_start()
        # elif self.current_state != self.idleState and new_state == self.idleState:
        #     self.evaluation = None
        #     # self.timeout_timer.timer_stop()

        # Call the state transition action
        action = self.transitions[self._data.new_state.name]
        action(self)


    def _timeout_handle(self, args):
        '''This is a watchdog type timer to ensure the cat flap is always opened
        when a movement sequence ends.'''
        logger.info(f"PUML {self.current_state.id} -> idleState: Timeout")
        self.evaluation = None
        self.returnIdle()

    def exit(self):
        '''Called when the state machine must exit. Unlock cat flap and exit.'''
        logger.info(f"Handling state machine exit event")
        # self._data.timeout_timer.timer_cancel()

    ''' ------------------ Generic Transition Events ------------------
        These are sorted into order, called on specific transitions only
    '''
    def before_transition(self, event, state):
        '''1. The first step in a state transition
            The state parameter is the current state'''
        # logger.debug(f"{self.__class__.__name__} before_transition: event  '{event}', in state '{state.id}'.")

    def on_exit_state(self, event, state):
        '''2. Exit the current state
            The state parameter is the current state'''
        logger.debug(f"{self.__class__.__name__} on_exit_state: event  '{event}', exiting state '{state.id}'.")
        if hasattr(state, "on_exit_state") == True:
            state.on_exit_state(event, self._data)

    def on_transition(self, event, state):
        '''3. Ready to transition from the old state to the new one. 
            The state parameter here is the state we are leaving'''
        logger.info(f"PUML {self.current_state.id} -> {self._data.new_state.id}: {event}")
        pass

    def on_enter_state(self, event, state):
        '''4. Entering the new state - the state parameter here is now the new state
            This will also enter idleState from __initial__'''
        logger.debug(f"{self.__class__.__name__} on_enter_state: event '{event}', entering state '{state.id}'.")
        if hasattr(state, "on_enter_state") == True:
            state.on_enter_state(event, self._data)

    def after_transition(self, event, state):
        pass
        # logger.debug(f"After '{event}', on the '{state.id}' state.")


    ''' ------------------ Specific Transition Events ------------------
        These are called only for actual transitions and not for
        'to-self' events. For the exit events state will be the current
        state, for the enter events state is the new state.
    '''
    # def on_exit_idleState(self, event, state):
    #     state.on_exit_state(event, state, self.catflapcontrol)
    #     pass

    # def on_enter_idleState(self, event, state):


    # def on_enter_movementLockedState(self, event, state):
    #     pass

def main(data):

    # Create a test data structure - same as it would come from the ML system
    detections = TFLDetections("Name")
    for d in data:
        (index, name, score, origin_x, origin_y, width, height) = d
        detections.add(name, index, score, BoundingRect(origin_x, origin_y, width, height))

    # Run the data set through the event handler
    fsm = CatFlapFSM()
    for d in detections:
        fsm.event_handle(Event(d))

    import time
    i = 0
    while i < 90:
        i += 1
        time.sleep(1)
    a = [s.name for s in fsm.states]
    print(a)
    b = [s for s in fsm.states_map]
    print(b)


# Running image classification on /Users/toby/work/projects/catflap/data/test_data/Cat-with-mouse-52-20230804-001533.496_catcam.jpg
# Number of detections 3 in /Users/toby/work/projects/catflap/data/test_data/Cat-with-mouse-52-20230804-001533.496_catcam.jpg
# 1:Cat-with-mouse    0.55   BoundingRect(origin_x=67, origin_y=-5, width=441, height=181)
# 1:Cat-with-mouse    0.36   BoundingRect(origin_x=177, origin_y=-2, width=454, height=255)
# 1:Cat-with-mouse    0.36   BoundingRect(origin_x=58, origin_y=13, width=563, height=382)
# 
# BoundingBox(origin_x=190, origin_y=327, width=109, height=113
data = [
        (2, "Background", 0.53, 190, 327,109,113),
        (2, "Background", 0.53, 190, 327,108,113),
        (2, "Background", 0.53, 190, 327,107,113),
        (2, "Background", 0.53, 190, 327,106,113),
        (0, "Cat-alone",  0.53, 190, 327,105,113),
        (0, "Cat-alone",  0.53, 190, 327,104,113),
        (0, "Cat-alone",  0.53, 190, 327,103,113),
        (0, "Cat-alone",  0.55, 194, 337,119,133),
        (0, "Cat-alone",  0.59, 195, 347,122,183),
        (0, "Cat-alone",  0.63, 199, 357,199,193)
       ]

if __name__ == "__main__":
    # logger.basicConfig(level=logger.DEBUG)
    main(data)
