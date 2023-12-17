
from statemachine import StateMachine, State
from enum import Enum
from base_logger import logger

from detections import TFLDetections, TFLDetection, BoundingRect
from evaluation import Evaluation, CatDetection
from catflapcontrol import CatFlapControl
from statetimer import StateTimer

class States(int, Enum):
    IDLE = 1
    UNLOCKED = 2
    MOVEMENT_LOCKED = 3
    MOUSE_LOCKED = 4


from abc import ABC, abstractmethod

class TState(State, ABC):
    @abstractmethod
    def run(self, event:TFLDetection) -> States:
        pass

   
    def __cmp__(self, other):
        return self.name == other.name
    
    # Necessary when __cmp__ or __eq__ is defined
    # in order to make this class usable as a
    # dictionary key:
    def __hash__(self):
        return hash(self.name)


class Idle(TState):
    def __init__(self, *args, **kwargs) -> None:
        super(Idle, self).__init__(*args, **kwargs)
        self._id = States.IDLE

    def run(self, event:TFLDetection, evaluation:Evaluation):
        '''Idle always goes straight to movement locked'''
        evaluation.add_record(event.label, event.score)
        return States.MOVEMENT_LOCKED

    def on_enter_state(self, event:TFLDetection, state:TState, control:CatFlapControl) -> None:
        '''When we return to idle we unlock the cat flap, so cats can exit from the inside, and
        reset the evaluation class ready for the next event sequence'''
        logger.info(f"Entering {self.__class__.__name__} state")
        control.cat_flap_unlock()
        logger.info(f"PUML idleState --> flapControl: cat-flap-unlock")
        
    def on_exit_state(self, event:TFLDetection, state:TState, control:CatFlapControl) -> None:
        '''Leaving idle state - create a newly initialised statistics class'''
        logger.debug(f"Exiting {self.__class__.__name__} state")


class MovementLocked(TState):
    def __init__(self, *args, **kwargs) -> None:
        super(MovementLocked, self).__init__(*args, **kwargs)

    def run(self, event:TFLDetection, evaluation:Evaluation):
        '''From Movement Locked we can either unlock (no mouse is there) or go into
        permanent lock when a mouse is detected.'''
        retval = States.MOVEMENT_LOCKED
        evaluation.add_record(event.label, event.score)
        if evaluation.result == CatDetection.CAT_WITH_MOUSE:
            retval = States.MOUSE_LOCKED
        elif evaluation.result == CatDetection.CAT_ALONE:
            retval = States.UNLOCKED
        return retval

    def on_enter_state(self, event:TFLDetection, state:TState, control:CatFlapControl) -> None:
        logger.info(f"Entering {self.__class__.__name__} state")
        control.cat_flap_lock()
        logger.info(f"PUML movementLockedState --> flapControl: cat-flap-lock")

class Unlocked(TState):
    def __init__(self, *args, **kwargs) -> None:
        super(Unlocked, self).__init__(*args, **kwargs)

    def run(self, event:TFLDetection, evaluation:Evaluation):
        '''The state machine will remain unlocked until a new movement is
        detected or a timeout takes it back to idle'''
        retval = States.UNLOCKED
        evaluation.add_record(event.label, event.score)
        if evaluation.result == CatDetection.CAT_WITH_MOUSE:
            retval = States.MOUSE_LOCKED
        return retval


    def on_enter_state(self, event:TFLDetection, state:TState, control:CatFlapControl) -> None:
        logger.info(f"Entering {self.__class__.__name__} state")
        control.cat_flap_unlock()
        logger.info(f"PUML unlockedState --> flapControl: cat-flap-unlock")

class MouseLocked(TState):
    def __init__(self, *args, **kwargs) -> None:
        super(MouseLocked, self).__init__(*args, **kwargs)

    def run(self, event:TFLDetection, evaluation:Evaluation):
        retval = States.MOUSE_LOCKED
        evaluation.add_record(event.label, event.score)
        if evaluation.result == CatDetection.CAT_ALONE:
            retval = States.MOVEMENT_LOCKED
        return retval

    def on_enter_state(self, event:TFLDetection, state:TState, control:CatFlapControl) -> None:
        logger.info(f"Entering {self.__class__.__name__} state")
        control.cat_flap_lock()
        logger.info(f"PUML mouseLockedState --> flapControl: cat-flap-lock")

    def on_exit_state(self, event:TFLDetection, state:TState, control:CatFlapControl) -> None:
        logger.info(f"Exiting {self.__class__.__name__} state")
        pass

class CatFlapFSM(StateMachine):

    idleState = Idle(name=States.IDLE, initial=True)
    unlockedState = Unlocked(name=States.UNLOCKED)
    movementLockedState = MovementLocked(name=States.MOVEMENT_LOCKED)
    mouseLockedState = MouseLocked(name=States.MOUSE_LOCKED)

    movementLock = idleState.to(movementLockedState) | mouseLockedState.to(movementLockedState)
    unlock = movementLockedState.to(unlockedState)
    returnIdle = movementLockedState.to(idleState) | mouseLockedState.to(idleState) |  unlockedState.to(idleState)
    mouseLock = movementLockedState.to(mouseLockedState) | unlockedState.to(mouseLockedState)


    def __init__(self) -> None:
        # Must come before parent init as this calls the Idle.on_enter_state method
        self.catflapcontrol = CatFlapControl()
        StateMachine.__init__(self, CatFlapFSM.idleState)
        
        logger.info("PUML nullState -> idleState: Startup")

        self.states = {
            States.IDLE: self.idleState,
            States.UNLOCKED: self.unlockedState,
            States.MOVEMENT_LOCKED: self.movementLockedState,
            States.MOUSE_LOCKED: self.mouseLockedState
        }
        self.transitions = {
            States.IDLE : self.returnIdle,
            States.UNLOCKED : self.unlock,
            States.MOVEMENT_LOCKED : self.movementLock,
            States.MOUSE_LOCKED : self.mouseLock
            }
        self.timeout_timer = None
        self.evaluation = None


    def event_handle(self, event:TFLDetection):
        '''Handle detection events from the cat AI system'''
        logger.info(f"Handling incoming detection event {event.detail} in state {self.current_state.id}")
        if self.evaluation == None:
            logger.debug(f"Reset the evaluation statistics to default values")
            self.evaluation = Evaluation()
        new_state = self.states[self.current_state.run(event, self.evaluation)]
        logger.info(f"PUML {self.current_state.id} -> {new_state.id}: {event}")

        if self.current_state == self.idleState and new_state != self.idleState:
            '''Start the timeout timer on the transition out of idleState only'''
            self.timeout_timer = StateTimer(self._timeout_handle)
            self.timeout_timer.timer_start()
        elif self.current_state != self.idleState and new_state == self.idleState:
            self.evaluation = None
            self.timeout_timer.timer_stop()

        if new_state != self.current_state:
            action = self.transitions[new_state.name]
            action(self)
            self.current_state = new_state


    def _timeout_handle(self, args):
        '''This is a watchdog type timer to ensure the cat flap is always opened
        when a movement sequence ends.'''
        logger.info(f"Handling timeout event")
        logger.info(f"PUML {self.current_state.id} -> idleState: Timeout")
        self.evaluation = None
        self.returnIdle()

    def exit(self):
        '''Called when the state machine must exit. Unlock cat flap and exit.'''
        logger.info(f"Handling state machine exit event")
        self.timeout_timer.timer_cancel()

    def before_transition(self, event, state):
        pass
        # logger.debug(f"Before '{event}', on the '{state.id}' state.")

    def on_transition(self, event, state):
        pass
        # logger.debug(f"On '{event}', on the '{state.id}' state.")

    def on_exit_state(self, event, state):
        if hasattr(state, "on_exit_state") == True:
            logger.debug(f"Exiting '{state.id}' state from '{event}' event.")
            state.on_exit_state(event, state, self.catflapcontrol)

    def on_enter_state(self, event, state):
        if hasattr(state, "on_enter_state") == True:
            logger.debug(f"Entering '{state.id}' state from '{event}' event.")
            state.on_enter_state(event, state, self.catflapcontrol)

    def after_transition(self, event, state):
        pass
        # logger.debug(f"After '{event}', on the '{state.id}' state.")


def main(data):
    # Create a test data structure - same as it would come from the ML system
    detections = TFLDetections("Name")
    for d in data:
        (index, name, score, origin_x, origin_y, width, height) = d
        detections.add(name, index, score, BoundingRect(origin_x, origin_y, width, height))

    # Run the data set through the event handler
    fsm = CatFlapFSM()
    for d in detections:
        fsm.event_handle(d)

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

#    BoundingBox(origin_x=190, origin_y=327, width=109, height=113
data = [
            (0, "Cat-alone",    0.53, 190, 327,109,113),
            (0, "Cat-alone",    0.55, 194, 337,119,133),
            (0, "Cat-alone",    0.59, 195, 347,122,183),
            (0, "Cat-alone",    0.63, 199, 357,199,193)
       ]

if __name__ == "__main__":
    logger.basicConfig(level=logger.DEBUG)
    main(data)
