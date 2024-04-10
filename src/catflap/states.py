
from statemachine import StateMachine
import sys
from base_logger import logger
from catflapcontrol import CatFlapControl
from statetimer import StateTimer

from statetypes import Event, States
from stateidle import IdleState
from statetriggering import TriggeringState
from statemovementlocked import MovementLockedState
from statemouselocked import MouseLockedState
from stateunlocked import UnlockedState


class CatFlapFSM(StateMachine):

    '''Define the states - NOTE - this is called as soon as the import statement is executed.
        This means the inits can't be used for anything that depends on the global data'''
    idleState = IdleState(name=States.IDLE, initial=True)
    triggeringState = TriggeringState(name=States.TRIGGERING)
    unlockedState = UnlockedState(name=States.UNLOCKED)
    movementLockedState = MovementLockedState(name=States.MOVEMENT_LOCKED)
    mouseLockedState = MouseLockedState(name=States.MOUSE_LOCKED)
    '''State transitions. All same-state transitions are internal, which means they
        don't trigger the _exit and _enter actions if the from and too states are 
        the same. 

        !!! Be careful editing this table - it can break internally for no apparent
        reason if you enter the wrong transition in the wrong place !!! If it gets broken
        the _enter_state and _exit_state get called every time and the Internal flag is
        ignored.
    '''
    triggering = idleState.to(triggeringState) | triggeringState.to.itself(internal=True)
    movementLock = triggeringState.to(movementLockedState) | movementLockedState.to.itself(internal=True)
    unlock = movementLockedState.to(unlockedState) | unlockedState.to.itself(internal=True)
    returnIdle = triggeringState.to(idleState) | movementLockedState.to(idleState) | mouseLockedState.to(idleState) |  unlockedState.to(idleState) | idleState.to.itself(internal=True)
    mouseLock = movementLockedState.to(mouseLockedState) | unlockedState.to(mouseLockedState) | mouseLockedState.to.itself(internal=True)


    def __init__(self, global_data) -> None:
        # Initialise the global data. This must be before the state machine init
        # as this init will cause us to call the on_enter_state for idle.
        # Create the timeout timer, and cat flap control object
        self._global_data = global_data
        self._global_data.timeout_timer = StateTimer(self._timeout_handle, interval=5)
        self._global_data.cat_flap_control = CatFlapControl()

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
        self._return_idle = False
        

    def event_handle(self, event:Event):
        '''Handle detection events from the cat AI system
            This is the first step in handling a new event from outside'''
        if type(event.payload) == type(None):
            logger.debug(f"{self.__class__.__name__} State machine exiting on None event")
            return

        logger.debug(f"{self.__class__.__name__} Handling incoming event {event} in state {self.current_state.id}")
        # Run the event handler for the current state with the incoming event and
        # the global data
        self._global_data.add_event(event)
        self._global_data.new_state = self.states[self.current_state.run(event, self._global_data)]
        if self._return_idle == True:
            self._global_data.new_state = self.states[States.IDLE]
            self._return_idle = False
        # Call the state transition action
        action = self.transitions[self._global_data.new_state.name]
        action(self)


    def _timeout_handle(self, args):
        '''This is a watchdog type timer to ensure the cat flap is always opened
        when a movement sequence ends.
        Note - is called in the Timer thread, so do not change state machine stuff.'''
        logger.info(f"PUML {self.current_state.id} -> idleState: Timeout")
        self._return_idle = True

    def exit(self):
        '''Called when the state machine must exit. Unlock cat flap and exit.'''
        logger.info(f"Handling state machine exit event")
        self._global_data.timeout_timer.cancel()
        self._global_data.cat_flap_control.unlock()

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
        # logger.debug(f"{self.__class__.__name__} on_exit_state: event  '{event}', exiting state '{state.id}'.")
        logger.info(f"PUML {self.current_state.id} -> {self._global_data.new_state.id}: {event}")
        if hasattr(state, "on_exit_state") == True:
            state.on_exit_state(event, self._global_data)

    def on_transition(self, event, state):
        '''3. Ready to transition from the old state to the new one. 
            The state parameter here is the state we are leaving.
            This is called on every transition - even self to self '''
        pass

    def on_enter_state(self, event, state):
        '''4. Entering the new state - the state parameter here is now the new state
            This will also enter idleState from __initial__'''
        logger.debug(f"{self.__class__.__name__} on_enter_state: event '{event}', entering state '{state.id}'.")
        if hasattr(state, "on_enter_state") == True:
            state.on_enter_state(event, self._global_data)

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
