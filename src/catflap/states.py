
from statemachine import StateMachine
from base_logger import logger
from catflapcontrol import CatFlapControl
from statetimer import StateTimer

from statetypes import Event, EventKind, States
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

    def event_handle(self, event:Event):
        '''Handle an event from the analysis thread or the timeout timer.
        This is the first step in handling a new event from outside, and it
        always runs on the main thread.'''
        data = self._global_data
        if event.kind == EventKind.END:
            return
        in_idle = self.current_state.name == States.IDLE
        if event.kind == EventKind.TIMEOUT:
            if in_idle:
                return
            logger.info(f"PUML {self.current_state.id} -> idleState: Timeout")
            data.new_state = self.states[States.IDLE]
        else:
            if event.payload is None:
                return
            if event.armed == in_idle:
                # Analysed for a state we have already left (idle: movement
                # analysis, otherwise: ML), so the result is stale
                return
            logger.debug(f"{self.__class__.__name__} Handling incoming event {event} in state {self.current_state.id}")
            data.add_event(event)
            if event.armed:
                data.annotations.add(event.ts, event.frame_id, event.detections, self.current_state.id)
                if data.clips is not None:
                    data.clips.trigger(event.ts)    # Something is going on: keep recording
            data.new_state = self.states[self.current_state.run(event, data)]
        # Call the state transition action
        action = self.transitions[data.new_state.name]
        action(self)


    def _timeout_handle(self, args):
        '''This is a watchdog type timer to ensure the cat flap is always opened
        when a movement sequence ends.
        Note - is called in the Timer thread, so it only posts an event for the main thread.'''
        self._global_data.post_timeout()

    def exit(self):
        '''Called when the state machine must exit. Unlock cat flap and exit.'''
        logger.info("Handling state machine exit event")
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
        # The ML model only runs while there is something to look at
        self._global_data.set_armed(state.name != States.IDLE)
        if state.name != States.IDLE and self._global_data.clips is not None:
            ts = self._global_data.last_event_ts()
            if ts is not None:
                self._global_data.clips.trigger(ts)
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
