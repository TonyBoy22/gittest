class StateMachine(object):

    def __init__(self, name):
        self._name = name
        self._states: List[State] = []
        self._events: List[Event] = []
        self._transitions: List[Transition] = []
        self._initial_state: Optional[State] = None
        self._current_state: Optional[State] = None
        self._exit_callback: Optional[Callable[[ExitState, Any], None]] = None
        self._exit_state = ExitState()
        self.add_state(self._exit_state)
        self._exited = True

    def __eq__(self, other):
        if other.name == self._name:
            return True
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return self._name

    def start(self, data: Any):
        if not self._initial_state:
            raise ValueError("initial state is not set")
        self._current_state = self._initial_state
        self._exited = False
        self._current_state.start(data)

    def stop(self, data: Any):
        if not self._initial_state:
            raise ValueError("initial state is not set")
        if self._current_state is None:
            raise ValueError("state machine has not been started")
        self._current_state.stop(data)
        self._current_state = self._exit_state
        self._exited = True

    def on_exit(self, callback):
        self._exit_callback = callback

    def is_running(self) -> bool:
        if self._current_state and self._current_state != self._exit_state:
            return True
        else:
            return False

    def add_state(self, state: State, initial_state: bool = False):
        if state in self._states:
            raise ValueError("attempting to add same state twice")
        self._states.append(state)
        state.set_parent_sm(self)
        if not self._initial_state and initial_state:
            self._initial_state = state

    def add_event(self, event: Event):
        self._events.append(event)

    def add_transition(self, src: State, dst: State, evt: Event) -> \
            Optional[Transition]:
        transition = None
        if src in self._states and dst in self._states and evt in self._events:
            transition = NormalTransition(src, dst, evt)
            self._transitions.append(transition)
        return transition

    def add_self_transition(self, state: State, evt: Event) -> \
            Optional[Transition]:
        transition = None
        if state in self._states and evt in self._events:
            transition = SelfTransition(state, evt)
            self._transitions.append(transition)
        return transition

    def add_null_transition(self, state: State, evt: Event) -> \
            Optional[Transition]:
        transition = None
        if state in self._states and evt in self._events:
            transition = NullTransition(state, evt)
            self._transitions.append(transition)
        return transition

    def trigger_event(self, evt: Event, data: Any = None,
                      propagate: bool = False):
        transition_valid = False
        if not self._initial_state:
            raise ValueError("initial state is not set")

        if self._current_state is None:
            raise ValueError("state machine has not been started")

        for transition in self._transitions:
            if transition.source_state == self._current_state and \
                    transition.event == evt:
                self._current_state = transition.destination_state
                transition(data)
                if isinstance(self._current_state, ExitState) and \
                        not self._exited:
                    self._exited = True
                    if self._exit_callback:
                        self._exit_callback(self._current_state, data)
                transition_valid = True
                break

    @property
    def exit_state(self):
        return self._exit_state

    @property
    def current_state(self):
        return self._current_state

    @property
    def name(self):
        return self._name