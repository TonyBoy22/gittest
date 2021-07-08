class Transition(object):
    '''
    On peut définir avec les 2 points une propriété dans la déclaration des
    arguments de la fonction
    '''
    def __init__(self, event: Event, src: State, dst: State) -> None:
        self._event = event
        self._source_state = src
        self._destination_state = dst
        self._condition: Optional[Callable[[Any], bool]] = None
        self._action: Optional[Callable[[Any], None]] = None

    def __call__(self, data: Any):
        raise NotImplementedError

    def add_condition(self, callback: Callable[[Any], bool]):
        self._condition = callback

    def add_action(self, callback: Callable[[Any], Any]):
        self._action = callback

    @property
    def event(self):
        return self._event

    @property
    def source_state(self):
        return self._source_state

    @property
    def destination_state(self):
        return self._destination_state

class NormalTransition(Transition):

    def __init__(self, source_state: State, destination_state: State,
                 event: Event):
        super().__init__(event, source_state, destination_state)
        self._from = source_state
        self._to = destination_state

    def __call__(self, data: Any):
        if not self._condition or self._condition(data):
            if self._action:
                self._action(data)
            self._from.stop(data)
            self._to.start(data)

class SelfTransition(Transition):

    def __init__(self, source_state: State, event: Event):
        super().__init__(event, source_state, source_state)
        self._state = source_state

    def __call__(self, data: Any):
        if not self._condition or self._condition(data):
            if self._action:
                self._action(data)
            self._state.stop(data)
            self._state.start(data)

class NullTransition(Transition):

    def __init__(self, source_state: State, event: Event):
        super().__init__(event, source_state, source_state)
        self._state = source_state

    def __call__(self, data: Any):
        if not self._condition or self._condition(data):
            if self._action:
                self._action(data)