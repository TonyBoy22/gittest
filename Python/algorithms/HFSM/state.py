'''
Implementation of a hierarchical Finite State Machine

Source: https://towardsdatascience.com/how-i-implemented-hfsm-in-python-65899c1fb1d0
https://github.com/debbynirwan/hfsm

About the decorators and properties
https://www.freecodecamp.org/news/python-property-decorator/
https://ron.sh/how-to-write-custom-python-decorators/
'''

'''
Separate classes for 3 components of the HFSM: State, Transition, Action
'''

class State(object):
    def __init__(self, name, child_sm=None) -> None:
        self._name = name
        # What the FSM does when it enter the state
        self._entry_callback: list[callable[[any], None]] = []
        # What it does when just before exiting the state
        self._exit_callback: list[callable[[any], None]] = []

    # Overload? Otherwise how is other_object.name == object.name evaluated?
    def __eq__(self, other):
        if other.name == self.name:
            return True
        else:
            return False
    
    # Negation. __eq__ and __ne__ are required for Python 2 compatibility?
    def __ne__(self, other):
        return not self.__eq__(other)

    # Useful to have the ability to call the class as if it was a function
    # If just pass, maybe just here for formality?   
    def __call__(self, data:any):
        pass

    def on_entry(self, callback: callable[[any], None]):
        self._entry_callback.append(callback)

    def on_exit(self, callback: callable[[], None]):
        self._exit_callbacks.append(callback)

    # Execute the accumulated callbacks when the FSM enter the state
    def start(self, data: any):
        for callback in self._entry_callbacks:
            callback(data)

    # 
    def stop(self, data: any):
        for callback in self._exit_callbacks:
            callback(data)

    # Property decorator defines a function as an attribute of the class object
    # Need the abolity to change its name?
    @property
    def name(self):
        return self._name

