
class Event(object):

    def __init__(self, name):
        self._name = name

    def __eq__(self, other):
        if other.name == self._name:
            return True
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    @property
    def name(self):
        return self._name