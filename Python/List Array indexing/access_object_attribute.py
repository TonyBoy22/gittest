'''
Check methods to access attributes from an object that is inside a list
'''

class RandomObject(object):
    def __init__(self, i) -> None:
        self._a = 'a'
        self._b = 8
        self._index = i
        return None
    
object_list = []

for i in range(5):
    obj = RandomObject(i)
    object_list.append(obj)
    
    
print(object_list)
    