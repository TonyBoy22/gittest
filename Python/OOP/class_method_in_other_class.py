'''
How to use a class method or parameter inside another class
'''

class A(object):
    def __init__(self) -> None:
        self._p1 = 'first_parameter'
        self._p2 = 5.0
        
    def print_params(self):
        print('init parameters: ', self._p1, self._p2)
        
class B(object):
    def __init__(self) -> None:
        self._classA_method = None
        
    def use_A_method(self, A):
        self._classA_method = A.print_params
        return 
    
classA = A()
classB = B()

# initialize class B method
classB.use_A_method(classA)

classB._classA_method()