'''
Class and metaclass definition
Tutorial source:
https://www.youtube.com/watch?v=NAQEj-c2CI8
'''

# classic definition of a class
class Test():
    pass

# Can also be defined as
AlsoTest = type('AlsoTest', (), {})
'''
Arguments in parenthesis are parent classes from which AlsoTest inherits
Arguments in brackets are the attributes of AlsoTest in a dict format
i.e. type(name, bases, attrs)
'''
