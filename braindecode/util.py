from lasagne.init import Initializer
import theano

class FuncAndArgs(object):
    """Container for a function and its arguments. 
    Useful in case you want to pass a function and its arguments 
    to another function without creating a new class.
    You can call the new instance either with the apply method or 
    the ()-call operator:
    
    >>> FuncAndArgs(max, 2,3).apply(4)
    4
    >>> FuncAndArgs(max, 2,3)(4)
    4
    >>> FuncAndArgs(sum, [3,4])(8)
    15
    
    """
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        
    
    def apply(self, *other_args, **other_kwargs):
        all_args = self.args + other_args
        all_kwargs = self.kwargs.copy()
        all_kwargs.update(other_kwargs)
        return self.func(*all_args, **all_kwargs)
        
    def __call__(self, *other_args, **other_kwargs):
        return self.apply(*other_args, **other_kwargs)
    
    
class GetAttr(object):
    """ Hacky class for yaml to return attr of something.
    Uses new to actually return completely different object... """
    def __new__(cls, obj, attr):
        return getattr(obj, attr) 
