import numpy as np
import theano
from theano.tensor.nnet import conv2d
import theano.tensor as T
import logging
from numpy.random import RandomState
from collections import OrderedDict
import theano.gof
import theano.compile
import theano.tensor.nnet
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from copy import copy, deepcopy
from collections import deque
from theano.tensor.signal import downsample
import time
from abc import abstractmethod, ABCMeta
from itertools import chain
from inspect import getargspec
from difflib import get_close_matches
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

# sigmoid
def sigmoid(x):
    """Sigmoid activation function :math:`\\varphi(x) = \\frac{1}{1 + e^{-x}}`
    Parameters
    ----------
    x : float32
        The activation (the summed, weighted input of a neuron).
    Returns
    -------
    float32 in [0, 1]
        The output of the sigmoid function applied to the activation.
    """
    return theano.tensor.nnet.sigmoid(x)


# softmax (row-wise)
def softmax(x):
    """Softmax activation function
    :math:`\\varphi(\\mathbf{x})_j =
    \\frac{e^{\mathbf{x}_j}}{\sum_{k=1}^K e^{\mathbf{x}_k}}`
    where :math:`K` is the total number of neurons in the layer. This
    activation function gets applied row-wise.
    Parameters
    ----------
    x : float32
        The activation (the summed, weighted input of a neuron).
    Returns
    -------
    float32 where the sum of the row is 1 and each single value is in [0, 1]
        The output of the softmax function applied to the activation.
    """
    return theano.tensor.nnet.softmax(x)



# rectify
def rectify(x):
    """Rectify activation function :math:`\\varphi(x) = \\max(0, x)`
    Parameters
    ----------
    x : float32
        The activation (the summed, weighted input of a neuron).
    Returns
    -------
    float32
        The output of the rectify function applied to the activation.
    """
    return theano.tensor.nnet.relu(x)


# linear
def linear(x):
    """Linear activation function :math:`\\varphi(x) = x`
    Parameters
    ----------
    x : float32
        The activation (the summed, weighted input of a neuron).
    Returns
    -------
    float32
        The output of the identity applied to the activation.
    """
    return x

identity = linear
def collect_shared_vars(expressions):
    """Returns all shared variables the given expression(s) depend on.
    Parameters
    ----------
    expressions : Theano expression or iterable of Theano expressions
        The expressions to collect shared variables from.
    Returns
    -------
    list of Theano shared variables
        All shared variables the given expression(s) depend on, in fixed order
        (as found by a left-recursive depth-first search). If some expressions
        are shared variables themselves, they are included in the result.
    """
    # wrap single expression in list
    if isinstance(expressions, theano.Variable):
        expressions = [expressions]
    # return list of all shared variables
    return [v for v in theano.gof.graph.inputs(reversed(expressions))
            if isinstance(v, theano.compile.SharedVariable)]
    
def as_tuple(x, N, t=None):
    """
    Coerce a value to a tuple of given length (and possibly given type).
    Parameters
    ----------
    x : value or iterable
    N : integer
        length of the desired tuple
    t : type, optional
        required type for all elements
    Returns
    -------
    tuple
        ``tuple(x)`` if `x` is iterable, ``(x,) * N`` otherwise.
    Raises
    ------
    TypeError
        if `type` is given and `x` or any of its elements do not match it
    ValueError
        if `x` is iterable, but does not have exactly `N` elements
    """
    try:
        X = tuple(x)
    except TypeError:
        X = (x,) * N

    if (t is not None) and not all(isinstance(v, t) for v in X):
        raise TypeError("expected a single value or an iterable "
                        "of {0}, got {1} instead".format(t.__name__, x))

    if len(X) != N:
        raise ValueError("expected a single value or an iterable "
                         "with length {0}, got {1} instead".format(N, x))

    return X


class Layer(object):
    """
    The :class:`Layer` class represents a single layer of a neural network. It
    should be subclassed when implementing new types of layers.
    Because each layer can keep track of the layer(s) feeding into it, a
    network's output :class:`Layer` instance can double as a handle to the full
    network.
    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape.
    name : a string or None
        An optional name to attach to this layer.
    """
    def __init__(self, incoming, name=None):
        if isinstance(incoming, tuple):
            self.input_shape = incoming
            self.input_layer = None
        else:
            self.input_shape = incoming.output_shape
            self.input_layer = incoming

        self.name = name
        self.params = OrderedDict()
        self.get_output_kwargs = []

        if any(d is not None and d <= 0 for d in self.input_shape):
            raise ValueError((
                "Cannot create Layer with a non-positive input_shape "
                "dimension. input_shape=%r, self.name=%r") % (
                    self.input_shape, self.name))

    @property
    def output_shape(self):
        return self.get_output_shape_for(self.input_shape)

    def get_params(self, **tags):
        """
        Returns a list of Theano shared variables that parameterize the layer.
        By default, all shared variables that participate in the forward pass
        will be returned (in the order they were registered in the Layer's
        constructor via :meth:`add_param()`). The list can optionally be
        filtered by specifying tags as keyword arguments. For example,
        ``trainable=True`` will only return trainable parameters, and
        ``regularizable=True`` will only return parameters that can be
        regularized (e.g., by L2 decay).
        If any of the layer's parameters was set to a Theano expression instead
        of a shared variable, the shared variables involved in that expression
        will be returned rather than the expression itself. Tag filtering
        considers all variables within an expression to be tagged the same.
        Parameters
        ----------
        **tags (optional)
            tags can be specified to filter the list. Specifying ``tag1=True``
            will limit the list to parameters that are tagged with ``tag1``.
            Specifying ``tag1=False`` will limit the list to parameters that
            are not tagged with ``tag1``. Commonly used tags are
            ``regularizable`` and ``trainable``.
        Returns
        -------
        list of Theano shared variables
            A list of variables that parameterize the layer
        Notes
        -----
        For layers without any parameters, this will return an empty list.
        """
        result = list(self.params.keys())

        only = set(tag for tag, value in tags.items() if value)
        if only:
            # retain all parameters that have all of the tags in `only`
            result = [param for param in result
                      if not (only - self.params[param])]

        exclude = set(tag for tag, value in tags.items() if not value)
        if exclude:
            # retain all parameters that have none of the tags in `exclude`
            result = [param for param in result
                      if not (self.params[param] & exclude)]

        return collect_shared_vars(result)

    def get_output_shape_for(self, input_shape):
        """
        Computes the output shape of this layer, given an input shape.
        Parameters
        ----------
        input_shape : tuple
            A tuple representing the shape of the input. The tuple should have
            as many elements as there are input dimensions, and the elements
            should be integers or `None`.
        Returns
        -------
        tuple
            A tuple representing the shape of the output of this layer. The
            tuple has as many elements as there are output dimensions, and the
            elements are all either integers or `None`.
        Notes
        -----
        This method will typically be overridden when implementing a new
        :class:`Layer` class. By default it simply returns the input
        shape. This means that a layer that does not modify the shape
        (e.g. because it applies an elementwise operation) does not need
        to override this method.
        """
        return input_shape

    def get_output_for(self, input, **kwargs):
        """
        Propagates the given input through this layer (and only this layer).
        Parameters
        ----------
        input : Theano expression
            The expression to propagate through this layer.
        Returns
        -------
        output : Theano expression
            The output of this layer given the input to this layer.
        Notes
        -----
        This is called by the base :meth:`lasagne.layers.get_output()`
        to propagate data through a network.
        This method should be overridden when implementing a new
        :class:`Layer` class. By default it raises `NotImplementedError`.
        """
        raise NotImplementedError

    def add_param(self, spec, shape, name=None, **tags):
        """
        Register and possibly initialize a parameter tensor for the layer.
        When defining a layer class, this method is called in the constructor
        to define which parameters the layer has, what their shapes are, how
        they should be initialized and what tags are associated with them.
        This allows layer classes to transparently support parameter
        initialization from numpy arrays and callables, as well as setting
        parameters to existing Theano shared variables or Theano expressions.
        All registered parameters are stored along with their tags in the
        ordered dictionary :attr:`Layer.params`, and can be retrieved with
        :meth:`Layer.get_params()`, optionally filtered by their tags.
        Parameters
        ----------
        spec : Theano shared variable, expression, numpy array or callable
            initial value, expression or initializer for this parameter.
            See :func:`lasagne.utils.create_param` for more information.
        shape : tuple of int
            a tuple of integers representing the desired shape of the
            parameter tensor.
        name : str (optional)
            a descriptive name for the parameter variable. This will be passed
            to ``theano.shared`` when the variable is created, prefixed by the
            layer's name if any (in the form ``'layer_name.param_name'``). If
            ``spec`` is already a shared variable or expression, this parameter
            will be ignored to avoid overwriting an existing name.
        **tags (optional)
            tags associated with the parameter can be specified as keyword
            arguments. To associate the tag ``tag1`` with the parameter, pass
            ``tag1=True``.
            By default, the tags ``regularizable`` and ``trainable`` are
            associated with the parameter. Pass ``regularizable=False`` or
            ``trainable=False`` respectively to prevent this.
        Returns
        -------
        Theano shared variable or Theano expression
            the resulting parameter variable or parameter expression
        Notes
        -----
        It is recommended to assign the resulting parameter variable/expression
        to an attribute of the layer for easy access, for example:
        >>> self.W = self.add_param(W, (2, 3), name='W')  #doctest: +SKIP
        """
        # prefix the param name with the layer name if it exists
        if name is not None:
            if self.name is not None:
                name = "%s.%s" % (self.name, name)
        # create shared variable, or pass through given variable/expression
        param = create_param(spec, shape, name)
        # parameters should be trainable and regularizable by default
        tags['trainable'] = tags.get('trainable', True)
        tags['regularizable'] = tags.get('regularizable', True)
        self.params[param] = set(tag for tag, value in tags.items() if value)

        return param
    

def floatX(arr):
    """Converts data to a numpy array of dtype ``theano.config.floatX``.
    Parameters
    ----------
    arr : array_like
        The data to be converted.
    Returns
    -------
    numpy ndarray
        The input array in the ``floatX`` dtype configured for Theano.
        If `arr` is an ndarray of correct dtype, it is returned as is.
    """
    return np.asarray(arr, dtype=theano.config.floatX)

def create_param(spec, shape, name=None):
    """
    Helper method to create Theano shared variables for layer parameters
    and to initialize them.
    Parameters
    ----------
    spec : numpy array, Theano expression, or callable
        Either of the following:
        * a numpy array with the initial parameter values
        * a Theano expression or shared variable representing the parameters
        * a function or callable that takes the desired shape of
          the parameter array as its single argument and returns
          a numpy array.
    shape : iterable of int
        a tuple or other iterable of integers representing the desired
        shape of the parameter array.
    name : string, optional
        If a new variable is created, the name to give to the parameter
        variable. This is ignored if `spec` is already a Theano expression
        or shared variable.
    Returns
    -------
    Theano shared variable or Theano expression
        A Theano shared variable or expression representing layer parameters.
        If a numpy array was provided, a shared variable is initialized to
        contain this array. If a shared variable or expression was provided,
        it is simply returned. If a callable was provided, it is called, and
        its output is used to initialize a shared variable.
    Notes
    -----
    This function is called by :meth:`Layer.add_param()` in the constructor
    of most :class:`Layer` subclasses. This enables those layers to
    support initialization with numpy arrays, existing Theano shared variables
    or expressions, and callables for generating initial parameter values.
    """
    shape = tuple(shape)  # convert to tuple if needed
    if any(d <= 0 for d in shape):
        raise ValueError((
            "Cannot create param with a non-positive shape dimension. "
            "Tried to create param with shape=%r, name=%r") % (shape, name))

    if isinstance(spec, theano.Variable):
        # We cannot check the shape here, Theano expressions (even shared
        # variables) do not have a fixed compile-time shape. We can check the
        # dimensionality though.
        # Note that we cannot assign a name here. We could assign to the
        # `name` attribute of the variable, but the user may have already
        # named the variable and we don't want to override this.
        if spec.ndim != len(shape):
            raise RuntimeError("parameter variable has %d dimensions, "
                               "should be %d" % (spec.ndim, len(shape)))
        return spec

    elif isinstance(spec, np.ndarray):
        if spec.shape != shape:
            raise RuntimeError("parameter array has shape %s, should be "
                               "%s" % (spec.shape, shape))
        return theano.shared(spec, name=name)

    elif hasattr(spec, '__call__'):
        arr = spec(shape)
        try:
            arr = floatX(arr)
        except Exception:
            raise RuntimeError("cannot initialize parameters: the "
                               "provided callable did not return an "
                               "array-like value")
        if arr.shape != shape:
            raise RuntimeError("cannot initialize parameters: the "
                               "provided callable did not return a value "
                               "with the correct shape")
        return theano.shared(arr, name=name)

    else:
        raise RuntimeError("cannot initialize parameters: 'spec' is not "
                           "a numpy array, a Theano expression, or a "
                           "callable")
    
class InputLayer(Layer):
    """
    This layer holds a symbolic variable that represents a network input. A
    variable can be specified when the layer is instantiated, else it is
    created.
    Parameters
    ----------
    shape : tuple of `int` or `None` elements
        The shape of the input. Any element can be `None` to indicate that the
        size of that dimension is not fixed at compile time.
    input_var : Theano symbolic variable or `None` (default: `None`)
        A variable representing a network input. If it is not provided, a
        variable will be created.
    Raises
    ------
    ValueError
        If the dimension of `input_var` is not equal to `len(shape)`
    Notes
    -----
    The first dimension usually indicates the batch size. If you specify it,
    Theano may apply more optimizations while compiling the training or
    prediction function, but the compiled function will not accept data of a
    different batch size at runtime. To compile for a variable batch size, set
    the first shape element to `None` instead.
    Examples
    --------
    >>> from lasagne.layers import InputLayer
    >>> l_in = InputLayer((100, 20))
    """
    def __init__(self, shape, input_var=None, name=None, **kwargs):
        self.shape = shape
        if any(d is not None and d <= 0 for d in self.shape):
            raise ValueError((
                "Cannot create InputLayer with a non-positive shape "
                "dimension. shape=%r, self.name=%r") % (
                    self.shape, name))

        ndim = len(shape)
        if input_var is None:
            # create the right TensorType for the given number of dimensions
            input_var_type = T.TensorType(theano.config.floatX, [False] * ndim)
            var_name = ("%s.input" % name) if name is not None else "input"
            input_var = input_var_type(var_name)
        else:
            # ensure the given variable has the correct dimensionality
            if input_var.ndim != ndim:
                raise ValueError("shape has %d dimensions, but variable has "
                                 "%d" % (ndim, input_var.ndim))
        self.input_var = input_var
        self.name = name
        self.params = OrderedDict()

    @Layer.output_shape.getter
    def output_shape(self):
        return self.shape


class NonlinearityLayer(Layer):
    """
    lasagne.layers.NonlinearityLayer(incoming,
    nonlinearity=lasagne.nonlinearities.rectify, **kwargs)
    A layer that just applies a nonlinearity.
    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape
    nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations. If None
        is provided, the layer will be linear.
    """
    def __init__(self, incoming, nonlinearity,
                 **kwargs):
        super(NonlinearityLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = nonlinearity

    def get_output_for(self, input, **kwargs):
        return self.nonlinearity(input)


class DimshuffleLayer(Layer):
    """
    A layer that rearranges the dimension of its input tensor, maintaining
    the same same total number of elements.
    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        the layer feeding into this layer, or the expected input shape
    pattern : tuple
        The new dimension order, with each element giving the index
        of the dimension in the input tensor or `'x'` to broadcast it.
        For example `(3,2,1,0)` will reverse the order of a 4-dimensional
        tensor. Use `'x'` to broadcast, e.g. `(3,2,1,'x',0)` will
        take a 4 tensor of shape `(2,3,5,7)` as input and produce a
        tensor of shape `(7,5,3,1,2)` with the 4th dimension being
        broadcast-able. In general, all dimensions in the input tensor
        must be used to generate the output tensor. Omitting a dimension
        attempts to collapse it; this can only be done to broadcast-able
        dimensions, e.g. a 5-tensor of shape `(7,5,3,1,2)` with the 4th
        being broadcast-able can be shuffled with the pattern `(4,2,1,0)`
        collapsing the 4th dimension resulting in a tensor of shape
        `(2,3,5,7)`.
    Examples
    --------
    >>> from lasagne.layers import InputLayer, DimshuffleLayer
    >>> l_in = InputLayer((2, 3, 5, 7))
    >>> l1 = DimshuffleLayer(l_in, (3, 2, 1, 'x', 0))
    >>> l1.output_shape
    (7, 5, 3, 1, 2)
    >>> l2 = DimshuffleLayer(l1, (4, 2, 1, 0))
    >>> l2.output_shape
    (2, 3, 5, 7)
    """
    def __init__(self, incoming, pattern, **kwargs):
        super(DimshuffleLayer, self).__init__(incoming, **kwargs)

        # Sanity check the pattern
        used_dims = set()
        for p in pattern:
            if isinstance(p, int):
                # Dimension p
                if p in used_dims:
                    raise ValueError("pattern contains dimension {0} more "
                                     "than once".format(p))
                used_dims.add(p)
            elif p == 'x':
                # Broadcast
                pass
            else:
                raise ValueError("pattern should only contain dimension"
                                 "indices or 'x', not {0}".format(p))

        self.pattern = pattern

        # try computing the output shape once as a sanity check
        self.get_output_shape_for(self.input_shape)

    def get_output_shape_for(self, input_shape):
        # Build output shape while keeping track of the dimensions that we are
        # attempting to collapse, so we can ensure that they are broadcastable
        output_shape = []
        dims_used = [False] * len(input_shape)
        for p in self.pattern:
            if isinstance(p, int):
                if p < 0 or p >= len(input_shape):
                    raise ValueError("pattern contains {0}, but input shape "
                                     "has {1} dimensions "
                                     "only".format(p, len(input_shape)))
                # Dimension p
                o = input_shape[p]
                dims_used[p] = True
            elif p == 'x':
                # Broadcast; will be of size 1
                o = 1
            output_shape.append(o)

        for i, (dim_size, used) in enumerate(zip(input_shape, dims_used)):
            if not used and dim_size != 1 and dim_size is not None:
                raise ValueError(
                    "pattern attempted to collapse dimension "
                    "{0} of size {1}; dimensions with size != 1/None are not"
                    "broadcastable and cannot be "
                    "collapsed".format(i, dim_size))

        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        return input.dimshuffle(self.pattern)
    

def conv_output_length(input_length, filter_size, stride, pad=0):
    """Helper function to compute the output size of a convolution operation
    This function computes the length along a single axis, which corresponds
    to a 1D convolution. It can also be used for convolutions with higher
    dimensionalities by using it individually for each axis.
    Parameters
    ----------
    input_length : int
        The size of the input.
    filter_size : int
        The size of the filter.
    stride : int
        The stride of the convolution operation.
    pad : int, 'full' or 'same' (default: 0)
        By default, the convolution is only computed where the input and the
        filter fully overlap (a valid convolution). When ``stride=1``, this
        yields an output that is smaller than the input by ``filter_size - 1``.
        The `pad` argument allows you to implicitly pad the input with zeros,
        extending the output size.
        A single integer results in symmetric zero-padding of the given size on
        both borders.
        ``'full'`` pads with one less than the filter size on both sides. This
        is equivalent to computing the convolution wherever the input and the
        filter overlap by at least one position.
        ``'same'`` pads with half the filter size on both sides (one less on
        the second side for an even filter size). When ``stride=1``, this
        results in an output size equal to the input size.
    Returns
    -------
    int
        The output size corresponding to the given convolution parameters.
    Raises
    ------
    RuntimeError
        When an invalid padding is specified, a `RuntimeError` is raised.
    """
    if input_length is None:
        return None
    if pad == 'valid':
        output_length = input_length - filter_size + 1
    elif pad == 'full':
        output_length = input_length + filter_size - 1
    elif pad == 'same':
        output_length = input_length
    elif isinstance(pad, int):
        output_length = input_length + 2 * pad - filter_size + 1
    else:
        raise ValueError('Invalid pad: {0}'.format(pad))

    # This is the integer arithmetic equivalent to
    # np.ceil(output_length / stride)
    output_length = (output_length + stride - 1) // stride

    return output_length




class Initializer(object):
    """Base class for parameter tensor initializers.
    The :class:`Initializer` class represents a weight initializer used
    to initialize weight parameters in a neural network layer. It should be
    subclassed when implementing new types of weight initializers.
    """
    def __call__(self, shape):
        """
        Makes :class:`Initializer` instances callable like a function, invoking
        their :meth:`sample()` method.
        """
        return self.sample(shape)

    def sample(self, shape):
        """
        Sample should return a theano.tensor of size shape and data type
        theano.config.floatX.
        Parameters
        -----------
        shape : tuple or int
            Integer or tuple specifying the size of the returned
            matrix.
        returns : theano.tensor
            Matrix of size shape and dtype theano.config.floatX.
        """
        raise NotImplementedError()

_rng = np.random


def get_rng():
    """Get the package-level random number generator.
    Returns
    -------
    :class:`numpy.random.RandomState` instance
        The :class:`numpy.random.RandomState` instance passed to the most
        recent call of :func:`set_rng`, or ``numpy.random`` if :func:`set_rng`
        has never been called.
    """
    return _rng


def set_rng(new_rng):
    """Set the package-level random number generator.
    Parameters
    ----------
    new_rng : ``numpy.random`` or a :class:`numpy.random.RandomState` instance
        The random number generator to use.
    """
    global _rng
    _rng = new_rng
    
class Uniform(Initializer):
    """Sample initial weights from the uniform distribution.
    Parameters are sampled from U(a, b).
    Parameters
    ----------
    range : float or tuple
        When std is None then range determines a, b. If range is a float the
        weights are sampled from U(-range, range). If range is a tuple the
        weights are sampled from U(range[0], range[1]).
    std : float or None
        If std is a float then the weights are sampled from
        U(mean - np.sqrt(3) * std, mean + np.sqrt(3) * std).
    mean : float
        see std for description.
    """
    def __init__(self, range=0.01, std=None, mean=0.0):
        if std is not None:
            a = mean - np.sqrt(3) * std
            b = mean + np.sqrt(3) * std
        else:
            try:
                a, b = range  # range is a tuple
            except TypeError:
                a, b = -range, range  # range is a number

        self.range = (a, b)

    def sample(self, shape):
        return floatX(get_rng().uniform(
            low=self.range[0], high=self.range[1], size=shape))


class Constant(Initializer):
    """Initialize weights with constant value.
    Parameters
    ----------
     val : float
        Constant value for weights.
    """
    def __init__(self, val=0.0):
        self.val = val

    def sample(self, shape):
        return floatX(np.ones(shape) * self.val)    
    
class Glorot(Initializer):
    """Glorot weight initialization.
    This is also known as Xavier initialization [1]_.
    Parameters
    ----------
    initializer : lasagne.init.Initializer
        Initializer used to sample the weights, must accept `std` in its
        constructor to sample from a distribution with a given standard
        deviation.
    gain : float or 'relu'
        Scaling factor for the weights. Set this to ``1.0`` for linear and
        sigmoid units, to 'relu' or ``sqrt(2)`` for rectified linear units, and
        to ``sqrt(2/(1+alpha**2))`` for leaky rectified linear units with
        leakiness ``alpha``. Other transfer functions may need different
        factors.
    c01b : bool
        For a :class:`lasagne.layers.cuda_convnet.Conv2DCCLayer` constructed
        with ``dimshuffle=False``, `c01b` must be set to ``True`` to compute
        the correct fan-in and fan-out.
    References
    ----------
    .. [1] Xavier Glorot and Yoshua Bengio (2010):
           Understanding the difficulty of training deep feedforward neural
           networks. International conference on artificial intelligence and
           statistics.
    Notes
    -----
    For a :class:`DenseLayer <lasagne.layers.DenseLayer>`, if ``gain='relu'``
    and ``initializer=Uniform``, the weights are initialized as
    .. math::
       a &= \\sqrt{\\frac{12}{fan_{in}+fan_{out}}}\\\\
       W &\sim U[-a, a]
    If ``gain=1`` and ``initializer=Normal``, the weights are initialized as
    .. math::
       \\sigma &= \\sqrt{\\frac{2}{fan_{in}+fan_{out}}}\\\\
       W &\sim N(0, \\sigma)
    See Also
    --------
    GlorotNormal  : Shortcut with Gaussian initializer.
    GlorotUniform : Shortcut with uniform initializer.
    """
    def __init__(self, initializer, gain=1.0, c01b=False):
        if gain == 'relu':
            gain = np.sqrt(2)

        self.initializer = initializer
        self.gain = gain
        self.c01b = c01b

    def sample(self, shape):
        if self.c01b:
            if len(shape) != 4:
                raise RuntimeError(
                    "If c01b is True, only shapes of length 4 are accepted")

            n1, n2 = shape[0], shape[3]
            receptive_field_size = shape[1] * shape[2]
        else:
            if len(shape) < 2:
                raise RuntimeError(
                    "This initializer only works with shapes of length >= 2")

            n1, n2 = shape[:2]
            receptive_field_size = np.prod(shape[2:])

        std = self.gain * np.sqrt(2.0 / ((n1 + n2) * receptive_field_size))
        return self.initializer(std=std).sample(shape)

class GlorotUniform(Glorot):
    """Glorot with weights sampled from the Uniform distribution.
    See :class:`Glorot` for a description of the parameters.
    """
    def __init__(self, gain=1.0, c01b=False):
        super(GlorotUniform, self).__init__(Uniform, gain, c01b)
class BaseConvLayer(Layer):
    """
    lasagne.layers.BaseConvLayer(incoming, num_filters, filter_size,
    stride=1, pad=0, untie_biases=False,
    W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
    nonlinearity=lasagne.nonlinearities.rectify, flip_filters=True,
    n=None, **kwargs)
    Convolutional layer base class
    Base class for performing an `n`-dimensional convolution on its input,
    optionally adding a bias and applying an elementwise nonlinearity. Note
    that this class cannot be used in a Lasagne network, only its subclasses
    can (e.g., :class:`Conv1DLayer`, :class:`Conv2DLayer`).
    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape. Must
        be a tensor of 2+`n` dimensions:
        ``(batch_size, num_input_channels, <n spatial dimensions>)``.
    num_filters : int
        The number of learnable convolutional filters this layer has.
    filter_size : int or iterable of int
        An integer or an `n`-element tuple specifying the size of the filters.
    stride : int or iterable of int
        An integer or an `n`-element tuple specifying the stride of the
        convolution operation.
    pad : int, iterable of int, 'full', 'same' or 'valid' (default: 0)
        By default, the convolution is only computed where the input and the
        filter fully overlap (a valid convolution). When ``stride=1``, this
        yields an output that is smaller than the input by ``filter_size - 1``.
        The `pad` argument allows you to implicitly pad the input with zeros,
        extending the output size.
        A single integer results in symmetric zero-padding of the given size on
        all borders, a tuple of `n` integers allows different symmetric padding
        per dimension.
        ``'full'`` pads with one less than the filter size on both sides. This
        is equivalent to computing the convolution wherever the input and the
        filter overlap by at least one position.
        ``'same'`` pads with half the filter size (rounded down) on both sides.
        When ``stride=1`` this results in an output size equal to the input
        size. Even filter size is not supported.
        ``'valid'`` is an alias for ``0`` (no padding / a valid convolution).
        Note that ``'full'`` and ``'same'`` can be faster than equivalent
        integer values due to optimizations by Theano.
    untie_biases : bool (default: False)
        If ``False``, the layer will have a bias parameter for each channel,
        which is shared across all positions in this channel. As a result, the
        `b` attribute will be a vector (1D).
        If ``True``, the layer will have separate bias parameters for each
        position in each channel. As a result, the `b` attribute will be an
        `n`-dimensional tensor.
    W : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the weights.
        These should be a tensor of 2+`n` dimensions with shape
        ``(num_filters, num_input_channels, <n spatial dimensions>)``.
        See :func:`lasagne.utils.create_param` for more information.
    b : Theano shared variable, expression, numpy array, callable or ``None``
        Initial value, expression or initializer for the biases. If set to
        ``None``, the layer will have no biases. Otherwise, biases should be
        a 1D array with shape ``(num_filters,)`` if `untied_biases` is set to
        ``False``. If it is set to ``True``, its shape should be
        ``(num_filters, <n spatial dimensions>)`` instead.
        See :func:`lasagne.utils.create_param` for more information.
    nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations. If None
        is provided, the layer will be linear.
    flip_filters : bool (default: True)
        Whether to flip the filters before sliding them over the input,
        performing a convolution (this is the default), or not to flip them and
        perform a correlation. Note that for some other convolutional layers in
        Lasagne, flipping incurs an overhead and is disabled by default --
        check the documentation when using learned weights from another layer.
    n : int or None
        The dimensionality of the convolution (i.e., the number of spatial
        dimensions of each feature map and each convolutional filter). If
        ``None``, will be inferred from the input shape.
    **kwargs
        Any additional keyword arguments are passed to the `Layer` superclass.
    Attributes
    ----------
    W : Theano shared variable or expression
        Variable or expression representing the filter weights.
    b : Theano shared variable or expression
        Variable or expression representing the biases.
    """
    def __init__(self, incoming, num_filters, filter_size, stride=1, pad=0,
                 untie_biases=False,
                 W=GlorotUniform(), b=Constant(0.),
                 nonlinearity=rectify, flip_filters=True,
                 n=None, **kwargs):
        super(BaseConvLayer, self).__init__(incoming, **kwargs)
        if nonlinearity is None:
            self.nonlinearity = identity
        else:
            self.nonlinearity = nonlinearity

        if n is None:
            n = len(self.input_shape) - 2
        elif n != len(self.input_shape) - 2:
            raise ValueError("Tried to create a %dD convolution layer with "
                             "input shape %r. Expected %d input dimensions "
                             "(batchsize, channels, %d spatial dimensions)." % 
                             (n, self.input_shape, n + 2, n))
        self.n = n
        self.num_filters = num_filters
        self.filter_size = as_tuple(filter_size, n, int)
        self.flip_filters = flip_filters
        self.stride = as_tuple(stride, n, int)
        self.untie_biases = untie_biases

        if pad == 'same':
            if any(s % 2 == 0 for s in self.filter_size):
                raise NotImplementedError(
                    '`same` padding requires odd filter size.')
        if pad == 'valid':
            self.pad = as_tuple(0, n)
        elif pad in ('full', 'same'):
            self.pad = pad
        else:
            self.pad = as_tuple(pad, n, int)

        self.W = self.add_param(W, self.get_W_shape(), name="W")
        if b is None:
            self.b = None
        else:
            if self.untie_biases:
                biases_shape = (num_filters,) + self.output_shape[2:]
            else:
                biases_shape = (num_filters,)
            self.b = self.add_param(b, biases_shape, name="b",
                                    regularizable=False)

    def get_W_shape(self):
        """Get the shape of the weight matrix `W`.
        Returns
        -------
        tuple of int
            The shape of the weight matrix.
        """
        num_input_channels = self.input_shape[1]
        return (self.num_filters, num_input_channels) + self.filter_size

    def get_output_shape_for(self, input_shape):
        pad = self.pad if isinstance(self.pad, tuple) else (self.pad,) * self.n
        batchsize = input_shape[0]
        return ((batchsize, self.num_filters) + 
                tuple(conv_output_length(input, filter, stride, p)
                      for input, filter, stride, p
                      in zip(input_shape[2:], self.filter_size,
                             self.stride, pad)))

    def get_output_for(self, input, **kwargs):
        conved = self.convolve(input, **kwargs)

        if self.b is None:
            activation = conved
        elif self.untie_biases:
            activation = conved + T.shape_padleft(self.b, 1)
        else:
            activation = conved + self.b.dimshuffle(('x', 0) + ('x',) * self.n)

        return self.nonlinearity(activation)

    def convolve(self, input, **kwargs):
        """
        Symbolically convolves `input` with ``self.W``, producing an output of
        shape ``self.output_shape``. To be implemented by subclasses.
        Parameters
        ----------
        input : Theano tensor
            The input minibatch to convolve
        **kwargs
            Any additional keyword arguments from :meth:`get_output_for`
        Returns
        -------
        Theano tensor
            `input` convolved according to the configuration of this layer,
            without any bias or nonlinearity applied.
        """
        raise NotImplementedError("BaseConvLayer does not implement the "
                                  "convolve() method. You will want to "
                                  "use a subclass such as Conv2DLayer.")



class Conv2DLayer(BaseConvLayer):
    """
    lasagne.layers.Conv2DLayer(incoming, num_filters, filter_size,
    stride=(1, 1), pad=0, untie_biases=False,
    W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
    nonlinearity=lasagne.nonlinearities.rectify, flip_filters=True,
    convolution=theano.tensor.nnet.conv2d, **kwargs)
    2D convolutional layer
    Performs a 2D convolution on its input and optionally adds a bias and
    applies an elementwise nonlinearity.
    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape. The
        output of this layer should be a 4D tensor, with shape
        ``(batch_size, num_input_channels, input_rows, input_columns)``.
    num_filters : int
        The number of learnable convolutional filters this layer has.
    filter_size : int or iterable of int
        An integer or a 2-element tuple specifying the size of the filters.
    stride : int or iterable of int
        An integer or a 2-element tuple specifying the stride of the
        convolution operation.
    pad : int, iterable of int, 'full', 'same' or 'valid' (default: 0)
        By default, the convolution is only computed where the input and the
        filter fully overlap (a valid convolution). When ``stride=1``, this
        yields an output that is smaller than the input by ``filter_size - 1``.
        The `pad` argument allows you to implicitly pad the input with zeros,
        extending the output size.
        A single integer results in symmetric zero-padding of the given size on
        all borders, a tuple of two integers allows different symmetric padding
        per dimension.
        ``'full'`` pads with one less than the filter size on both sides. This
        is equivalent to computing the convolution wherever the input and the
        filter overlap by at least one position.
        ``'same'`` pads with half the filter size (rounded down) on both sides.
        When ``stride=1`` this results in an output size equal to the input
        size. Even filter size is not supported.
        ``'valid'`` is an alias for ``0`` (no padding / a valid convolution).
        Note that ``'full'`` and ``'same'`` can be faster than equivalent
        integer values due to optimizations by Theano.
    untie_biases : bool (default: False)
        If ``False``, the layer will have a bias parameter for each channel,
        which is shared across all positions in this channel. As a result, the
        `b` attribute will be a vector (1D).
        If True, the layer will have separate bias parameters for each
        position in each channel. As a result, the `b` attribute will be a
        3D tensor.
    W : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the weights.
        These should be a 4D tensor with shape
        ``(num_filters, num_input_channels, filter_rows, filter_columns)``.
        See :func:`lasagne.utils.create_param` for more information.
    b : Theano shared variable, expression, numpy array, callable or ``None``
        Initial value, expression or initializer for the biases. If set to
        ``None``, the layer will have no biases. Otherwise, biases should be
        a 1D array with shape ``(num_filters,)`` if `untied_biases` is set to
        ``False``. If it is set to ``True``, its shape should be
        ``(num_filters, output_rows, output_columns)`` instead.
        See :func:`lasagne.utils.create_param` for more information.
    nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations. If None
        is provided, the layer will be linear.
    flip_filters : bool (default: True)
        Whether to flip the filters before sliding them over the input,
        performing a convolution (this is the default), or not to flip them and
        perform a correlation. Note that for some other convolutional layers in
        Lasagne, flipping incurs an overhead and is disabled by default --
        check the documentation when using learned weights from another layer.
    convolution : callable
        The convolution implementation to use. Usually it should be fine to
        leave this at the default value.
    **kwargs
        Any additional keyword arguments are passed to the `Layer` superclass.
    Attributes
    ----------
    W : Theano shared variable or expression
        Variable or expression representing the filter weights.
    b : Theano shared variable or expression
        Variable or expression representing the biases.
    """
    def __init__(self, incoming, num_filters, filter_size, stride=(1, 1),
                 pad=0, untie_biases=False,
                 W=GlorotUniform(), b=Constant(0.),
                 nonlinearity=rectify, flip_filters=True,
                 convolution=T.nnet.conv2d, **kwargs):
        super(Conv2DLayer, self).__init__(incoming, num_filters, filter_size,
                                          stride, pad, untie_biases, W, b,
                                          nonlinearity, flip_filters, n=2,
                                          **kwargs)
        self.convolution = convolution

    def convolve(self, input, **kwargs):
        border_mode = 'half' if self.pad == 'same' else self.pad
        print border_mode
        border_mode = 'valid'
        conved = self.convolution(input, self.W,
                                  image_shape=self.input_shape, filter_shape=self.get_W_shape(),
                                  subsample=self.stride,
                                  border_mode=border_mode)#filter_flip=self.flip_filters)
        print border_mode
        print conved.ndim
                                 
        return conved

class DropoutLayer(Layer):
    """Dropout layer
    Sets values to zero with probability p. See notes for disabling dropout
    during testing.
    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        the layer feeding into this layer, or the expected input shape
    p : float or scalar tensor
        The probability of setting a value to zero
    rescale : bool
        If true the input is rescaled with input / (1-p) when deterministic
        is False.
    Notes
    -----
    The dropout layer is a regularizer that randomly sets input values to
    zero; see [1]_, [2]_ for why this might improve generalization.
    During training you should set deterministic to false and during
    testing you should set deterministic to true.
    If rescale is true the input is scaled with input / (1-p) when
    deterministic is false, see references for further discussion. Note that
    this implementation scales the input at training time.
    References
    ----------
    .. [1] Hinton, G., Srivastava, N., Krizhevsky, A., Sutskever, I.,
           Salakhutdinov, R. R. (2012):
           Improving neural networks by preventing co-adaptation of feature
           detectors. arXiv preprint arXiv:1207.0580.
    .. [2] Srivastava Nitish, Hinton, G., Krizhevsky, A., Sutskever,
           I., & Salakhutdinov, R. R. (2014):
           Dropout: A Simple Way to Prevent Neural Networks from Overfitting.
           Journal of Machine Learning Research, 5(Jun)(2), 1929-1958.
    """
    def __init__(self, incoming, p=0.5, rescale=True, **kwargs):
        super(DropoutLayer, self).__init__(incoming, **kwargs)
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))
        self.p = p
        self.rescale = rescale

    def get_output_for(self, input, deterministic=False, **kwargs):
        """
        Parameters
        ----------
        input : tensor
            output from the previous layer
        deterministic : bool
            If true dropout and scaling is disabled, see notes
        """
        if deterministic or self.p == 0:
            return input
        else:
            retain_prob = 1 - self.p
            if self.rescale:
                input /= retain_prob

            # use nonsymbolic shape for dropout mask if possible
            input_shape = self.input_shape
            if any(s is None for s in input_shape):
                input_shape = input.shape

            return input * self._srng.binomial(input_shape, p=retain_prob,
                                               dtype=input.dtype)

dropout = DropoutLayer  # shortcut


def categorical_crossentropy(predictions, targets):
    """Computes the categorical cross-entropy between predictions and targets.
    .. math:: L_i = - \\sum_j{t_{i,j} \\log(p_{i,j})}
    Parameters
    ----------
    predictions : Theano 2D tensor
        Predictions in (0, 1), such as softmax output of a neural network,
        with data points in rows and class probabilities in columns.
    targets : Theano 2D tensor or 1D tensor
        Either targets in [0, 1] matching the layout of `predictions`, or
        a vector of int giving the correct class index per data point.
    Returns
    -------
    Theano 1D tensor
        An expression for the item-wise categorical cross-entropy.
    Notes
    -----
    This is the loss function of choice for multi-class classification
    problems and softmax output units. For hard targets, i.e., targets
    that assign all of the probability to a single class per data point,
    providing a vector of int for the targets is usually slightly more
    efficient than providing a matrix with a single 1.0 per row.
    """
    return theano.tensor.nnet.categorical_crossentropy(predictions, targets)


def get_or_compute_grads(loss_or_grads, params):
    """Helper function returning a list of gradients
    Parameters
    ----------
    loss_or_grads : symbolic expression or list of expressions
        A scalar loss expression, or a list of gradient expressions
    params : list of shared variables
        The variables to return the gradients for
    Returns
    -------
    list of expressions
        If `loss_or_grads` is a list, it is assumed to be a list of
        gradients and returned as is, unless it does not match the length
        of `params`, in which case a `ValueError` is raised.
        Otherwise, `loss_or_grads` is assumed to be a cost expression and
        the function returns `theano.grad(loss_or_grads, params)`.
    Raises
    ------
    ValueError
        If `loss_or_grads` is a list of a different length than `params`, or if
        any element of `params` is not a shared variable (while we could still
        compute its gradient, we can never update it and want to fail early).
    """
    if any(not isinstance(p, theano.compile.SharedVariable) for p in params):
        raise ValueError("params must contain shared variables only. If it "
                         "contains arbitrary parameter expressions, then "
                         "lasagne.utils.collect_shared_vars() may help you.")
    if isinstance(loss_or_grads, list):
        if not len(loss_or_grads) == len(params):
            raise ValueError("Got %d gradient expressions for %d parameters" % 
                             (len(loss_or_grads), len(params)))
        return loss_or_grads
    else:
        return theano.grad(loss_or_grads, params)
    
def sgd(loss_or_grads, params, learning_rate):
    """Stochastic Gradient Descent (SGD) updates
    Generates update expressions of the form:
    * ``param := param - learning_rate * gradient``
    Parameters
    ----------
    loss_or_grads : symbolic expression or list of expressions
        A scalar loss expression, or a list of gradient expressions
    params : list of shared variables
        The variables to generate update expressions for
    learning_rate : float or symbolic scalar
        The learning rate controlling the size of update steps
    Returns
    -------
    OrderedDict
        A dictionary mapping each parameter to its update expression
    """
    grads = get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()

    for param, grad in zip(params, grads):
        updates[param] = param - learning_rate * grad

    return updates

class Conv2DAllColsLayer(Conv2DLayer):
    """Convolutional layer always convolving over the full height
    of the layer before. See Conv2DLayer of lasagne for arguments.
    """
    def __init__(self, incoming, num_filters, filter_size, stride=(1, 1),
                 pad=0, untie_biases=False,
                 W=GlorotUniform(), b=Constant(0.),
                 nonlinearity=rectify,
                 convolution=T.nnet.conv2d, **kwargs):
        input_shape = incoming.output_shape
        assert filter_size[1] == -1, ("Please specify second dimension as -1"
            " , this dimension wil be replaced by number of cols of input shape")
        filter_size = [filter_size[0], input_shape[3]]
        super(Conv2DAllColsLayer, self).__init__(incoming, num_filters,
            filter_size, stride=stride,
             pad=pad, untie_biases=untie_biases,
             W=W, b=b, nonlinearity=nonlinearity,
             convolution=convolution, **kwargs)
        

def get_all_layers(layer, treat_as_input=None):
    """
    This function gathers all layers below one or more given :class:`Layer`
    instances, including the given layer(s). Its main use is to collect all
    layers of a network just given the output layer(s). The layers are
    guaranteed to be returned in a topological order: a layer in the result
    list is always preceded by all layers its input depends on.
    Parameters
    ----------
    layer : Layer or list
        the :class:`Layer` instance for which to gather all layers feeding
        into it, or a list of :class:`Layer` instances.
    treat_as_input : None or iterable
        an iterable of :class:`Layer` instances to treat as input layers
        with no layers feeding into them. They will show up in the result
        list, but their incoming layers will not be collected (unless they
        are required for other layers as well).
    Returns
    -------
    list
        a list of :class:`Layer` instances feeding into the given
        instance(s) either directly or indirectly, and the given
        instance(s) themselves, in topological order.
    Examples
    --------
    >>> from lasagne.layers import InputLayer, DenseLayer
    >>> l_in = InputLayer((100, 20))
    >>> l1 = DenseLayer(l_in, num_units=50)
    >>> get_all_layers(l1) == [l_in, l1]
    True
    >>> l2 = DenseLayer(l_in, num_units=10)
    >>> get_all_layers([l2, l1]) == [l_in, l2, l1]
    True
    >>> get_all_layers([l1, l2]) == [l_in, l1, l2]
    True
    >>> l3 = DenseLayer(l2, num_units=20)
    >>> get_all_layers(l3) == [l_in, l2, l3]
    True
    >>> get_all_layers(l3, treat_as_input=[l2]) == [l2, l3]
    True
    """
    # We perform a depth-first search. We add a layer to the result list only
    # after adding all its incoming layers (if any) or when detecting a cycle.
    # We use a LIFO stack to avoid ever running into recursion depth limits.
    try:
        queue = deque(layer)
    except TypeError:
        queue = deque([layer])
    seen = set()
    done = set()
    result = []

    # If treat_as_input is given, we pretend we've already collected all their
    # incoming layers.
    if treat_as_input is not None:
        seen.update(treat_as_input)

    while queue:
        # Peek at the leftmost node in the queue.
        layer = queue[0]
        if layer is None:
            # Some node had an input_layer set to `None`. Just ignore it.
            queue.popleft()
        elif layer not in seen:
            # We haven't seen this node yet: Mark it and queue all incomings
            # to be processed first. If there are no incomings, the node will
            # be appended to the result list in the next iteration.
            seen.add(layer)
            if hasattr(layer, 'input_layers'):
                queue.extendleft(reversed(layer.input_layers))
            elif hasattr(layer, 'input_layer'):
                queue.appendleft(layer.input_layer)
        else:
            # We've been here before: Either we've finished all its incomings,
            # or we've detected a cycle. In both cases, we remove the layer
            # from the queue and append it to the result list.
            queue.popleft()
            if layer not in done:
                result.append(layer)
                done.add(layer)

    return result



def reshape_for_stride_theano(topo_var, topo_shape, n_stride,
        invalid_fill_value=0):
    assert topo_shape[3] == 1, ("Not tested for nonempty third dim, "
        "might work though")
    # Create a different
    # out tensor for each offset from 0 to stride (exclusive),
    # e.g. 0,1,2 for stride 3
    # Then concatenate them together again.
    # From 4 different variants (this, using scan, using output preallocation 
    # + set_subtensor, using scan + output preallocation + set_subtensor)
    # this was the fastest, but only by a few percent
    
    n_third_dim = int(np.ceil(topo_shape[2] / float(n_stride)))
    reshaped_out = []
    if topo_shape[0] is not None:
        zero_dim_len = topo_shape[0]
    else:
        zero_dim_len = topo_var.shape[0]
    reshape_shape = (zero_dim_len, topo_shape[1], n_third_dim, topo_shape[3])
    for i_stride in xrange(n_stride):
        reshaped_this = T.ones(reshape_shape, dtype=np.float32) * invalid_fill_value
        i_length = int(np.ceil((topo_shape[2] - i_stride) / float(n_stride)))
        reshaped_this = T.set_subtensor(reshaped_this[:, :, :i_length],
            topo_var[:, :, i_stride::n_stride])
        reshaped_out.append(reshaped_this)
    reshaped_out = T.concatenate(reshaped_out)
    return reshaped_out

def get_output_shape_after_stride(input_shape, n_stride):
    time_length_after = int(np.ceil(input_shape[2] / float(n_stride)))
    if input_shape[0] is None:
        trials_after = None
    else:
        trials_after = int(input_shape[0] * n_stride)
        
    output_shape = (trials_after, input_shape[1], time_length_after, 1)
    return output_shape

class StrideReshapeLayer(Layer):
    def __init__(self, incoming, n_stride, invalid_fill_value=0, **kwargs):
        self.n_stride = n_stride
        self.invalid_fill_value = invalid_fill_value
        super(StrideReshapeLayer, self).__init__(incoming, **kwargs)

    def get_output_for(self, input, **kwargs):
        return reshape_for_stride_theano(input, self.input_shape, self.n_stride,
            invalid_fill_value=self.invalid_fill_value)

    def get_output_shape_for(self, input_shape):
        assert input_shape[3] == 1, "Not tested for nonempty last dim"
        return get_output_shape_after_stride(input_shape, self.n_stride)
    
class FinalReshapeLayer(Layer):
    def __init__(self, incoming, remove_invalids=True, flatten=True,
             **kwargs):
        self.remove_invalids = remove_invalids
        self.flatten = flatten
        super(FinalReshapeLayer, self).__init__(incoming, **kwargs)

    def get_output_for(self, input, input_var=None, **kwargs):
        # need input_var to determine number of trials
        # cannot use input var of entire net since maybe you want to
        # get output given an activation for a later layer...
        #
        # before we have sth like this (example where there was only a stride 2
        # in the computations before, and input length just 5)
        # showing with 1-based indexing here, sorry ;)
        # trial 1 sample 1, trial 1 sample 3, trial 1 sample 5
        # trial 2 sample 1, trial 2 sample 3, trial 2 sample 5
        # trial 1 sample 2, trial 1 sample 4, trial 1 NaN/invalid
        # trial 2 sample 2, trial 2 sample 4, trial 2 NaN/invalid
        # and this matrix for each filter/class... so if we transpose this matrix for
        # each filter, we get 
        # trial 1 sample 1, trial 2 sample 1, trial 1 sample 2, trial 2 sample 2
        # trial 1 sample 2, ...
        # ...
        # after flattening past the filter dim we then have
        # trial 1 sample 1, trial 2 sample1, ..., trial 1 sample 2, trial 2 sample 2
        # which is our output shape which allows to remove invalids easily:
        # (sample 1 for all trials), (sample 2 for all trials), etc
         
        # After removal of invalids,
        #  we reshape again to (trial 1 all samples), (trial 2 all samples)
        
        # Reshape/flatten into #predsamples x #classes
        n_classes = self.input_shape[1]
        input = input.dimshuffle(1, 2, 0, 3).reshape((n_classes, -1)).T
        if input_var is None:
            input_var = get_all_layers(self)[0].input_var
        input_shape = get_all_layers(self)[0].shape
        if input_shape[0] is not None:
            trials = input_shape[0]
        else:
            trials = input_var.shape[0]
            
        if self.remove_invalids:
            # remove invalid values (possibly nans still contained before)
            n_sample_preds = get_n_sample_preds(self)
                
            input = input[:trials * n_sample_preds]
        
        # reshape to (trialsxsamples) again, i.e.
        # (trial1 all samples), (trial 2 all samples), ...
        # By doing this:
        # transpose to classes x (samplepreds*trials)
        # then reshape to classes x sample preds x trials, 
        # dimshuffle to classes x trials x sample preds to flatten again to
        # final output:
        # (trial 1 all samples), (trial 2 all samples), ...
        
        # if not flatten, instead reshape to:
        #  trials x classes/filters x sample preds x emptydim
        input = input.T.reshape((self.input_shape[1],
            - 1, trials))
        if self.flatten:
            input = input.dimshuffle(0, 2, 1).reshape((n_classes, -1)).T
        else:
            input = input.dimshuffle(2, 0, 1, 'x')
            
        
        return input
        
    def get_output_shape_for(self, input_shape):
        assert input_shape[3] == 1, ("Not tested and thought about " 
            "for nonempty last dim, likely not to work")
        return [None, input_shape[1]]
    
def get_3rd_dim_shapes_without_invalids(layer):
    all_layers = get_all_layers(layer)
    return get_3rd_dim_shapes_without_invalids_for_layers(all_layers)

def get_3rd_dim_shapes_without_invalids_for_layers(all_layers):
    # handle as special case that first layer is dim shuffle layer that shuffles last 2 dims.. 
    # hack for the stupiding kaggle code
    if hasattr(all_layers[1], 'pattern') and all_layers[1].pattern == (0, 1, 3, 2):
        all_layers = all_layers[1:]

    cur_lengths = np.array([all_layers[0].output_shape[2]])
    # todelay: maybe redo this by using get_output_shape_for function?
    for l in all_layers:
        if hasattr(l, 'filter_size'):
            if l.pad == (0, 0):
                cur_lengths = cur_lengths - l.filter_size[0] + 1
            elif l.pad == ((l.filter_size[0] - 1) / 2, (l.filter_size[1] - 1) / 2):
                cur_lengths = cur_lengths
            else:
                print l
                raise ValueError("Not implemented this padding:", l.pad, l, l.filter_size)
                
        if hasattr(l, 'pool_size'):
            cur_lengths = cur_lengths - l.pool_size[0] + 1
        if hasattr(l, 'n_stride'):
            # maybe it should be floor not ceil?
            cur_lengths = np.array([int(np.ceil((length - i_stride) / 
                                               float(l.n_stride)))
                for length in cur_lengths for i_stride in range(l.n_stride)])
    return cur_lengths

def get_n_sample_preds(layer):
    paths = get_all_paths(layer)
    preds_per_path = [np.sum(get_3rd_dim_shapes_without_invalids_for_layers(
        layers)) for layers in paths]
    # all path should have same length
    assert len(np.unique(preds_per_path)) == 1, ("All paths, should have same "
        "lengths, pathlengths are" + str(preds_per_path))
    return preds_per_path[0]


def get_input_time_length(layer):
    return get_all_layers(layer)[0].shape[2]

def get_model_input_window(cnt_model):
    return get_input_time_length(cnt_model) - get_n_sample_preds(cnt_model) + 1


def get_all_paths(layer, treat_as_input=None):
    """
    This function gathers all paths through the net ending at the given final layer.
    ----------
    layer : Layer or list
        the :class:`Layer` instance for which to gather all layers feeding
        into it, or a list of :class:`Layer` instances.
    treat_as_input : None or iterable
        an iterable of :class:`Layer` instances to treat as input layers
        with no layers feeding into them. They will show up in the result
        list, but their incoming layers will not be collected (unless they
        are required for other layers as well).
    Returns
    -------
    list of list
        a list of lists of :class:`Layer` instances feeding into the given
        instance(s) either directly or indirectly, and the given
        instance(s) themselves, in topological order.
    """
    # We perform a depth-first search. We add a layer to the result list only
    # after adding all its incoming layers (if any) or when detecting a cycle.
    # We use a LIFO stack to avoid ever running into recursion depth limits.
    try:
        queue = deque(layer)
    except TypeError:
        queue = deque([layer])
    seen = set()
    done = set()

    # If treat_as_input is given, we pretend we've already collected all their
    # incoming layers.
    if treat_as_input is not None:
        seen.update(treat_as_input)

    paths_queue = deque()
    paths_queue.appendleft((queue, seen, done))
    all_paths = []
    while paths_queue:
        result = []
        queue, seen, done = paths_queue.pop()
        while queue:
            # Peek at the leftmost node in the queue.
            layer = queue[0]
            if layer is None:
                # Some node had an input_layer set to `None`. Just ignore it.
                queue.popleft()
            elif layer not in seen:
                # We haven't seen this node yet: Mark it and queue all incomings
                # to be processed first. If there are no incomings, the node will
                # be appended to the result list in the next iteration.
                seen.add(layer)
                if hasattr(layer, 'input_layers'):
                    for input_layer in layer.input_layers:
                        # Create a new queue for each input layer
                        # they will be used outside of that path
                        this_queue = copy(queue)
                        this_queue.appendleft(input_layer)
                        this_path_parts = (this_queue, copy(seen), copy(done))
                        paths_queue.appendleft(this_path_parts)

                    queue, seen, done = paths_queue.pop()

                elif hasattr(layer, 'input_layer'):
                    queue.appendleft(layer.input_layer)
            else:
                # We've been here before: Either we've finished all its incomings,
                # or we've detected a cycle. In both cases, we remove the layer
                # from the queue and append it to the result list.
                queue.popleft()
                if layer not in done:
                    result.append(layer)
                    done.add(layer)
        all_paths.append(result)


    return all_paths

def pool_output_length(input_length, pool_size, stride, pad, ignore_border):
    """
    Compute the output length of a pooling operator
    along a single dimension.
    Parameters
    ----------
    input_length : integer
        The length of the input in the pooling dimension
    pool_size : integer
        The length of the pooling region
    stride : integer
        The stride between successive pooling regions
    pad : integer
        The number of elements to be added to the input on each side.
    ignore_border: bool
        If ``True``, partial pooling regions will be ignored.
        Must be ``True`` if ``pad != 0``.
    Returns
    -------
    output_length
        * None if either input is None.
        * Computed length of the pooling operator otherwise.
    Notes
    -----
    When ``ignore_border == True``, this is given by the number of full
    pooling regions that fit in the padded input length,
    divided by the stride (rounding down).
    If ``ignore_border == False``, a single partial pooling region is
    appended if at least one input element would be left uncovered otherwise.
    """
    if input_length is None or pool_size is None:
        return None

    if ignore_border:
        output_length = input_length + 2 * pad - pool_size + 1
        output_length = (output_length + stride - 1) // stride

    # output length calculation taken from:
    # https://github.com/Theano/Theano/blob/master/theano/tensor/signal/downsample.py
    else:
        assert pad == 0

        if stride >= pool_size:
            output_length = (input_length + stride - 1) // stride
        else:
            output_length = max(
                0, (input_length - pool_size + stride - 1) // stride) + 1

    return output_length

class Pool2DLayer(Layer):
    """
    2D pooling layer
    Performs 2D mean or max-pooling over the two trailing axes
    of a 4D input tensor.
    Parameters
    ----------
    incoming : a :class:`Layer` instance or tuple
        The layer feeding into this layer, or the expected input shape.
    pool_size : integer or iterable
        The length of the pooling region in each dimension.  If an integer, it
        is promoted to a square pooling region. If an iterable, it should have
        two elements.
    stride : integer, iterable or ``None``
        The strides between sucessive pooling regions in each dimension.
        If ``None`` then ``stride = pool_size``.
    pad : integer or iterable
        Number of elements to be added on each side of the input
        in each dimension. Each value must be less than
        the corresponding stride.
    ignore_border : bool
        If ``True``, partial pooling regions will be ignored.
        Must be ``True`` if ``pad != (0, 0)``.
    mode : {'max', 'average_inc_pad', 'average_exc_pad'}
        Pooling mode: max-pooling or mean-pooling including/excluding zeros
        from partially padded pooling regions. Default is 'max'.
    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.
    See Also
    --------
    MaxPool2DLayer : Shortcut for max pooling layer.
    Notes
    -----
    The value used to pad the input is chosen to be less than
    the minimum of the input, so that the output of each pooling region
    always corresponds to some element in the unpadded input region.
    Using ``ignore_border=False`` prevents Theano from using cuDNN for the
    operation, so it will fall back to a slower implementation.
    """

    def __init__(self, incoming, pool_size, stride=None, pad=(0, 0),
                 ignore_border=True, mode='max', **kwargs):
        super(Pool2DLayer, self).__init__(incoming, **kwargs)

        self.pool_size = as_tuple(pool_size, 2)

        if len(self.input_shape) != 4:
            raise ValueError("Tried to create a 2D pooling layer with "
                             "input shape %r. Expected 4 input dimensions "
                             "(batchsize, channels, 2 spatial dimensions)."
                             % (self.input_shape,))

        if stride is None:
            self.stride = self.pool_size
        else:
            self.stride = as_tuple(stride, 2)

        self.pad = as_tuple(pad, 2)

        self.ignore_border = ignore_border
        self.mode = mode

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)  # copy / convert to mutable list

        output_shape[2] = pool_output_length(input_shape[2],
                                             pool_size=self.pool_size[0],
                                             stride=self.stride[0],
                                             pad=self.pad[0],
                                             ignore_border=self.ignore_border,
                                             )

        output_shape[3] = pool_output_length(input_shape[3],
                                             pool_size=self.pool_size[1],
                                             stride=self.stride[1],
                                             pad=self.pad[1],
                                             ignore_border=self.ignore_border,
                                             )

        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        pooled = downsample.max_pool_2d(input,
                                        ds=self.pool_size,
                                        st=self.stride,
                                        ignore_border=self.ignore_border,
                                        padding=self.pad,
                                        mode=self.mode,
                                        )
        return pooled


class SumPool2dLayer(Pool2DLayer):
    def get_output_for(self, input, **kwargs):
        pooled = downsample.max_pool_2d(input,
                                        ds=self.pool_size,
                                        st=self.stride,
                                        ignore_border=self.ignore_border,
                                        padding=self.pad,
                                        mode=self.mode,
                                        )
        # cast size to float32 to prevent upcast to float64 of entire data
        return pooled * np.float32(np.prod(self.pool_size))


def safe_log(x, eps=1e-6):
    """ Prevents log(0) by using max of eps and given x."""
    return  T.log(T.maximum(x, eps))
   
def get_balanced_batches(n_trials, batch_size, rng, shuffle):
    n_batches = n_trials // batch_size
    if n_batches > 0:
        min_batch_size = n_trials // n_batches
        n_batches_with_extra_trial = n_trials % n_batches
    else:
        n_batches = 1
        min_batch_size = n_trials
        n_batches_with_extra_trial = 0
    assert n_batches_with_extra_trial < n_batches
    all_inds = np.array(range(n_trials))
    if shuffle:
        rng.shuffle(all_inds)
    i_trial = 0
    end_trial = 0
    batches = []
    for i_batch in xrange(n_batches):
        end_trial += min_batch_size
        if i_batch < n_batches_with_extra_trial:
            end_trial += 1
        batch_inds = all_inds[range(i_trial, end_trial)]
        batches.append(batch_inds)
        i_trial = end_trial
    assert i_trial == n_trials
    return batches

class CntWindowsFromCntIterator(object):
    def __init__(self, batch_size, input_time_length, n_sample_preds,
            oversample_targets=False, remove_baseline_mean=False):
        self.batch_size = batch_size
        self.input_time_length = input_time_length
        self.n_sample_preds = n_sample_preds
        self.oversample_targets = oversample_targets
        self.remove_baseline_mean = remove_baseline_mean
        self.rng = RandomState(328774)
    
    def get_batches(self, dataset, shuffle):
        n_samples = dataset.get_topological_view().shape[0]
        n_lost_samples = self.input_time_length - self.n_sample_preds
        # Create blocks with start and end sample for entire dataset
        start_end_blocks = []
        last_input_sample = n_samples - self.input_time_length
        
        # To get predictions for all samples, also block at the start that
        # account for the lost samples at the start
        n_sample_start = -n_lost_samples
        n_sample_stop = last_input_sample + self.n_sample_preds
        for i_start_sample in range(n_sample_start, n_sample_stop,
                self.n_sample_preds):
            i_adjusted_start = min(i_start_sample, n_samples - self.input_time_length)
            start_end_blocks.append((i_adjusted_start, i_adjusted_start + self.input_time_length))
        
        
        if shuffle and self.oversample_targets:
            # Hacky heuristic for oversampling...
            # duplicate those blocks that contain
            # more targets than the mean per block
            # if they contian 2 times as much as mean,
            # put them in 2 times, 3 times as much,
            # put in 3 times, etc.
            n_targets_in_block = []
            for start, end in start_end_blocks:
                n_targets_in_block.append(np.sum(dataset.y[start + n_lost_samples:end]))
            mean_targets_in_block = np.mean(n_targets_in_block)
            for i_block in xrange(len(start_end_blocks)):
                target_ratio = int(np.round(n_targets_in_block[i_block] / 
                    float(mean_targets_in_block)))
                if target_ratio > 1:
                    for _ in xrange(target_ratio - 1):
                        start_end_blocks.append(start_end_blocks[i_block])
            
        block_ind_batches = get_balanced_batches(len(start_end_blocks),
            batch_size=self.batch_size, rng=self.rng, shuffle=shuffle)
       
        topo = dataset.get_topological_view()
        for block_inds in block_ind_batches:
            batch_size = len(block_inds)
            # have to wrap into float32, cause np.nan upcasts to float64!
            batch_topo = np.float32(np.ones((batch_size, topo.shape[1],
                 self.input_time_length, topo.shape[3])) * np.nan)
            batch_y = np.ones((self.n_sample_preds * batch_size, dataset.y.shape[1])) * np.nan
            for i_batch_block, i_block in enumerate(block_inds):
                start, end = start_end_blocks[i_block]
                # switch samples into last axis, (dim 2 shd be empty before)
                assert topo.shape[2] == 1
                # check if start is negative and end positive
                # could happen from padding blocks at start
                if start >= 0 or (start < 0 and end < 0):
                    batch_topo[i_batch_block] = topo[start:end].swapaxes(0, 2)
                else:
                    assert start < 0 and end >= 0
                    # do wrap around padding
                    batch_topo[i_batch_block] = np.concatenate((topo[start:],
                        topo[:end])).swapaxes(0, 2)
                assert start + n_lost_samples >= 0, ("Wrapping should only "
                    "account for lost samples at start and never lead "
                    "to negative y inds")
                batch_start_y = self.n_sample_preds * i_batch_block
                batch_end_y = batch_start_y + self.n_sample_preds
                batch_y[batch_start_y:batch_end_y] = (
                    dataset.y[start + n_lost_samples:end])
    
            if self.remove_baseline_mean:
                batch_topo -= np.mean(
                    # should produce mean per batchxchan
                    batch_topo[:, :, :n_lost_samples, :], axis=(2, 3),
                    keepdims=True)
                
            assert not np.any(np.isnan(batch_topo))
            assert not np.any(np.isnan(batch_y))
            batch_y = batch_y.astype(np.int32)
            yield batch_topo, batch_y 

    def reset_rng(self):
        self.rng = RandomState(328774)
        

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


def roc_auc_score(y_true, y_score, average="macro", sample_weight=None):
    """Compute Area Under the Curve (AUC) from prediction scores
    Note: this implementation is restricted to the binary classification task
    or multilabel classification task in label indicator format.
    Read more in the :ref:`User Guide <roc_metrics>`.
    Parameters
    ----------
    y_true : array, shape = [n_samples] or [n_samples, n_classes]
        True binary labels in binary label indicators.
    y_score : array, shape = [n_samples] or [n_samples, n_classes]
        Target scores, can either be probability estimates of the positive
        class, confidence values, or binary decisions.
    average : string, [None, 'micro', 'macro' (default), 'samples', 'weighted']
        If ``None``, the scores for each class are returned. Otherwise,
        this determines the type of averaging performed on the data:
        ``'micro'``:
            Calculate metrics globally by considering each element of the label
            indicator matrix as a label.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label).
        ``'samples'``:
            Calculate metrics for each instance, and find their average.
    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.
    Returns
    -------
    auc : float
    References
    ----------
    .. [1] `Wikipedia entry for the Receiver operating characteristic
            <http://en.wikipedia.org/wiki/Receiver_operating_characteristic>`_
    See also
    --------
    average_precision_score : Area under the precision-recall curve
    roc_curve : Compute Receiver operating characteristic (ROC)
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import roc_auc_score
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    >>> roc_auc_score(y_true, y_scores)
    0.75
    """
    def _binary_roc_auc_score(y_true, y_score, sample_weight=None):
        if len(np.unique(y_true)) != 2:
            raise ValueError("Only one class present in y_true. ROC AUC score "
                             "is not defined in that case.")

        fpr, tpr, tresholds = roc_curve(y_true, y_score,
                                        sample_weight=sample_weight)
        return auc(fpr, tpr, reorder=True)

    return _average_binary_score(
        _binary_roc_auc_score, y_true, y_score, average,
        sample_weight=sample_weight)    

def _average_binary_score(binary_metric, y_true, y_score, average,
                          sample_weight=None):
    """Average a binary metric for multilabel classification
    Parameters
    ----------
    y_true : array, shape = [n_samples] or [n_samples, n_classes]
        True binary labels in binary label indicators.
    y_score : array, shape = [n_samples] or [n_samples, n_classes]
        Target scores, can either be probability estimates of the positive
        class, confidence values, or binary decisions.
    average : string, [None, 'micro', 'macro' (default), 'samples', 'weighted']
        If ``None``, the scores for each class are returned. Otherwise,
        this determines the type of averaging performed on the data:
        ``'micro'``:
            Calculate metrics globally by considering each element of the label
            indicator matrix as a label.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label).
        ``'samples'``:
            Calculate metrics for each instance, and find their average.
    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.
    binary_metric : callable, returns shape [n_classes]
        The binary metric function to use.
    Returns
    -------
    score : float or array of shape [n_classes]
        If not ``None``, average the score, else return the score for each
        classes.
    """
    average_options = (None, 'micro', 'macro', 'weighted', 'samples')
    if average not in average_options:
        raise ValueError('average has to be one of {0}'
                         ''.format(average_options))

    return binary_metric(y_true, y_score, sample_weight=sample_weight)


    not_average_axis = 1
    score_weight = sample_weight
    average_weight = None

    if average == "micro":
        if score_weight is not None:
            score_weight = np.repeat(score_weight, y_true.shape[1])
        y_true = y_true.ravel()
        y_score = y_score.ravel()

    elif average == 'weighted':
        if score_weight is not None:
            average_weight = np.sum(np.multiply(
                y_true, np.reshape(score_weight, (-1, 1))), axis=0)
        else:
            average_weight = np.sum(y_true, axis=0)
        if average_weight.sum() == 0:
            return 0

    elif average == 'samples':
        # swap average_weight <-> score_weight
        average_weight = score_weight
        score_weight = None
        not_average_axis = 0

    if y_true.ndim == 1:
        y_true = y_true.reshape((-1, 1))

    if y_score.ndim == 1:
        y_score = y_score.reshape((-1, 1))

    n_classes = y_score.shape[not_average_axis]
    score = np.zeros((n_classes,))
    for c in range(n_classes):
        y_true_c = y_true.take([c], axis=not_average_axis).ravel()
        y_score_c = y_score.take([c], axis=not_average_axis).ravel()
        score[c] = binary_metric(y_true_c, y_score_c,
                                 sample_weight=score_weight)

    # Average the results
    if average is not None:
        return np.average(score, weights=average_weight)
    else:
        return score
    

def auc(x, y, reorder=False):
    """Compute Area Under the Curve (AUC) using the trapezoidal rule
    This is a general function, given points on a curve.  For computing the
    area under the ROC-curve, see :func:`roc_auc_score`.
    Parameters
    ----------
    x : array, shape = [n]
        x coordinates.
    y : array, shape = [n]
        y coordinates.
    reorder : boolean, optional (default=False)
        If True, assume that the curve is ascending in the case of ties, as for
        an ROC curve. If the curve is non-ascending, the result will be wrong.
    Returns
    -------
    auc : float
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn import metrics
    >>> y = np.array([1, 1, 2, 2])
    >>> pred = np.array([0.1, 0.4, 0.35, 0.8])
    >>> fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
    >>> metrics.auc(fpr, tpr)
    0.75
    See also
    --------
    roc_auc_score : Computes the area under the ROC curve
    precision_recall_curve :
        Compute precision-recall pairs for different probability thresholds
    """

    if x.shape[0] < 2:
        raise ValueError('At least 2 points are needed to compute'
                         ' area under curve, but x.shape = %s' % x.shape)

    direction = 1
    if reorder:
        # reorder the data points according to the x axis and using y to
        # break ties
        order = np.lexsort((y, x))
        x, y = x[order], y[order]
    else:
        dx = np.diff(x)
        if np.any(dx < 0):
            if np.all(dx <= 0):
                direction = -1
            else:
                raise ValueError("Reordering is not turned on, and "
                                 "the x array is not increasing: %s" % x)

    area = direction * np.trapz(y, x)

    return area
    
def roc_curve(y_true, y_score, pos_label=None, sample_weight=None,
              drop_intermediate=True):
    """Compute Receiver operating characteristic (ROC)
    Note: this implementation is restricted to the binary classification task.
    Read more in the :ref:`User Guide <roc_metrics>`.
    Parameters
    ----------
    y_true : array, shape = [n_samples]
        True binary labels in range {0, 1} or {-1, 1}.  If labels are not
        binary, pos_label should be explicitly given.
    y_score : array, shape = [n_samples]
        Target scores, can either be probability estimates of the positive
        class or confidence values.
    pos_label : int
        Label considered as positive and others are considered negative.
    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.
    drop_intermediate : boolean, optional (default=True)
        Whether to drop some suboptimal thresholds which would not appear
        on a plotted ROC curve. This is useful in order to create lighter
        ROC curves.
        .. versionadded:: 0.17
           parameter *drop_intermediate*.
    Returns
    -------
    fpr : array, shape = [>2]
        Increasing false positive rates such that element i is the false
        positive rate of predictions with score >= thresholds[i].
    tpr : array, shape = [>2]
        Increasing true positive rates such that element i is the true
        positive rate of predictions with score >= thresholds[i].
    thresholds : array, shape = [n_thresholds]
        Decreasing thresholds on the decision function used to compute
        fpr and tpr. `thresholds[0]` represents no instances being predicted
        and is arbitrarily set to `max(y_score) + 1`.
    See also
    --------
    roc_auc_score : Compute Area Under the Curve (AUC) from prediction scores
    Notes
    -----
    Since the thresholds are sorted from low to high values, they
    are reversed upon returning them to ensure they correspond to both ``fpr``
    and ``tpr``, which are sorted in reversed order during their calculation.
    References
    ----------
    .. [1] `Wikipedia entry for the Receiver operating characteristic
            <http://en.wikipedia.org/wiki/Receiver_operating_characteristic>`_
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn import metrics
    >>> y = np.array([1, 1, 2, 2])
    >>> scores = np.array([0.1, 0.4, 0.35, 0.8])
    >>> fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)
    >>> fpr
    array([ 0. ,  0.5,  0.5,  1. ])
    >>> tpr
    array([ 0.5,  0.5,  1. ,  1. ])
    >>> thresholds
    array([ 0.8 ,  0.4 ,  0.35,  0.1 ])
    """
    fps, tps, thresholds = _binary_clf_curve(
        y_true, y_score, pos_label=pos_label, sample_weight=sample_weight)

    # Attempt to drop thresholds corresponding to points in between and
    # collinear with other points. These are always suboptimal and do not
    # appear on a plotted ROC curve (and thus do not affect the AUC).
    # Here np.diff(_, 2) is used as a "second derivative" to tell if there
    # is a corner at the point. Both fps and tps must be tested to handle
    # thresholds with multiple data points (which are combined in
    # _binary_clf_curve). This keeps all cases where the point should be kept,
    # but does not drop more complicated cases like fps = [1, 3, 7],
    # tps = [1, 2, 4]; there is no harm in keeping too many thresholds.
    if drop_intermediate and len(fps) > 2:
        optimal_idxs = np.where(np.r_[True,
                                      np.logical_or(np.diff(fps, 2),
                                                    np.diff(tps, 2)),
                                      True])[0]
        fps = fps[optimal_idxs]
        tps = tps[optimal_idxs]
        thresholds = thresholds[optimal_idxs]

    if tps.size == 0 or fps[0] != 0:
        # Add an extra threshold position if necessary
        tps = np.r_[0, tps]
        fps = np.r_[0, fps]
        thresholds = np.r_[thresholds[0] + 1, thresholds]

    fpr = fps / fps[-1]
    tpr = tps / tps[-1]
    


    return fpr, tpr, thresholds



def _binary_clf_curve(y_true, y_score, pos_label=None, sample_weight=None):
    """Calculate true and false positives per binary classification threshold.
    Parameters
    ----------
    y_true : array, shape = [n_samples]
        True targets of binary classification
    y_score : array, shape = [n_samples]
        Estimated probabilities or decision function
    pos_label : int, optional (default=None)
        The label of the positive class
    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.
    Returns
    -------
    fps : array, shape = [n_thresholds]
        A count of false positives, at index i being the number of negative
        samples assigned a score >= thresholds[i]. The total number of
        negative samples is equal to fps[-1] (thus true negatives are given by
        fps[-1] - fps).
    tps : array, shape = [n_thresholds <= len(np.unique(y_score))]
        An increasing count of true positives, at index i being the number
        of positive samples assigned a score >= thresholds[i]. The total
        number of positive samples is equal to tps[-1] (thus false negatives
        are given by tps[-1] - tps).
    thresholds : array, shape = [n_thresholds]
        Decreasing score values.
    """
    # y_true = column_or_1d(y_true)
    # y_score = column_or_1d(y_score)

    # ensure binary classification if pos_label is not specified
    classes = np.unique(y_true)
    if pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    if sample_weight is not None:
        weight = sample_weight[desc_score_indices]
    else:
        weight = 1.

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    # We need to use isclose to avoid spurious repeated thresholds
    # stemming from floating point roundoff errors.
    distinct_value_indices = np.where(np.logical_not(np.isclose(
        np.diff(y_score), 0)))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = (y_true * weight).cumsum()[threshold_idxs]
    if sample_weight is not None:
        fps = weight.cumsum()[threshold_idxs] - tps
    else:
        fps = 1 + threshold_idxs - tps
    return fps, tps, y_score[threshold_idxs]


def safe_roc_auc_score(y_true, y_score):
    """Returns nan if targets only contain one class"""
    if len(np.unique(y_true)) == 1:
        return np.nan
    else:
        return roc_auc_score(y_true, y_score)
    
def auc_classes_mean(y, preds):
    # nanmean to ignore classes that are not present
    return np.nanmean([safe_roc_auc_score(
            np.int32(y[:, i] == 1), preds[:, i]) 
            for i in range(y.shape[1])])

class Monitor(object):
    __metaclass__ = ABCMeta
    @abstractmethod
    def setup(self, monitor_chans, datasets):
        raise NotImplementedError("Subclass needs to implement this")

    @abstractmethod
    def monitor_epoch(self, monitor_chans):
        raise NotImplementedError("Subclass needs to implement this")
    @abstractmethod
    def monitor_set(self, monitor_chans, setname, preds, losses,
            all_batch_sizes, targets, dataset):
        raise NotImplementedError("Subclass needs to implement this")

class MonitorManager(object):
    def __init__(self, monitors):
        self.monitors = monitors
        
    def create_theano_functions(self, input_var, target_var, predictions_var,
            loss_var):
        self.pred_loss_func = theano.function([input_var, target_var],
            [predictions_var, loss_var])
        
    def setup(self, monitor_chans, datasets):
        for monitor in self.monitors:
            monitor.setup(monitor_chans, datasets)
        
    def monitor_epoch(self, monitor_chans, datasets, iterator):
        # first call monitor epoch of all monitors, 
        # then monitor set with losses and preds
        # maybe change this method entirely if it turns out to be too
        # inflexible
        for m in self.monitors:
            m.monitor_epoch(monitor_chans)
        
        for setname in datasets:
            assert setname in ['train', 'valid', 'test']
            dataset = datasets[setname]
            all_preds = []
            all_losses = []
            batch_sizes = []
            targets = []
            for batch in iterator.get_batches(dataset, shuffle=False):
                preds, loss = self.pred_loss_func(batch[0], batch[1])
                all_preds.append(preds)
                all_losses.append(loss)
                batch_sizes.append(len(batch[0]))
                targets.append(batch[1])

            for m in self.monitors:
                m.monitor_set(monitor_chans, setname, all_preds, all_losses,
                    batch_sizes, targets, dataset)

class RuntimeMonitor(Monitor):
    def setup(self, monitor_chans, datasets):
        self.last_call_time = None
        monitor_chans['runtime'] = []

    def monitor_epoch(self, monitor_chans):
        cur_time = time.time()
        if self.last_call_time is None:
            # just in case of first call
            self.last_call_time = cur_time
        monitor_chans['runtime'].append(cur_time - self.last_call_time)
        self.last_call_time = cur_time

    def monitor_set(self, monitor_chans, setname, all_preds, losses,
            all_batch_sizes, targets, dataset):
        return


class LossMonitor(Monitor):
    def setup(self, monitor_chans, datasets):
        for setname in datasets:
            assert setname in ['train', 'valid', 'test']
            monitor_key = "{:s}_loss".format(setname)
            monitor_chans[monitor_key] = []

    def monitor_epoch(self, monitor_chans):
        return

    def monitor_set(self, monitor_chans, setname, preds, losses,
            all_batch_sizes, targets, dataset):
        total_loss = 0.0
        num_trials = 0
        for i_batch in range(len(all_batch_sizes)):
            batch_size = all_batch_sizes[i_batch]
            batch_loss = losses[i_batch]
            # at the end we want the mean over whole dataset
            # so weigh this mean (loss func arleady computes mean for batch)
            # by the size of the batch... this works also if batches
            # not all the same size
            num_trials += batch_size
            total_loss += (batch_loss * batch_size)
            
        mean_loss = total_loss / num_trials
        monitor_key = "{:s}_loss".format(setname)
        monitor_chans[monitor_key].append(float(mean_loss))
        
def get_reshaped_cnt_preds(all_preds, n_samples, input_time_length,
        n_sample_preds):
    """Taking predictions from a multiple prediction/parallel net
    and removing the last predictions (from last batch) which are duplicated.
    """
    all_preds = deepcopy(all_preds)
    # fix the last predictions, they are partly duplications since the last window
    # is made to fit into the timesignal 
    # sample preds
    # might not exactly fit into number of samples)
    legitimate_last_preds = n_samples % n_sample_preds
    if legitimate_last_preds != 0:  # in case = 0 there was no overlap, no need to do anything!
        fixed_last_preds = all_preds[-1][-legitimate_last_preds:]
        final_batch_size = all_preds[-1].shape[0] / n_sample_preds
        if final_batch_size > 1:
            # need to take valid sample preds from batches before
            samples_from_legit_batches = n_sample_preds * (final_batch_size - 1)
            fixed_last_preds = np.append(all_preds[-1][:samples_from_legit_batches],
                 fixed_last_preds, axis=0)
        all_preds[-1] = fixed_last_preds
    
    all_preds_arr = np.concatenate(all_preds)
    
    return all_preds_arr

class AUCMeanMisclassMonitor(Monitor):
    def __init__(self, input_time_length=None, n_sample_preds=None):
        self.input_time_length = input_time_length
        self.n_sample_preds = n_sample_preds
    
    def setup(self, monitor_chans, datasets):
        for setname in datasets:
            assert setname in ['train', 'valid', 'test']
            monitor_key = "{:s}_misclass".format(setname)
            monitor_chans[monitor_key] = []

    def monitor_epoch(self, monitor_chans):
        return

    def monitor_set(self, monitor_chans, setname, all_preds, losses,
            all_batch_sizes, targets, dataset):
        # remove last preds that were duplicates due to overlap of final windows
        n_samples = len(dataset.y)
        if self.input_time_length is not None:
            all_preds_arr = get_reshaped_cnt_preds(all_preds, n_samples,
                self.input_time_length, self.n_sample_preds)
        else:
            all_preds_arr = np.concatenate(all_preds)
        
        auc_mean = auc_classes_mean(dataset.y, all_preds_arr)
        misclass = 1 - auc_mean
        monitor_key = "{:s}_misclass".format(setname)
        monitor_chans[monitor_key].append(float(misclass))
            
class MaxEpochs(object):
    def  __init__(self, num_epochs):
        self.num_epochs = num_epochs
        
    def should_stop(self, monitor_chans):
        # -1 due to doing one monitor at start of training
        i_epoch = len(monitor_chans.values()[0]) - 1
        return i_epoch >= self.num_epochs
    
class DenseDesignMatrixWrapper(object):
    reloadable = False
    def __init__(self, topo_view, y, axes):
        self.topo = topo_view
        self.y = y
        self.view_converter = lambda x:None
        self.view_converter.axes = ('b', 'c', 0, 1)
        self.X = topo_view
        
    def ensure_is_loaded(self):
        pass
    
    def get_topological_view(self):
        return self.topo
    
class TrainValidTestSplitter(object):
    __metaclass__ = ABCMeta
    @abstractmethod
    def split_into_train_valid_test(self, dataset):
        raise NotImplementedError("Subclass needs to implement this")
    
def split_set_by_indices(dataset, train_fold, valid_fold, test_fold):
        n_trials = dataset.get_topological_view().shape[0]
        # Make sure there are no overlaps and we have all possible trials
        # assigned
        assert np.intersect1d(valid_fold, test_fold).size == 0
        assert np.intersect1d(train_fold, test_fold).size == 0
        assert np.intersect1d(train_fold, valid_fold).size == 0
        assert (set(np.concatenate((train_fold, valid_fold, test_fold))) == 
            set(range(n_trials)))
        
        train_set = DenseDesignMatrixWrapper(
            topo_view=dataset.get_topological_view()[train_fold],
            y=dataset.y[train_fold],
            axes=dataset.view_converter.axes)
        valid_set = DenseDesignMatrixWrapper(
            topo_view=dataset.get_topological_view()[valid_fold],
            y=dataset.y[valid_fold],
            axes=dataset.view_converter.axes)
        test_set = DenseDesignMatrixWrapper(
            topo_view=dataset.get_topological_view()[test_fold],
            y=dataset.y[test_fold],
            axes=dataset.view_converter.axes)
        # make ordered dict to make it easier to iterate, i.e. for logging
        datasets = OrderedDict([('train', train_set), ('valid', valid_set),
                ('test', test_set)])
        return datasets

class FixedTrialSplitter(TrainValidTestSplitter):
    def __init__(self, n_train_trials, valid_set_fraction):
        self.n_train_trials = n_train_trials
        self.valid_set_fraction = valid_set_fraction
        
    def split_into_train_valid_test(self, dataset):
        """ Split into train valid test by splitting 
        dataset into num folds, test fold nr should be given, 
        valid fold will be the one immediately before the test fold, 
        train folds the remaining 8 folds"""
        assert dataset.view_converter.axes[0] == 'b'
        assert hasattr(dataset, 'X')  # needs to be loaded already
        n_trials = dataset.get_topological_view().shape[0]
        assert n_trials > self.n_train_trials
        
        # split train into train and valid
        # valid is at end, just subtract -1 because of zero-based indexing
        i_last_valid_trial = self.n_train_trials - 1
        i_last_train_trial = self.n_train_trials - 1 - int(
            self.valid_set_fraction * self.n_train_trials)
        # always +1 since ranges are exclusive the end index(!)
        train_fold = range(i_last_train_trial + 1)
        valid_fold = range(i_last_train_trial + 1, i_last_valid_trial + 1)
        test_fold = range(i_last_valid_trial + 1, n_trials)
        
        datasets = split_set_by_indices(dataset, train_fold, valid_fold,
            test_fold)
        
        # remerge sets here

        return datasets

def concatenate_sets(first_set, second_set):
        """ Concatenates topo views and y(targets)"""
        assert first_set.view_converter.axes == second_set.view_converter.axes, \
            "first set and second set should have same axes ordering"
        assert first_set.view_converter.axes[0] == 'b', ("Expect batch axis "
            "as first axis")
        merged_topo_view = np.concatenate((first_set.get_topological_view(),
            second_set.get_topological_view()))
        merged_y = np.concatenate((first_set.y, second_set.y)) 
        merged_set = DenseDesignMatrixWrapper(
            topo_view=merged_topo_view,
            y=merged_y,
            axes=first_set.view_converter.axes)
        return merged_set

class PreprocessedSplitter(object):
    def __init__(self, dataset_splitter, preprocessor):
        self.dataset_splitter = dataset_splitter
        self.preprocessor = preprocessor

    def get_train_valid_test(self, dataset):
        datasets = self.dataset_splitter.split_into_train_valid_test(dataset)
        if dataset.reloadable:
            dataset.free_memory()
        if self.preprocessor is not None:
            self.preprocessor.apply(datasets['train'], can_fit=True)
            self.preprocessor.apply(datasets['valid'], can_fit=False)
            self.preprocessor.apply(datasets['test'], can_fit=False)
        return datasets

    def get_train_merged_valid_test(self, dataset):
        dataset.ensure_is_loaded()
        this_datasets = self.dataset_splitter.split_into_train_valid_test(dataset)
        if dataset.reloadable:
            dataset.free_memory()
        train_valid_set = concatenate_sets(this_datasets['train'],
            this_datasets['valid'])
        test_set = this_datasets['test']
        n_train_set_trials = len(this_datasets['train'].y)
        del this_datasets['train']
        if self.preprocessor is not None:
            self.preprocessor.apply(train_valid_set, can_fit=True)
            self.preprocessor.apply(test_set, can_fit=False)
        _, valid_set = self.split_sets(train_valid_set,
            n_train_set_trials, len(this_datasets['valid'].y))
        # train valid is the new train set!!
        return {'train': train_valid_set, 'valid': valid_set,
            'test': test_set}

    
    def split_sets(self, full_set, split_index, split_to_end_num):
        """ Assumes that full set may be doubled or tripled in size
        and split index refers to original size. So
        if we originally had 100 trials (set1) + 20 trials (set2) 
        merged to 120 trials, we get a split index of 100.
        If we later have 360 trials we assume that the 360 trials 
        consist of:
        100 trials set1 + 20 trials set2 + 100 trials set1 + 20 trials set2
        + 100 trials set1 + 20 trials set2
        (and not 300 trials set1 + 60 trials set2)"""
        full_topo = full_set.get_topological_view()
        full_y = full_set.y
        original_full_len = split_index + split_to_end_num
        topo_first = full_topo[:split_index]
        y_first = full_y[:split_index]
        topo_second = full_topo[split_index:original_full_len]
        y_second = full_y[split_index:original_full_len]
        next_start = original_full_len
        # Go through possibly appended transformed copies of dataset
        # If preprocessors did not change dataset size, this is not 
        # necessary
        for next_split in xrange(next_start + split_index,
                len(full_set.y), original_full_len):
            assert False, "Please check/test this code again if you need it"
            next_end = next_split + split_to_end_num
            topo_first = np.concatenate((topo_first,
                full_topo[next_start:next_split]))
            y_first = np.concatenate((y_first, full_y[next_start:next_split]))
            topo_second = np.concatenate((topo_second,
                full_topo[next_split:next_end]))
            y_second = np.concatenate((y_second, full_y[next_split:next_end]))
            next_start = next_end
        first_set = DenseDesignMatrixWrapper(
            topo_view=topo_first,
            y=y_first,
            axes=full_set.view_converter.axes)
        second_set = DenseDesignMatrixWrapper(
            topo_view=topo_second,
            y=y_second,
            axes=full_set.view_converter.axes)
        return first_set, second_set
    
def as_theano_expression(input):
    """Wrap as Theano expression.
    Wraps the given input as a Theano constant if it is not
    a valid Theano expression already. Useful to transparently
    handle numpy arrays and Python scalars, for example.
    Parameters
    ----------
    input : number, numpy array or Theano expression
        Expression to be converted to a Theano constant.
    Returns
    -------
    Theano symbolic constant
        Theano constant version of `input`.
    """
    if isinstance(input, theano.gof.Variable):
        return input
    else:
        try:
            return theano.tensor.constant(input)
        except Exception as e:
            raise TypeError("Input of type %s is not a Theano expression and "
                            "cannot be wrapped as a Theano constant (original "
                            "exception: %s)" % (type(input), e))



class MergeLayer(Layer):
    """
    This class represents a layer that aggregates input from multiple layers.
    It should be subclassed when implementing new types of layers that obtain
    their input from multiple layers.
    Parameters
    ----------
    incomings : a list of :class:`Layer` instances or tuples
        The layers feeding into this layer, or expected input shapes.
    name : a string or None
        An optional name to attach to this layer.
    """
    def __init__(self, incomings, name=None):
        self.input_shapes = [incoming if isinstance(incoming, tuple)
                             else incoming.output_shape
                             for incoming in incomings]
        self.input_layers = [None if isinstance(incoming, tuple)
                             else incoming
                             for incoming in incomings]
        self.name = name
        self.params = OrderedDict()
        self.get_output_kwargs = []

    @Layer.output_shape.getter
    def output_shape(self):
        return self.get_output_shape_for(self.input_shapes)

    def get_output_shape_for(self, input_shapes):
        """
        Computes the output shape of this layer, given a list of input shapes.
        Parameters
        ----------
        input_shape : list of tuple
            A list of tuples, with each tuple representing the shape of one of
            the inputs (in the correct order). These tuples should have as many
            elements as there are input dimensions, and the elements should be
            integers or `None`.
        Returns
        -------
        tuple
            A tuple representing the shape of the output of this layer. The
            tuple has as many elements as there are output dimensions, and the
            elements are all either integers or `None`.
        Notes
        -----
        This method must be overridden when implementing a new
        :class:`Layer` class with multiple inputs. By default it raises
        `NotImplementedError`.
        """
        raise NotImplementedError

    def get_output_for(self, inputs, **kwargs):
        """
        Propagates the given inputs through this layer (and only this layer).
        Parameters
        ----------
        inputs : list of Theano expressions
            The Theano expressions to propagate through this layer.
        Returns
        -------
        Theano expressions
            The output of this layer given the inputs to this layer.
        Notes
        -----
        This is called by the base :meth:`lasagne.layers.get_output()`
        to propagate data through a network.
        This method should be overridden when implementing a new
        :class:`Layer` class with multiple inputs. By default it raises
        `NotImplementedError`.
        """
        raise NotImplementedError

def get_output(layer_or_layers, inputs=None, **kwargs):
    """
    Computes the output of the network at one or more given layers.
    Optionally, you can define the input(s) to propagate through the network
    instead of using the input variable(s) associated with the network's
    input layer(s).
    Parameters
    ----------
    layer_or_layers : Layer or list
        the :class:`Layer` instance for which to compute the output
        expressions, or a list of :class:`Layer` instances.
    inputs : None, Theano expression, numpy array, or dict
        If None, uses the input variables associated with the
        :class:`InputLayer` instances.
        If a Theano expression, this defines the input for a single
        :class:`InputLayer` instance. Will throw a ValueError if there
        are multiple :class:`InputLayer` instances.
        If a numpy array, this will be wrapped as a Theano constant
        and used just like a Theano expression.
        If a dictionary, any :class:`Layer` instance (including the
        input layers) can be mapped to a Theano expression or numpy
        array to use instead of its regular output.
    Returns
    -------
    output : Theano expression or list
        the output of the given layer(s) for the given network input
    Notes
    -----
    Depending on your network architecture, `get_output([l1, l2])` may
    be crucially different from `[get_output(l1), get_output(l2)]`. Only
    the former ensures that the output expressions depend on the same
    intermediate expressions. For example, when `l1` and `l2` depend on
    a common dropout layer, the former will use the same dropout mask for
    both, while the latter will use two different dropout masks.
    """
    # track accepted kwargs used by get_output_for
    accepted_kwargs = {'deterministic'}
    # obtain topological ordering of all layers the output layer(s) depend on
    treat_as_input = inputs.keys() if isinstance(inputs, dict) else []
    all_layers = get_all_layers(layer_or_layers, treat_as_input)
    # initialize layer-to-expression mapping from all input layers
    all_outputs = dict((layer, layer.input_var)
                       for layer in all_layers
                       if isinstance(layer, InputLayer) and
                       layer not in treat_as_input)
    # update layer-to-expression mapping from given input(s), if any
    if isinstance(inputs, dict):
        all_outputs.update((layer, as_theano_expression(expr))
                           for layer, expr in inputs.items())
    elif inputs is not None:
        if len(all_outputs) > 1:
            raise ValueError("get_output() was called with a single input "
                             "expression on a network with multiple input "
                             "layers. Please call it with a dictionary of "
                             "input expressions instead.")
        for input_layer in all_outputs:
            all_outputs[input_layer] = as_theano_expression(inputs)
    # update layer-to-expression mapping by propagating the inputs
    for layer in all_layers:
        if layer not in all_outputs:
            try:
                if isinstance(layer, MergeLayer):
                    layer_inputs = [all_outputs[input_layer]
                                    for input_layer in layer.input_layers]
                else:
                    layer_inputs = all_outputs[layer.input_layer]
            except KeyError:
                # one of the input_layer attributes must have been `None`
                raise ValueError("get_output() was called without giving an "
                                 "input expression for the free-floating "
                                 "layer %r. Please call it with a dictionary "
                                 "mapping this layer to an input expression."
                                 % layer)
            all_outputs[layer] = layer.get_output_for(layer_inputs, **kwargs)
            try:
                names, _, _, defaults = getargspec(layer.get_output_for)
            except TypeError:
                # If introspection is not possible, skip it
                pass
            else:
                if defaults is not None:
                    accepted_kwargs |= set(names[-len(defaults):])
            accepted_kwargs |= set(layer.get_output_kwargs)
    unused_kwargs = set(kwargs.keys()) - accepted_kwargs
    if unused_kwargs:
        suggestions = []
        for kwarg in unused_kwargs:
            suggestion = get_close_matches(kwarg, accepted_kwargs)
            if suggestion:
                suggestions.append('%s (perhaps you meant %s)'
                                   % (kwarg, suggestion[0]))
            else:
                suggestions.append(kwarg)
        log.warn("get_output() was called with unused kwargs:\n\t%s"
             % "\n\t".join(suggestions))
    # return the output(s) of the requested layer(s) only
    try:
        return [all_outputs[layer] for layer in layer_or_layers]
    except TypeError:
        return all_outputs[layer_or_layers]

def unique(l):
    """Filters duplicates of iterable.
    Create a new list from l with duplicate entries removed,
    while preserving the original order.
    Parameters
    ----------
    l : iterable
        Input iterable to filter of duplicates.
    Returns
    -------
    list
        A list of elements of `l` without duplicates and in the same order.
    """
    new_list = []
    seen = set()
    for el in l:
        if el not in seen:
            new_list.append(el)
            seen.add(el)

    return new_list
def get_all_params(layer, **tags):
    """
    Returns a list of Theano shared variables that parameterize the layer.
    This function gathers all parameter variables of all layers below one or
    more given :class:`Layer` instances, including the layer(s) itself. Its
    main use is to collect all parameters of a network just given the output
    layer(s).
    By default, all shared variables that participate in the forward pass will
    be returned. The list can optionally be filtered by specifying tags as
    keyword  arguments. For example, ``trainable=True`` will only return
    trainable parameters, and ``regularizable=True`` will only return
    parameters that can be regularized (e.g., by L2 decay).
    Parameters
    ----------
    layer : Layer or list
        The :class:`Layer` instance for which to gather all parameters, or a
        list of :class:`Layer` instances.
    **tags (optional)
        tags can be specified to filter the list. Specifying ``tag1=True``
        will limit the list to parameters that are tagged with ``tag1``.
        Specifying ``tag1=False`` will limit the list to parameters that
        are not tagged with ``tag1``. Commonly used tags are
        ``regularizable`` and ``trainable``.
    Returns
    -------
    params : list
        A list of Theano shared variables representing the parameters.
    Examples
    --------
    >>> from lasagne.layers import InputLayer, DenseLayer
    >>> l_in = InputLayer((100, 20))
    >>> l1 = DenseLayer(l_in, num_units=50)
    >>> all_params = get_all_params(l1)
    >>> all_params == [l1.W, l1.b]
    True
    Notes
    -----
    If any layer's parameter was set to a Theano expression instead of a shared
    variable, the shared variables involved in that expression will be returned
    rather than the expression itself. Tag filtering considers all variables
    within an expression to be tagged the same.
    >>> import theano
    >>> import numpy as np
    >>> from lasagne.utils import floatX
    >>> w1 = theano.shared(floatX(.01 * np.random.randn(50, 30)))
    >>> w2 = theano.shared(floatX(1))
    >>> l2 = DenseLayer(l1, num_units=30, W=theano.tensor.exp(w1) - w2, b=None)
    >>> all_params = get_all_params(l2, regularizable=True)
    >>> all_params == [l1.W, w1, w2]
    True
    """
    layers = get_all_layers(layer)
    params = chain.from_iterable(l.get_params(**tags) for l in layers)
    return unique(params)


    
class Experiment(object):
    def __init__(self, final_layer, dataset, splitter, preprocessor,
            iterator, loss_expression, updates_expression, updates_modifier,
            monitors, stop_criterion, remember_best_chan):
        self.final_layer = final_layer
        self.dataset = dataset
        self.dataset_provider = PreprocessedSplitter(splitter, preprocessor)
        self.preprocessor = preprocessor
        self.iterator = iterator
        self.loss_expression = loss_expression
        self.updates_expression = updates_expression
        self.updates_modifier = updates_modifier
        self.monitors = monitors
        self.stop_criterion = stop_criterion
        self.monitor_manager = MonitorManager(monitors)
    
    def setup(self, target_var=None):
        set_rng(RandomState(9859295))
        self.dataset.ensure_is_loaded()
        self.print_layer_sizes()
        log.info("Create theano functions...")
        self.create_theano_functions(target_var)
        log.info("Done.")

    def print_layer_sizes(self):
        log.info("Layers...")
        layers = get_all_layers(self.final_layer)
        for l in layers:
            log.info(l.__class__.__name__)
            log.info(l.output_shape)
    
    def create_theano_functions(self, target_var):
        if target_var is None:
            if self.dataset.y.ndim == 1:
                target_var = T.ivector('targets')
            elif self.dataset.y.ndim == 2:
                target_var = T.imatrix('targets')
            else:
                raise ValueError("expect y to either be a tensor or a matrix")
        prediction = get_output(self.final_layer,
            deterministic=False)
        
        # test as in during testing not as in "test set"
        test_prediction = get_output(self.final_layer,
            deterministic=True)
        # Loss function might need layers or not...
        try:
            loss = self.loss_expression(prediction, target_var).mean()
            test_loss = self.loss_expression(test_prediction, target_var).mean()
        except TypeError:
            loss = self.loss_expression(prediction, target_var, self.final_layer).mean()
            test_loss = self.loss_expression(test_prediction, target_var, self.final_layer).mean()
            
        # create parameter update expressions
        params = get_all_params(self.final_layer, trainable=True)
        updates = self.updates_expression(loss, params)
        if self.updates_modifier is not None:
            # put norm constraints on all layer, for now fixed to max kernel norm
            # 2 and max col norm 0.5
            updates = self.updates_modifier.modify(updates, self.final_layer)
        input_var = get_all_layers(self.final_layer)[0].input_var
        # Store all parameters, including update params like adam params,
        # needed for resetting to best model after early stop
        all_layer_params = get_all_params(self.final_layer)
        self.all_params = all_layer_params
        # now params from adam would still be missing... add them ...
        all_update_params = updates.keys()
        for param in all_update_params:
            if param not in self.all_params:
                self.all_params.append(param)

        self.train_func = theano.function([input_var, target_var], updates=updates)
        self.monitor_manager.create_theano_functions(input_var, target_var,
            test_prediction, test_loss)
        
    def run(self):
        log.info("Run until first stop...")
        self.run_until_early_stop()
        log.info("Setup for second stop...")
        self.setup_after_stop_training()
        log.info("Run until second stop...")
        self.run_until_second_stop()

    def run_until_early_stop(self):
        log.info("Split/Preprocess datasets...")
        datasets = self.dataset_provider.get_train_valid_test(self.dataset)
        log.info("...Done")
        self.create_monitors(datasets)
        self.run_until_stop(datasets, remember_best=True)
        


    def run_until_stop(self, datasets, remember_best):
        self.monitor_epoch(datasets)
        self.print_epoch()
        if remember_best:
            self.remember_extension.remember_epoch(self.monitor_chans,
                self.all_params)
            
        self.iterator.reset_rng()
        while not self.stop_criterion.should_stop(self.monitor_chans):
            self.run_one_epoch(datasets, remember_best)

    def run_one_epoch(self, datasets, remember_best):
        batch_generator = self.iterator.get_batches(datasets['train'], shuffle=True)
        for inputs, targets in batch_generator:
            self.train_func(inputs, targets)
        log.info("next epoch....")
        
        self.monitor_epoch(datasets)
        self.print_epoch()
        if remember_best:
            self.remember_extension.remember_epoch(self.monitor_chans, self.all_params)

    def run_until_second_stop(self):
        datasets = self.dataset_provider.get_train_merged_valid_test(
            self.dataset)
        self.run_until_stop(datasets, remember_best=False)

    def create_monitors(self, datasets):
        self.monitor_chans = OrderedDict()
        self.last_epoch_time = None
        for monitor in self.monitors:
            monitor.setup(self.monitor_chans, datasets)
            
    def monitor_epoch(self, all_datasets):
        self.monitor_manager.monitor_epoch(self.monitor_chans, all_datasets,
            self.iterator)

    def print_epoch(self):
        # -1 due to doing one monitor at start of training
        i_epoch = len(self.monitor_chans.values()[0]) - 1 
        log.info("Epoch {:d}".format(i_epoch))
        for chan_name in self.monitor_chans:
            log.info("{:25s} {:.5f}".format(chan_name,
                self.monitor_chans[chan_name][-1]))
        log.info("")

def sgd_exp():
    # successfully crashed Monday, 15th february 19:18
    # 4) adam durch sgd ersetzen
    network = InputLayer([None, 32, 2000, 1])
    network = DimshuffleLayer(network, [0, 3, 2, 1])
    network = Conv2DLayer(network, num_filters=40, filter_size=[30, 1],
        nonlinearity=identity, name='time_conv')
    network = Conv2DAllColsLayer(network, num_filters=40, filter_size=[1, -1],
        nonlinearity=T.sqr, name='spat_conv')
    network = SumPool2dLayer(network, pool_size=[50, 1], stride=[1, 1],
        mode='average_exc_pad')
    network = StrideReshapeLayer(network, n_stride=10)
    network = NonlinearityLayer(network, nonlinearity=safe_log)
    network = DropoutLayer(network, p=0.5)
    network = Conv2DLayer(network, num_filters=6, filter_size=[54, 1],
        nonlinearity=identity, name='final_dense')
    network = FinalReshapeLayer(network)
    network = NonlinearityLayer(network, nonlinearity=sigmoid)

    preprocessor = None

    iterator = CntWindowsFromCntIterator(batch_size=20,
        input_time_length=2000, n_sample_preds=1392,
        oversample_targets=True, remove_baseline_mean=False)
    loss_expression = categorical_crossentropy
    updates_expression = FuncAndArgs(sgd, learning_rate=0.1)
    updates_modifier = None
    monitors = [LossMonitor(), AUCMeanMisclassMonitor(input_time_length=2000,
        n_sample_preds=1392), RuntimeMonitor()]
    stop_criterion = MaxEpochs(1000)
    rng = RandomState(30493049)
    random_topo = rng.randn(1334972, 32, 1, 1)
    random_y = np.round(rng.rand(1334972, 6)).astype(np.int32)

    dataset = DenseDesignMatrixWrapper(topo_view=random_topo, y=random_y,
        axes=('b', 'c', 0, 1))
    splitter = FixedTrialSplitter(n_train_trials=1234972, valid_set_fraction=0.1)
    
    exp = Experiment(network, dataset, splitter, preprocessor, iterator, loss_expression,
        updates_expression, updates_modifier, monitors, stop_criterion, remember_best_chan='valid_misclass')
    exp.setup()
    datasets = exp.dataset_provider.get_train_merged_valid_test(dataset)
    exp.create_monitors(datasets)
    exp.run_until_second_stop()


if __name__ == "__main__":
    sgd_exp()
