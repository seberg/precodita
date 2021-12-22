import numpy as np
import abc

from precodita import Backend, Dispatchable


class ArrayLike(abc.ABC):
    """
    Simple ABC to show off that it is possible to provide generic
    implementations
    """
    @classmethod
    def __subclasshook__(cls, other):
        if hasattr(other, "__array_function__"):
            return True
        return NotImplemented


# ----------------------------------
# Lets define a couple of "backends"
# ----------------------------------
# I use the `uarray` definition of backend here, i.e. a type-dispatching target
# one is also a backend.  One used for "backend-selection" is one that has
# the `opt_in=True` set.
# However, backends do not do much work, they say what types they work with
# and get a priority assigned. Some are "opt in", they are disabled and set
# to "highest priority" using a with-statement (nesting multiple should work).


# I decided for now to always have backends.  This gives a nice way to
# define what typoes to support.
b1 = Backend("NumPy", np.ndarray, ())

# Add a backend for NumPy matrices (which also accept ndarray though!)
b2 = Backend("Matrix", np.matrix, (np.ndarray,))

# b3 will be auto-registered in its callback. The thought is that this
# could be useful to allow lazy-imports.
# This is probably a bad idea, I think I now prefer the thought of adding a
# `LazyDispatchable(mod, qualname)`.  This would use a global list and events
# events on each Dispatchable creation to register the function.
# (A callback may still be useful, e.g. for debug-tracing.)
def register_late(b, func):
    print("auto-registering backend:", b, func)
    func.register(b)(generic_)

b3 = Backend("generic", ArrayLike, (), callback=register_late)

# The following is disabled by default, but can be opted in and prioritized
# by using `with b4:`
b4 = Backend("Mat2", np.matrix, (np.ndarray,), opt_in=True)


# Will be auto-registered:
def generic_(*args, **kwargs):
    return b3, args, kwargs


# -----------------------------------
# Using a function without a fallback
# -----------------------------------

@Dispatchable
def func(a, b, c=None):
    """
    The "extractor", similar to `__array_function__`, we use this only to
    define the parameters which are dispatched on.  In this case, `a` and `c`.
    `None` will be ignored during dispatch.

    Currently, there is no "fallback" when there are no matches, this could
    be added easily however!
    
    Further, `uarray` has the "replacer" dynamic, which is convenient.
    Replacers could be used for extraction of parameters as well, so we could
    allow to provide an extractor only for speed (or even create a C-extractor
    for shaving off a tiny bit more).

    """
    return a, c

@func.register(b1)
def _(a, b, c=None):
    return b1, a, b, c

@func.register(b2)
def _(a, b, c=None):
    return b2, a, b, c

# b3 was auto-registered.

@func.register(b4)
def _(a, b, c=None):
    return b4, a, b, c


# The NumPy backend of course:
print(func(np.array(1), 2, np.array(3)))
# Matrix, since numpy cannot match it:
print(func(np.matrix(1), 2, np.array(3)))
# Only the generic one `ArrayLike` can match this one:
print(func(np.ma.array(1), 2, np.matrix(3)))

# Enable and enforce preference of the `Mat2` backend:
with b4:
    print(func(np.matrix(1), 2, np.array(3)))


# --------------------------------
# Using a function with a fallback
# --------------------------------

def _new_func_extractor(a):
    return (a,)  # Take care: must be a sequence!

@Dispatchable.from_fallback(_new_func_extractor)
def new_func(a):
    """This is the original function, that will always be used if things
    fail otherwise.
    """
    return "original", a

@new_func.register(b1)
def _(a):
    return b1, a


# definitely no match for string input...
print(new_func("asdf"))
# Actually, remember that the ArrayLike one got auto-registered?
print(new_func(np.matrix(1)))



# ------------------------------------------------
# Using a function without dispatchable parameters
# ------------------------------------------------

def _creation_extractor(shape, like=None):
    return (like,)


@Dispatchable.from_fallback(_creation_extractor)
def new(shape, like=None):
    return "Default version!", np.ones(shape)

@new.register(b2)  # matrix!
def _(shape, like=None):
    return "Matrix version!", np.ones(shape).view(np.matrix)

# The default version (of course):
print(new((2, 2)))
# We can get a matrix result in two ways:
print(new.invoke(np.matrix)((2, 2)))
print(new((2, 2), like=np.matrix([2])))

