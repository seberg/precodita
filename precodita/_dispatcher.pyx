#cython: language_level=3
"""
Implements the backend and main dispatch logic. 

"""
cimport cython

from cpython.object cimport PyObject, Py_LT
from libc.stdint cimport int64_t


cdef extern from "Python.h":
    cdef object PyObject_Vectorcall(
            object callable, PyObject **args, Py_ssize_t nargs, PyObject *kwnames)

    cdef unsigned long Py_TPFLAGS_HAVE_VECTORCALL

    ctypedef struct PyTypeObject:
        unsigned long tp_flags
        Py_ssize_t tp_vectorcall_offset

import contextvars

import abc
from functools import update_wrapper


cdef is_strict_subclass(cls, other):
    """Returns True if cls is other, or if other is an ABC and a superclass.
    """
    if cls is other:
        return True
    if not isinstance(other, abc.ABCMeta):
        # Other is concrete, a subclass of a known class cannot be considered
        # known (Dask can "promote" `np.ndarray`, but maybe not `np.matrix`).
        return False

    # For ABCs, assume that weird subclasses are anticipated.
    return issubclass(cls, other)


cdef int64_t _backend_counter = 0
# The "max priority".  When activating a backend, this is incremented by 1
# and then the priority is set to the new maximum.
# TODO: Does this need thread-locality?  Presumably not, or only when
#       subinterpreters become a thing, then it needs module locality.
#       (unless the GIL vanishes, then we may need locking...)
cdef int64_t _max_priority = 2**16

cdef list _all_backends = []


@cython.final
cdef class Backend:
    cdef unicode name
    cdef readonly object dispatch_type
    cdef readonly set promotable_types
    cdef int64_t fixed_priority
    cdef object ctx_priority
    cdef list tokens
    cdef object callback

    def __cinit__(self,
            name, dispatch_type, promotable_types, opt_in=False,
            callback=None):
        """
        Create a new backend.
        
        Parameters
        ----------
        name : str
            A name for the backend
        dispatch_type : type
            The main type to dispatch for
        promotable_types : sequence of types
            Further types which this backend recognizes and accepts.
        opt_in : boolean
            If `True`, the backend is disabled by default and can be enabled
            using a `with backend:` statement.  If `False`, the backend is
            always enabled.
        callback : callable or None
            If given, a function which is called with the new backend and
            any function newly wrapped as dispatchable.
        """
        global _backend_counter, _all_backends

        self.name = name
        self.dispatch_type = dispatch_type
        self.promotable_types = set(promotable_types)
        
        _backend_counter += 1
        if _backend_counter > 2**16:
            raise SystemError("Too many backends, oooops! :)")

        if opt_in:
            self.fixed_priority = -1  # Lower priority
            self.tokens = []
            self.ctx_priority = contextvars.ContextVar(
                    f"precodita.backend_priority.{self.name}", default=-1)
        else:
            self.fixed_priority = _backend_counter
            self.ctx_priority = None

        self.callback = callback

        _all_backends.append(self)

    @classmethod
    def all_backends(cls):
        global _all_backends
        return _all_backends[:]

    cdef int matches(self, tuple given_types) except -1:
        cdef Py_ssize_t i
        cdef int ntypes = len(given_types)
        cdef object main = None

        # Do a quick check whether any of the classes matches with the main
        # type of the backend.
        if not isinstance(self.dispatch_type, abc.ABCMeta):
            for i, t in enumerate(given_types):
                if t is self.dispatch_type:
                    main = t
                    break
        else:
            for t in given_types:
                if issubclass(t, self.dispatch_type):
                    main = t
                    break
        
        if main is None:
            return 0

        for t in given_types:
            if t is main:
                continue

            if is_strict_subclass(t, self.dispatch_type):
                continue

            promotable = False
            for allowed in self.promotable_types:
                if is_strict_subclass(t, allowed):
                    promotable = True
                    break
            if not promotable:
                return 0

        return 1

    cdef _more_specific(self, Backend other):
        """Check if this alternative is strictly more precise than the
        other alternative.
        We define more specific by whether the dispach type is a subclass of
        any of the alternatives types (including the promotable ones).

        We further check the opposite direction an: it is ambiguous if both
        are "more specific" in this sense (it is OK if neither is though).

        This is pretty restrictive, but we assume that it is always easy to
        resolve in practice!

        Note that we consider two backends as "equal" if their dispatch type
        matches exactly.  In this case, they will be tried based on their
        priority!  (By default this means the later registered one has
        precedence.)
        We do not use the promotable types as a tie breaker.

        """
        cdef int more_specific = 0

        if self.dispatch_type == other.dispatch_type:
            if self.fixed_priority >= 0 and other.fixed_priority >= 0:
                return self.fixed_priority > other.fixed_priority
            # Whether this matches better, depends on the priority
            return False
        
        if is_strict_subclass(self.dispatch_type, other.dispatch_type):
            # in practice, this should almost never trigger, because we rarely
            # use ABCs for the dispatchable type.  But if this is concrete,
            # then the opposite direction _must_ also be true and we cannot
            # really define that one is better than the other.
            more_specific = 1

        for promotable in other.promotable_types:
            if more_specific:
                break
            more_specific = is_strict_subclass(self.dispatch_type, promotable)

        return more_specific

    cdef int64_t get_priority(self) except? -1:
        if self.fixed_priority >= 0:
            return self.fixed_priority

        return self.ctx_priority.get()

    def __repr__(self):
        # Maybe should print the promotable types also?
        mod = getattr(self.dispatch_type, "__module__", "<unknown>")
        r = (f"Backend[{self.name}, "
             f"type={mod}.{self.dispatch_type.__qualname__}, "
             f"enabled={'True' if self.get_priority() > 0 else 'False'}]")
        return r

    def __richcmp__(self, Backend other, int op):
        if op != Py_LT:
            raise NotImplementedError("Only less-than is implemented :)")


        m = self._more_specific(other)
        if not m:
            return False

        if other._more_specific(self):
            raise TypeError(
                    "invalid type-hierarchy for two backends both claim to be "
                    "more specific!")
        return True

    def __enter__(self):
        global _max_priority
        if self.fixed_priority >= 0:
            raise RuntimeError("Cannot prioritize this backend!")

        _max_priority += 1
        token = self.ctx_priority.set(_max_priority)
        self.tokens.append(token)
        return self

    def __exit__(self, *exc):
        global _max_priority
        # Also increment, to indicate possible state change:
        _max_priority += 1
        
        token = self.tokens.pop(-1)
        self.ctx_priority.reset(token)
        return False


# Because cython currently doesn't like it in a cpdef:
cdef sorted_unique_types(tuple types):
    """Simple insertion sort type sorter, that drops the unique items.
    """
    cdef list unique = []
    cdef Py_ssize_t i

    for t in types:
        for i in range(len(unique)):
            if t is unique[i]:
                continue
            if <PyObject *>unique[i] > <PyObject *>t:
                unique.insert(i, t)
            break
        else:
            unique.append(t)
        
    return tuple(unique)


DEF DCACHE_SIZE = 20
DEF FCACHE_SIZE = 20


@cython.final
cdef class Dispatchable:
    cdef void *_vectorcall
    cdef object extractor
    cdef object fallback
    # fcache: dict to cache fixed-type results (lists of matches)
    # Note: We pop and readd to the cache, to use it as an LRU...
    cdef dict fcache  # cache for non-mutable results
    cdef int64_t fc_hits
    cdef int64_t fc_misses
    # dcache: dict to cache "dynamic" final result (context dependend)
    cdef int64_t dcache_priority  # priority
    cdef dict dcache
    cdef int64_t dc_hits
    cdef int64_t dc_misses

    cdef list alternatives
    cdef unicode _repr
    cdef dict __dict__

    @classmethod
    def from_fallback(cls, extractor, fallback=None):
        """Allows using an existing method as a fallback.  In this path
        the method will be used as the symbol and provider of documentation.
        (But things like argument deprecations should live on the extractor)

        Parameters
        ----------
        extractor : callable
            The function which returns the "relevant arguments".
        fallback : callable, optional
            Typically not passed.  If not provided, returns a function to
            be used as a decorator: ``@Dispatchable.from_fallback(extractor)``.

        Notes
        -----
        Unlike using ``Dispatchable`` on its own, this also copies the
        docstring from the fallback itself.  Note that it is possible to
        replace such a fallback in principle by carefully registering with
        a backend that matches everything (can be done using ABCs).

        Example::
        
            def _extractor(a, b, c=None):
                return (a, c)  # a and c can be arrays

            @Dispatchable.from_fallback
            def original_func(a, b, c=None):
                '''has a docstring and will be used if nothing else matches.'''
                return original_result
        """
        if fallback is not None:
            return cls(extractor, fallback)

        return lambda func: cls(extractor, func)

    def __cinit__(self, extractor, fallback=None):
        global _all_backends

        self._vectorcall = <void *>self.__vectorcall__
        self.extractor = extractor
        self.fallback = fallback
        self.alternatives = []
        if fallback is None:
            update_wrapper(self, extractor)
            self._repr = repr(extractor)
        else:
            update_wrapper(self, fallback)
            self._repr = repr(fallback)
        
        self.fcache = {}
        self.dcache = {}
        self.dcache_priority = -1

        for backend in _all_backends:
            if (<Backend>backend).callback is not None:
                (<Backend>backend).callback(backend, self)

    def __repr__(self):
        return self._repr

    def register(self, backend, func=None):
        """Register a new function with a backend.

        """
        if func is None:
            return lambda f: self.register(backend, f)

        if not isinstance(backend, Backend):
            raise ValueError("Register must be a Backend!")

        for alt in self.alternatives:
            if alt[0] is backend:
                raise ValueError("Backend already has an alternative registered.")

        self.alternatives.append((backend, func))
        self._clear_cache()
        return func

    cpdef _clear_cache(self):
        """Clear all caches and reset the cache stats.

        """
        self.fcache = {}
        self.c_fhits = 0
        self.c_fmisses = 0
        self.dcache = {}
        self.c_dhits = 0
        self.c_dmisses = 0

    @property
    def _cache_stats(self):
        """Cache stats for debugging only
        """
        return self.fc_hits, self.fc_misses, self.dc_hits, self.dc_misses

    @property
    def backends(self):
        return [a[0] for a in self.alternatives]

    @property
    def implementations(self):
        return self.alternatives[:]

    def invoke(self, *types):
        """See also dispatch, but returns only the final implementation.

        TODO: If we have more than just the array-like "category" on which
              we dispatch, using *types is no good!
              (Mainly a problem if types could be added to the same
              dispatchable at some future point?)
        """

        return self.dispatch(types)[1]

    cpdef dispatch(self, tuple types):
        """Return the dispatched version of the function.  This is currently
        a tuple containig the backend and its implementation for the function.
        
        Parameters
        ----------
        types : tuple of types
            Types (the user has to make sure they are types) to
            run the dispatching logic for.

        Returns
        -------
        backend : Backend or None
            The backend which provides this implementation. Can be None if the
            result is the Fallback implementation.
        implementation : callable
            The function that would (currently) be used for the given types.
        """
        global _max_priority
        
        cdef Py_ssize_t i
        cdef tuple typet
        cdef Backend best = None
        cdef Backend b, cb
        cdef int used_mutable_priority = 0
        cdef list curr = None
        cdef object res = None

        typet = sorted_unique_types(types)

        if len(typet) == 0:
            if self.fallback is not None:
                return None, self.fallback
            raise TypeError(f"Function called without types, but has no fallback!")

        # See if this is cached:
        cached = self.fcache.pop(typet, None)
        if cached is not None:
            self.fcache[typet] = cached
            self.fc_hits += 1
            curr = cached
        else:
            self.fc_misses += 1

        if curr is None or len(curr) > 1:
            # See if result is cached, assuming that no priority was changed:
            if _max_priority != self.dcache_priority:
                self.dcache = {}  # simply clear it
            else:
                res = self.dcache.pop(typet, None)
                if res is not None:
                    self.dcache[typet] = res
                    self.dc_hits += 1
                    return res

            self.dc_misses += 1

        if curr is None:
            # Caches failed us, build full list of alternatives:
            curr = []
            for b, f in self.alternatives:
                if not b.matches(typet):
                    continue
                
                for i, c in enumerate(curr):
                    cb = <Backend>(c[0])
                    if cb < b:
                        break
                    elif b < cb:
                        # replace and break
                        curr[i] = b, f
                        break
                else:
                    # A new possible backend to use:
                    curr.append((b, f))

            # Add to the fcache:
            if len(self.fcache) >= FCACHE_SIZE:
                self.fcache.pop(next(iter(self.fcache)))
            self.fcache[typet] = curr

        if len(curr) == 0:
            if self.fallback is not None:
                res = None, self.fallback
        elif len(curr) == 1:
            res = curr[0]
        elif len(curr) > 1:
            res = None
            for c in curr:
                cb = c[0]

                if cb.get_priority() < 0:
                    continue

                if res is not None and res[0].dispatch_type is not cb.dispatch_type:
                    print(res, cb)
                    raise TypeError("Multiple matching implementations found!")

                # TODO: If this is used, we may be using priorities (even of
                #       always-active backends) to tie-brake.  We may give a
                #       warning for this in some cases, or at least allow
                #       enabling a warning for debug purposes.
                if res is None or cb.get_priority() > (<Backend>res[0]).get_priority():
                    res = c

            if res is None and self.fallback is not None:
                res = None, self.fallback

            if res is not None:
                # Add the result to the dcache (presumably priorities were used)
                if len(self.dcache) >= DCACHE_SIZE:
                    self.dcache.pop(next(iter(self.dcache)))
                self.dcache[typet] = res
                self.dcache_priority = _max_priority

        if res is None:
            # No implementation seems to be available
            raise TypeError(f"No implementation found for types: {typet}")

        return res

    def __call__(self, *args, **kwargs):
        # Unfortunately, Cython does not use fastcall/vectorcall here.
        # This would be possible, since we just forward everything to the
        # extractor.
        # So we could micro-optimize this a bit more by tagging on vectorcall.
        relevant = self.extractor(*args, **kwargs)
        types = [type(r) for r in relevant if r is not None]

        _, func = self.dispatch(tuple(types))
        res = func(*args, **kwargs)

        if res is NotImplemented:
            raise NotImplementedError(
                    "backends cannot return NotImplemented to defer right now!")

        return res

    @staticmethod
    cdef __vectorcall__(Dispatchable self, PyObject **args, Py_ssize_t nargs, PyObject *kwnames):
        relevant = PyObject_Vectorcall(self.extractor, args, nargs, kwnames)
        types = [type(r) for r in relevant if r is not None]

        _, func = self.dispatch(tuple(types))
        res = PyObject_Vectorcall(func, args, nargs, kwnames)

        if res is NotImplemented:
            raise NotImplementedError(
                    "backends cannot return NotImplemented to defer right now!")

        return res


cdef Dispatchable dummy = Dispatchable(lambda: None, lambda: None)

(<PyTypeObject *>Dispatchable).tp_flags |=  Py_TPFLAGS_HAVE_VECTORCALL
(<PyTypeObject *>Dispatchable).tp_vectorcall_offset = 3 * sizeof(void *)

