This is a proof-of-concept for a minimal dispatcher system, it may be renamed,
replaced or just found to be a bad idea and go away :).

This is a very strict type-dispatcher compared to most system!
It has two, somewhat distinct, features under a single umbrella:

1. A restrictive, single-type but multiple-arguments type-dispatcher.  We
   currently have the assumption that the single type has some global meaning:
   It makes sense for any array-like function parameter.
2. A user opt-in, prioritized, alternative selection (backend-selection)

The sole purpose is to proto-type ideas related to the following discussions:
* https://discuss.scientific-python.org/t/requirements-and-discussion-of-a-type-dispatcher-for-the-ecosystem/157/34
* https://discuss.scientific-python.org/t/a-proposed-design-for-supporting-multiple-array-types-across-scipy-scikit-learn-scikit-image-and-beyond/131/31

**Please check `example.py` for a much clearer idea of what this does.**

### More detailed discussion (don't get lost here! ;))

The first means that it uses a simplified strategy similar to Python's
`singledispatch` to find the best implementation registered.  The additional
feature is that is that unlike `singledispatch` it works with multiple arguments
to find an implementation which works for all.
The major restriction is that, unlike `singledispatch`, `precodiate` linearizes
the implementation order rather than using the MRO of the dispatched type.
It further rejects matching subclasses, if the super-class is not abstract.

The first type-dispatching restriction means that especially in the case of
multiple-inheritance it is sometimes not possible to find the correct
"best" implementation (in cases where singledispatch can define it).
The second restriction is implemented because in the world of arrays,
subclasses usually add behaviour in a way that means implementations cannot be
expected to return useful results anymore.
This makes subclassing less convenient (because sometimes things work out)
but protects from those surprises common to NumPy's masked arrays and matrices
(why did I get a normal array back and the mask got ignored?).

One additional caveat: Unlike `singledispatch` we currently assume that
no new ABC subclasses are registered after any interaction with the dispatch
(registration an implementation or caching).

The second point provides alternative implementations for a specific type.
Checking alternatives is slower.  Alternative "backends" are disabled initially
and ignored, however, they can be enabled at runtime at which point they will
be the preferred implementation if they match equally well as another one.
(We expect they typically match identically to another implementation.)

This adds a layer of dynamism that comes with some caveats, requiring user care,
because modifying the priority may affect other libraries within the scope.
It also may be less efficient, because priorities need to be checked in a
context safe way, as they can change.

Potential features include things like deferral: Allowing an implementation
to indicate that the next best implementation should be used.

Why are alternatives disabled by default?  The reason is that we mainly rely
on the types to define what the "best" implementation is.  Thus priorities
would only apply if the type hierarchy finds two alternatives registered for
identical types.
Limiting us to either disabled or "prioritized" allows an alternative to
match identical or more precise types than the original implementation.
The case of more precise would not be simple otherwise.

### Current limitations and possible future additions

* Pickling probably doesn't work reasonably (or is a mess).  I am not actually
  sure how it should work!
* The code assumes all parameters to use or dispatching belong to the same
  "category" of array-like parameters.  It should be possible to relax this
  and allow more than one type (this makes dispatching harder to resolve and
  easier to get ambiguities, but I see no reason it is not possible.
  2. Make the global state specific so we can use nicer for more than just
     array-like parameters (in a program that also uses it for those).
* It is not really easy/possible to register with a function that does not
  exist yet.  I am considering implementing a:
  `@LazyDispatchable("skimage.morphology", "binary_erosion")` to avoid having
  to use the public symbol.  This probably needs a global dict of Dispatchables.
  If this Dispatchable exists, we use it directly, if not we check whenever a
  new Dispatchable is defined.  (Or something like this)
* The code does not validate that `ABC` are unmodified.  After any dispatch
  occurred, ABCs should remain unmodified.  This may mean that ABCs that want
  to use `ABC.register()` have to take some extra care.
  (We could probably relax this, at least somewhat.  `singledispatch` is
  careful about it.  I do not see that it is a relevant limitation right now.)
* Currently, whether a class is considered "abstract" is equivalent to it being
  an `abc.ABC`.  This seems like an OK limitation, but alternatives may be
  possible.  We could also allow it to make a backend for multiple explicit
  types `(np.ndarray | astropy.Quantity)`, but not sure it is really necessary
  in practice.


(PS: Adding a GPL license for the heck of it.  But obviously will relicense
as MIT on request, if contributions come in, or if it becomes a bit more
polished.)
