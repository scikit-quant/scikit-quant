from __future__ import print_function
import copy, string

__all__ = ['optset']

STANDALONE = True

def optset(optin=None, **optkwds):
    """
 Set the options in SnobFit.

  optout = optset(name1=val1, name2=val2, ...)
           creates the options dictionary with the named options
           set to something other than the defaults.

  optout = optset(optin, name1=val1, name2=val2, ...);
           modifies the dictionary <optin> and creates a new structure
           <optout>.

  And now for the lists. The details of the options, the motiviation
  for the defaults, and suggestions for their use live in the manual.

 Scalar Options:

  maxmp: Maximum number of model points requested for the local fit.
         Default = 2*npar+6, with npar the number of parameters.

  maxfail: Maximum number of consecutive failures before iteration stops.
         Default = 5.

  verbose: Provide verbose (debugging) output
         Default = False
   """

    if not optin:
    # Start from default structure
        optout = _create_defaults()
    else:
        optout = copy.copy(optin)

    for name, val in optkwds.items():
        setattr(optout, name.lower(), val)

    return optout


#-----
def _create_defaults():
    """Default values"""

    class OptSet(object):
        __slots__ = {
            'maxmp' : None,             # Maximum number of model points (default set in driver)
            #
            'maxfail' : 5,              # Stop iterating after maxfail failures to improve
            #
            'verbose' : False,          # Provide verbose output
        }

        def __init__(self):
            for key, value in self.__slots__.items():
                setattr(self, key, value)

    return OptSet()
