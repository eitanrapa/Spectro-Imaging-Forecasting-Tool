# -*- Python -*-
# -*- coding: utf-8 -*-
#
# the sift development team
# california institute of technology
# (c) 2023-2024 all rights reserved
#


# publish the local modules
# the bindings
from .ext import libsift
# basic functionality
from . import meta
from .Simulation import Simulation as simulation
from .Parameters import Parameters as parameters
from .Projection import Projection as projection

# by convention
__version__ = meta.version


# administrative
def copyright():
    """
    Return the copyright note
    """
    # pull and print the meta-data
    return print(meta.header)


def license():
    """
    Print the license
    """
    # pull and print the meta-data
    return print(meta.license)


def built():
    """
    Return the build timestamp
    """
    # pull and return the meta-data
    return meta.date


def credits():
    """
    Print the acknowledgments
    """
    return print(meta.acknowledgments)


def version():
    """
    Return the version
    """
    # pull and return the meta-data
    return meta.version

# end of file
