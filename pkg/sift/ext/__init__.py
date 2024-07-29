# -*- coding: utf-8 -*-
#
# the sift development team
# (c) 2023-2024 all rights reserved

from .SZpack import SZpack as szPack

# attempt to
try:
    # pull the extension module
    from . import sift as libsift
# if this fails
except ImportError:
    # indicate the bindings are not accessible
    libsift = None


# end of file 
