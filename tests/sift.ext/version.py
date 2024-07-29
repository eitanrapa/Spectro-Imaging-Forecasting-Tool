#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# the sift development team
# california institute of technology
# (c) 2023-2024 all rights reserved
#


"""
Version check
"""


def test():
    # access the {sift} extension
    from sift import libsift
    # verify that the static and dynamic versions match
    assert libsift.version.static() == libsift.version.dynamic()
    # all done
    return


# main
if __name__ == "__main__":
    # do...
    test()


# end of file
