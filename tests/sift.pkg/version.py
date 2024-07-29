#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# the sift development team
# (c) 2023-2024 all rights reserved


"""
Version check
"""


def test():
    # access the {sift} package
    import sift
    from sift import libsift
    # verify that the static and current versions match
    assert libsift.version.static() == sift.version()
    # verify that the dynamic and current versions match
    assert libsift.version.dynamic() == sift.version()
    # all done
    return


# main
if __name__ == "__main__":
    # do...
    test()

# end of file
