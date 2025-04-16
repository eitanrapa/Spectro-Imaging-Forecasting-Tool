# -*- makefile -*-
#
# the sift development team
# (c) 2023-2025 all rights reserved


# sift consists of a python package
sift.packages := sift.pkg
# libraries
sift.libraries := sift.lib
# python extensions
sift.extensions := sift.ext
# a ux bundle
sift.webpack :=
# and some tests
sift.tests := sift.lib.tests sift.ext.tests sift.pkg.tests


# load the packages
include $(sift.packages)
# the libraries
include $(sift.libraries)
# the extensions
include $(sift.extensions)
# the ux
include $(sift.webpack)
# and the test suites
include $(sift.tests)


# end of file
