#!/usr/bin/env python3
# -*- Python -*-
# -*- coding: utf-8 -*-
#
# the sift development team
# (c) 2023-2024 all rights reserved

import sift

# Create a conversions object
conversions = sift.ext.conversions(name="conversions", a=1, b=1, c=1)

# Convert some coordinates to geodetic
geodetic_coordinates = conversions.geodetic([10, 10, 10])

# Convert back to Cartesian
cartesian_coordinates = conversions.cartesian(geodetic_coordinates)

print(geodetic_coordinates, cartesian_coordinates)

# Convert some coordinates to geodetic
geodetic_coordinates = conversions.geodetic([[10, 10, 10], [5, 5, 5]])

# Convert back to Cartesian
cartesian_coordinates = conversions.cartesian(geodetic_coordinates)

print(geodetic_coordinates, cartesian_coordinates)

# end of file
