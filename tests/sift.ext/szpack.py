#!/usr/bin/env python3
# -*- Python -*-
# -*- coding: utf-8 -*-
#
# the sift development team
# (c) 2023-2024 all rights reserved

import sift

# Create a SZpack object
sz = sift.ext.szPack(tau=0.01, temperature=10, peculiar_velocity=10)
print(sz.sz_combo_means(100))
# end of file
