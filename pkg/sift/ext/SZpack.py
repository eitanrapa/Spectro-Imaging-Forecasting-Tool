#!/usr/bin/env python3
# -*- Python -*-
# -*- coding: utf-8 -*-
#
# the sift development team
# (c) 2023-2025 all rights reserved

import numpy as np
import sift


class SZpack:
    """

    """

    def __init__(self, tau, temperature, peculiar_velocity, omega=0, sigma=0, kappa=0, betac2_perp=0):
        self.tau = tau
        self.TeSZ = temperature
        self.betac_para = peculiar_velocity / 3e5
        self.omega = omega
        self.sigma = sigma
        self.kappa = kappa
        self.betac2_perp = betac2_perp

    def sz_combo_means(self, x):
        """

        """

        output = []
        for xo in x:
            output.append(sift.libsift.szpack_combo_means(xo=xo, tau=self.tau, TeSZ=self.TeSZ,
                                                          betac_para=self.betac_para,
                                                          omega=self.omega, sigma=self.sigma, kappa=self.kappa,
                                                          betac2_perp=self.betac2_perp))

        return np.asanyarray(output)

# end of file
