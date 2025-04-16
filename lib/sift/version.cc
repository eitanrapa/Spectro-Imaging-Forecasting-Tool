// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// the sift development team
// (c) 2023-2025 all rights reserved

// my declarations
#include "version.h"

// build and return the version tuple
auto
sift::version::version() -> version_t
{
    // easy enough
    return version_t { major, minor, micro, revision };
}

// end of file
