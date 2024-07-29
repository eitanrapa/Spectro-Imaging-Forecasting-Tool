// -*- C++ -*-
//
// the sift development team
// (c) 2023-2024 all rights reserved


// external
#include "external.h"
// namespace setup
#include "forward.h"


// the module entry point
PYBIND11_MODULE(sift, m)
{

    // the doc string
    m.doc() = "the libsift bindings";

    // bind the opaque types
    sift::py::opaque(m);
    // register the exception types
    sift::py::exceptions(m);
    // version info
    sift::py::version(m);

    return;
}


// end of file
