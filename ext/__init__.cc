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

    // Geodetic to Cartesian function
    m.def("szpack_combo_means", &sift::compute_SZ_signal_combo_means, "Get SZ distortion",
          py::arg("xo"), py::arg("tau"), py::arg("TeSZ"), py::arg("betac_para"), py::arg("omega"), py::arg("sigma"),
           py::arg("kappa"), py::arg("betac2_perp"));

    // bind the opaque types
    sift::py::opaque(m);
    // register the exception types
    sift::py::exceptions(m);
    // version info
    sift::py::version(m);

    return;
}


// end of file
