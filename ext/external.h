// -*- c++ -*-
//
// the sift development team
// (c) 2023-2024 all rights reserved


// code guard
#pragma once

// STL
#include <string>

// pybind support
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

// conversion code
#include <sift/sift.h>

namespace py = pybind11;

// type aliases
namespace sift::py {
    // import {pybind11}
    namespace py = pybind11;
    // get the special {pybind11} literals
    using namespace py::literals;

    // sizes of things
    using size_t = std::size_t;
    // strings
    using string_t = std::string;
}


// end of file
