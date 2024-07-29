// -*- c++ -*-
//
// the sift development team
// (c) 2023-2024 all rights reserved


// code guard
#pragma once

// the {project.name} namespace
namespace sift::py {

    // bindings of opaque types
    void opaque(py::module &);
    // exceptions
    void exceptions(py::module &);
    // version info
    void version(py::module &);
    // CartesianPoint
    void cartesianPoint(py::module &);
    // GeodeticPoint
    void geodeticPoint(py::module &);
    // TriaxialEllipsoid
    void triaxialEllipsoid(py::module &);

}


// end of file
