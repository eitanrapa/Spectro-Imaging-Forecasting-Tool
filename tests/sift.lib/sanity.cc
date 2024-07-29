// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// the sift development team
// (c) 2023-2024 all rights reserved


// get the {sift} version
#include <sift/version.h>


// the driver
int main(int argc, char *argv[])
{

    // get the {sift} version
    auto version = sift::version::version();

    // say something
    channel
        // the static version, straight from the headers
        << "   static: "
        << sift::version::major << "."
        << sift::version::minor << "."
        << sift::version::micro << "."
        // the dynamic version, from the library
        << "  dynamic: "
        << version.major << "."
        << version.minor << "."
        << version.micro << "."
        << version.revision << "."

    // all done
    return 0;
}


// end of file
