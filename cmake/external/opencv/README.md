This directory contains a subset of the files in the OpenCV repo at the 4.5.4 tag. In particular:
- https://github.com/opencv/opencv/tree/4.5.4/platforms/ios/cmake/Toolchains
- https://github.com/opencv/opencv/tree/4.5.4/platforms/ios/cmake/Modules/Platform
- https://github.com/opencv/opencv/blob/4.5.4/COPYRIGHT
- https://github.com/opencv/opencv/blob/4.5.4/LICENSE

We duplicate them in this repo in order to use the OpenCV-specific CMake toolchain files for iOS builds.
It's a hacky solution that works for now.
