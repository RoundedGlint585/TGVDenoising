#!/bin/bash
apt install libgtest-dev cmake
cd /usr/src/googletest
cmake .
cmake --build . --target install