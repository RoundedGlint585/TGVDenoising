#!/bin/bash
CMAKE_VERSION=3.14.0
CMAKE_VERSION_DIR=v3.14

CMAKE_OS=Linux-x86_64
CMAKE_TAR=cmake-$CMAKE_VERSION-$CMAKE_OS.tar.gz
CMAKE_URL=http://www.cmake.org/files/$CMAKE_VERSION_DIR/$CMAKE_TAR
CMAKE_DIR=$(pwd)/cmake-$CMAKE_VERSION

wget --quiet $CMAKE_URL
mkdir -p $CMAKE_DIR
tar --strip-components=1 -xzf $CMAKE_TAR -C $CMAKE_DIR
export PATH=$CMAKE_DIR/bin:$PATH

apt-get install libgtest-dev
mkdir /usr/src/gtest/build
cmake /usr/src/gtest/
make /usr/src/gtest/build
cp /usr/src/gtest/build/libgtest* /usr/lib/

