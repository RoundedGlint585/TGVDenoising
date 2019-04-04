#!/bin/bash

sudo apt install libgtest-dev build-essential cmake
sudo cmake  .
sudo cmake  /usr/src/googletest --build . --target install