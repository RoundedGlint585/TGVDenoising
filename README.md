![Build Status](https://travis-ci.org/DanonOfficial/TGVDenoising.svg?branch=master)
# TGV Image Denoising

This code repository is an implementation of the total generalized variation method based on this [paper](https://pdfs.semanticscholar.org/3cdf/b982d5f5c926f9ee257ee7d391ff716e08e6.pdf?_ga=2.99932824.785502720.1554466187-1112687837.1554466187)

## Getting Started

To add

## Platforms ##

  * Linux
  
### Requirements

These are the base requirements to build

  * Cmake
  * A C++17-standard-compliant compiler

## Installing 

```bash
  mkdir build
  cd build
  cmake -D BUILD_RELEASE:BOOL=true ..
  cmake --build .
```
## Running the tests

Compile with Cmake flag -D BUILD_RELEASE:BOOL=false
```bash
  mkdir build
  cd build
  cmake -D BUILD_RELEASE:BOOL=false ..
  cmake --build .
```

## To see results

As far it is not completed work, you can checkout partially result.
Clone code, build release (just  mkdir build && cd build && cmake .. && cmake --build)
and start it. After 4000 iterations from data/* pictures you will get result.png

## Authors

* **Daniil Smolyakov** - *Initial work and CPU based code* - [DanonOfficial](https://github.com/DanonOfficial)


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details


