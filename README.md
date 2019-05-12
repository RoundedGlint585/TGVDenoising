![Build Status](https://travis-ci.org/DanonOfficial/TGVDenoising.svg?branch=master)
# TGV Image Denoising

This code repository is an implementation of the total generalized variation method based on this [paper](https://pdfs.semanticscholar.org/3cdf/b982d5f5c926f9ee257ee7d391ff716e08e6.pdf?_ga=2.99932824.785502720.1554466187-1112687837.1554466187)

## Getting Started

To add

## Platforms ##

  * Linux (Tested)
    
### Requirements

These are the base requirements to build

  * Cmake
  * A C++17-standard-compliant compiler
  * For GPU version you actually need something, that support OpenCL 1.2 (so basically its not always GPU)

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

## How to use

Current solution based on command line interface with keys

| Key                  | Purpose                                 | Default Value |
| :------------------- | :-------------------------------------- | :------------ |
| -c                   | Use CPU                                 | False         |
| -g                   | Use GPU                                 | True          |
| -n  \<Num\>          | Amount of Iterations                    | 1000          |
| -p  \<Folder Path\>  | Path to folder with data                | data          |
| -a \<Num\>           | Index of gpu if there are a more than 1 | 0             |
| -r \<File name\>     | Name of the result file                 | result        |
| -i \<Num\>           | amount of images for GPU                | 10            |
   
So just start program with these keys

###Examples
./TGV -g -n 400 -p "data" -a "0" -r "resultFileName" -i 14

./TGV -p "data" -a "0" -r "DenoisedImage" -i 6
## Authors

* **Daniil Smolyakov** - *Initial work and CPU based code* - [DanonOfficial](https://github.com/DanonOfficial)


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details


