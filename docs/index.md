# What is HiCMA++
C++ implementation of KAUST ECRC HiCMA library.
The original C version of HiCMA : https://github.com/ecrc/hicma

# Features
* GEMM 
* Cholesky Factorization
* Shared and Distributed Memory Models
* Dynamic Runtime System

# Installation

Installation requires `CMake` of version 3.21.2 at least. To build HiCMA++, follow these instructions:

1.  Get HiCMA++ from git repository
```
git clone git@github.com:ecrc/hicmapp
```

2.  Go into HiCMA++ folder
```
cd hicmapp
```

3.  Run the Configuration Script
```
./config.sh
```

4.  You can also choose whether to build x86 support or CUDA support.
```
./config.sh -t -e (-c for CUDA) (-s for SYCL)
```

5.  Build HiCMA++
```
./clean-build.sh
```

6.  Build local documentation (optional)
```
cd bin && make docs
```

7.  Install HiCMA++
```
make install
```
8. Add line to your .bashrc file to use HiCMA++ as a library.
```
        export PKG_CONFIG_PATH=/path/to/install:$PKG_CONFIG_PATH
```
    
Now you can use `pkg-config` executable to collect compiler and linker flags for HiCMA++.
