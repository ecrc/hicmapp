
# Copyright (c) 2017-2023 King Abdullah University of Science and Technology,
# Copyright (c) 2023 by Brightskies inc,
# All rights reserved.
# ExaGeoStat is a software package, provided by King Abdullah University of Science and Technology (KAUST).

# @file ImportSTARPU.cmake
# @brief Find and include STARPU library as a dependency.
# @version 1.0.0
# @author Sameh Abdulah
# @date 2023-03-13

message("")
message("---------------------------------------- StarPU")
message(STATUS "Checking for StarPU")

include(macros/BuildDependency)

if (NOT TARGET STARPU)
    # Try to find STARPU.
    include(FindPkgConfig)
    find_package(PkgConfig QUIET)
    find_package(STARPU 1.4.1 QUIET COMPONENTS ${STARPU_COMPONENT_LIST})

    # If STARPU is found, print its location.
    if (STARPU_FOUND)
        message("   Found StarPU: ${STARPU_LIBRARIES}")
        # If not found, install it.
    else ()
        set(STARPU_DIR  ${PROJECT_SOURCE_DIR}/installdir/_deps/STARPU/)
        # Set the flags to be passed to the build command.
        set(ISCMAKE OFF)
        set(ISGIT ON)
        set(AUTO_GEN ON)

        if (USE_CUDA AND USE_MPI)
            message("Downloading STARPU - MPI CUDA" )
            set(FLAGS --prefix=${PROJECT_SOURCE_DIR}/installdir/_deps/STARPU/  \--enable-cuda  \--disable-opencl  \--enable-shared  \--disable-build-doc  \--disable-export-dynamic  \--enable-mpi)
        elseif(USE_CUDA)
            message("Downloading STARPU - CUDA" )
            set(FLAGS --prefix=${PROJECT_SOURCE_DIR}/installdir/_deps/STARPU/  \--enable-cuda  \--disable-opencl  \--enable-shared  \--disable-build-doc  \--disable-export-dynamic  \--disable-mpi)
        elseif(USE_MPI)
            message("Downloading STARPU - MPI" )
            set(FLAGS --prefix=${PROJECT_SOURCE_DIR}/installdir/_deps/STARPU/  \--disable-cuda  \--disable-opencl  \--enable-shared  \--disable-build-doc  \--disable-export-dynamic  \--enable-mpi)
        else()
            message("Downloading STARPU - SERIAL" )
            set(FLAGS --prefix=${PROJECT_SOURCE_DIR}/installdir/_deps/STARPU/  \--disable-cuda  \--disable-opencl  \--enable-shared  \--disable-build-doc  \--disable-export-dynamic  \--disable-mpi)

        endif()

        BuildDependency(STARPU "https://gitlab.inria.fr/starpu/starpu.git" "starpu-1.4.1"  ${FLAGS} ${ISCMAKE} ${ISGIT} ${AUTO_GEN})

        # Clear the flags.
        set(FLAGS "")
        # Find StarPU after installation.
        unset(STARPU_DIR)
        find_package(STARPU 1.4.1 QUIET COMPONENTS ${STARPU_COMPONENT_LIST})
    endif ()
else ()
    message("   STARPU already included")
endif ()

# Include STARPU headers.
list(APPEND LIBS ${STARPU_LIBRARIES})
link_directories(${STARPU_LIBRARY_DIRS_DEP})
include_directories(${STARPU_INCLUDE_DIRS})
include_directories(${STARPU_INCLUDE_DIRS}/runtime/starpu)
include_directories(${STARPU_INCLUDE_DIRS_DEP})

# Set linker flags.
if (STARPU_LINKER_FLAGS)
    list(APPEND CMAKE_EXE_LINKER_FLAGS "${STARPU_LINKER_FLAGS}")
endif ()
set(CMAKE_REQUIRED_INCLUDES "${STARPU_INCLUDE_DIRS_DEP}")
foreach (libdir ${STARPU_LIBRARY_DIRS_DEP})
    list(APPEND CMAKE_REQUIRED_FLAGS "-L${libdir}")
endforeach ()
set(CMAKE_REQUIRED_LIBRARIES "${STARPU_LIBRARIES_DEP}")

message(STATUS "starpu done")
