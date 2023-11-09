# set minimum cmake version
cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
cmake_policy(SET CMP0048 NEW)
# Project Options.
option(USE_CUDA "Use Cuda, if available" false)
option(HICMAPP_USE_TIMER "Enable Timer" false)
option(HICMAPP_BUILD_TESTS "Option to enable building tests" ON)
option(HICMAPP_BUILD_EXAMPLES "Option to enable building examples" ON)
option(HICMAPP_BUILD_DOCS "Build documentation in docs directory" ON)
option(USE_MPI "Use MPI, if available" OFF)
option(USE_OMP "Use OMP, if available" OFF)

if (NOT DEFINED HICMAPP_SCHED)
    set(HICMAPP_SCHED "Default")
endif()
string(TOLOWER "${HICMAPP_SCHED}" HICMAPP_SCHED_LOWER)

#enable_language(CXX)
# Add Module Paths(Own cmake modules/hcore++)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/libs/hcorepp/cmake)
#list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/libs/stars-h/cmake)
set(CMAKE_CXX_FLAGS_RELEASE "-g -fPIC -fno-common -O0 -Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-g -fPIC -fno-common -O0 -Wall")
set(HCOREPP_BUILD_DOCS OFF)   # This is to prevent having multiple targets with the
                              #name docs and to make each library responsible for its own documentation
if(HICMAPP_USE_TIMER)
    add_definitions(-DHICMAPP_USE_TIMER)
endif ()

# Toolchain selection
# Check if only one toolchain is set to true at a time
if (USE_CUDA AND (USE_SYCL OR USE_OMP))
    message(FATAL_ERROR "Only one toolchain USE_CUDA, USE_SYCL, and USE_OMP can be set to true at a time.")
elseif (USE_SYCL AND (USE_CUDA OR USE_OMP))
    message(FATAL_ERROR "Only one toolchain USE_CUDA, USE_SYCL, and USE_OMP can be set to true at a time.")
elseif (USE_OMP AND (USE_CUDA OR USE_SYCL))
    message(FATAL_ERROR "Only one toolchain USE_CUDA, USE_SYCL, and USE_OMP can be set to true at a time.")
endif()

if((${USE_OMP}) AND NOT (${HICMAPP_SCHED_LOWER} EQUAL "default"))
    message(FATAL_ERROR "Only one runtime/resource management toolchain can be set to true at a time. STARPU and OMP can't be used simultaneously")
endif()

if (USE_CUDA)
        add_definitions(-DUSE_CUDA=TRUE)
        enable_language(CUDA)
        include(toolchains/CudaToolchain)
        set(BLA_VENDOR NVHPC)
        list(APPEND STARPU_COMPONENT_LIST "CUDA")
elseif(USE_SYCL)
    add_definitions(-DUSE_SYCL=TRUE)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}  -L$ENV{MKLROOT}/lib/intel64 -fsycl -lsycl -lOpenCL -lpthread -lm -ldl")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -qmkl=parallel -DSYCL_DEVICES_cpu -O3 -Wno-return-type-c-linkage -fsycl-unnamed-lambda -qopenmp")
    add_definitions(-DMKL_ILP64)
    add_definitions(-DLAPACK_ILP64)
    add_definitions(-DLAPACK_COMPLEX_CPP)
    set(CMAKE_C_COMPILER icx)
    set(CMAKE_CXX_COMPILER icpx)
    set(CMAKE_CXX_COMPILER_ID IntelLLVM)
elseif(USE_OMP)
    add_definitions(-DUSE_OMP=TRUE)
else ()
    include(toolchains/GccToolchain)
endif ()

# Add HCorePP library
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/libs/hcorepp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/libs/hcorepp/include)
list(APPEND LIBS hcorepp)


# Project settings.
project(hicmapp VERSION 1.0.0 DESCRIPTION "HICMA C++ implementation")
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/libs/hcorepp/cmake)

if (HICMAPP_USE_MPI)
    add_definitions(-DSTARPU_USE_MPI)
    add_definitions(-DHICMAPP_USE_MPI=TRUE)
    find_package(MPI REQUIRED)
    list(APPEND STARPU_COMPONENT_LIST "MPI")
endif ()

set(USE_MPI ${HICMAPP_USE_MPI})

# HICMA depends on a runtime
# -------------------------------
if ("${HICMAPP_SCHED_LOWER}" STREQUAL "starpu")
    include(ImportSTARPU)
    add_definitions(-DHICMAPP_STARPU)
    if(USE_CUDA)
        add_definitions(-DSTARPU_USE_CUDA)
    endif()
else()
    set(HICMAPP_SCHED_LOWER "default")
endif()

#add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/libs/stars-h)
#include_directories(${CMAKE_CURRENT_SOURCE_DIR}/libs/stars-h/include)
#list(APPEND LIBS starsh)

include(ImportSTARSH)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/prerequisites)

add_subdirectory(src)

# Add tests if enabled
if (${HICMAPP_BUILD_TESTS})
    message(STATUS "Building Tests")
    include(ImportCatch2)
    include(CTest)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tests)
    enable_testing()

endif ()

# Add examples if enabled
if (${HICMAPP_BUILD_EXAMPLES})
    message(STATUS "Building Examples")
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/examples)
endif ()

message("---------------------------BUILDING HiCMA DOCS")
find_package(Doxygen)
if (DOXYGEN_FOUND)
    add_subdirectory("docs")
else ()
    message(STATUS "Doxygen NOT found, skipping it")
endif ()

# Installation actions
install(DIRECTORY include/${PROJECT_NAME} DESTINATION include)
## Install cmake find package.
include(CMakePackageConfigHelpers)
write_basic_package_version_file("${CMAKE_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake" COMPATIBILITY ExactVersion)
install(
        FILES
        "${CMAKE_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
        DESTINATION lib/cmake/${PROJECT_NAME}
)

configure_file(${PROJECT_NAME}Config.cmake.in
        ${PROJECT_NAME}Config.cmake @ONLY)

install(
        FILES
        "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
        DESTINATION lib/cmake/${PROJECT_NAME}
)

install(
        DIRECTORY
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
        DESTINATION lib/cmake/${PROJECT_NAME}/Modules
)

## Generate pkg-config file
configure_file(package.pc.in
        lib/pkgconfig/${PROJECT_NAME}.pc @ONLY)
install(
        FILES
        "${PROJECT_BINARY_DIR}/lib/pkgconfig/${PROJECT_NAME}.pc"
        DESTINATION lib/pkgconfig/
)


##################
# Release source #
##################
set(CPACK_SOURCE_GENERATOR "TGZ")
set(CPACK_PACKAGE_NAME "${PROJECT_NAME}")
set(CPACK_PACKAGE_DESCRIPTION_FILE ${CMAKE_CURRENT_SOURCE_DIR}/README.md)
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Hicma++ library ")
set(CPACK_PACKAGE_VERSION "${${PROJECT_NAME}_VERSION}")
set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}")
set(CPACK_SOURCE_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}")
set(CPACK_PACKAGE_VENDOR "KAUST")
set(CPACK_PACKAGE_CONTACT "hatem.ltaief@kaust.edu.sa")
set(CPACK_RESOURCE_FILE_README ${CMAKE_CURRENT_SOURCE_DIR}/README.md)
set(CPACK_RESOURCE_FILE_LICENSE ${CMAKE_CURRENT_SOURCE_DIR}/LICENSE)
set(CPACK_SOURCE_IGNORE_FILES "bin;.git;.gitmodules;Jenkinsfile")
include(CPack)
