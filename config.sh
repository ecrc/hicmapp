#!/bin/bash

# Set variables and default values
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

INSTALL_PREFIX=$PWD/installdir/hicmapp
PROJECT_SOURCE_DIR=$(dirname "$0")
BUILDING_TESTS="OFF"
BUILDING_EXAMPLES="OFF"
VERBOSE=OFF
BUILD_TYPE="RELEASE"
USE_CUDA="OFF"
USE_MPI="OFF"
HICMAPP_SCHED="Default"


# Parse command line options
while getopts ":tevhi:dscmoxr:" opt; do
  case $opt in
    i) ##### Define installation path  #####
       echo -e "${YELLOW}Installation path set to $OPTARG.${NC}"
       INSTALL_PREFIX=$OPTARG
       ;;
    t) ##### Building tests enabled #####
      echo -e "${GREEN}Building tests enabled.${NC}"
      BUILDING_TESTS="ON"
      ;;
    e) ##### Building examples enabled #####
      echo -e "${GREEN}Building examples enabled.${NC}"
      BUILDING_EXAMPLES="ON"
      ;;
    c)##### Using cuda enabled #####
        echo -e "${GREEN}Cuda enabled ${NC}"
        USE_CUDA=ON
        ;;
    s) ##### Using sycl enabled #####
      echo -e "${YELLOW}Sycl enabled ${NC}"
      USE_SYCL=ON
      ;;
    m)##### Using MPI enabled #####
        echo -e "${GREEN}MPI enabled ${NC}"
        USE_MPI=ON
        ;;
    o)##### Using OMP enabled #####
        echo -e "${GREEN}OMP enabled ${NC}"
        USE_OMP=ON
        ;;
    v) ##### printing full output of make #####
      echo -e "${YELLOW}printing make with details.${NC}"
      VERBOSE=ON
      ;;
    d)##### Using debug mode to build #####
      echo -e "${RED}Debug mode enabled ${NC}"
      BUILD_TYPE="DEBUG"
      ;;
    x) ##### Using Timer for debugging enabled #####
      echo -e "${BLUE}Timer for Debugging enabled ${NC}"
      HICMAPP_USE_TIMER=ON
      ;;
    r) ##### Using Starpu runtime enabled #####
      echo -e "${YELLOW}Selected $OPTARG as runtime ${NC}"
      HICMAPP_SCHED=$OPTARG
      ;;
    \?) ##### using default settings #####
      BUILDING_TESTS="OFF"
      BUILDING_EXAMPLES="OFF"
      VERBOSE=OFF
      BUILD_TYPE="RELEASE"
      USE_CUDA="OFF"
      USE_MPI="OFF"
      USE_SYCL="OFF"
      USE_OMP="OFF"
      HICMAPP_SCHED="Default"
      INSTALL_PREFIX=$PWD/installdir/hicmapp

      echo -e "${RED}Building tests disabled.${NC}"
      echo -e "${RED}Building examples disabled.${NC}"
      echo -e "${BLUE}Installation path set to $INSTALL_PREFIX.${NC}"
      ;;
    :) ##### Error in an option #####
      echo "Option $OPTARG requires parameter(s)"
      exit 0
      ;;
    h) ##### Prints the help #####
      echo "Usage of $(basename "$0"):"
      echo ""
      printf "%20s %s\n" "-t :" "to enable building tests."
      echo ""
      printf "%20s %s\n" "-e :" "to enable building examples."
      echo ""
      printf "%20s %s\n" "-i [path] :" "specify installation path."
      printf "%20s %s\n" "" "default = /hicmapp/installdir/hicmapp"
      echo ""
      exit 1
      ;;
    esac
done

echo -e "${BLUE}Installation path set to $INSTALL_PREFIX.${NC}"

if [ -z "$BUILDING_TESTS" ]; then
  BUILDING_TESTS="OFF"
  echo -e "${RED}Building tests disabled.${NC}"
fi

if [ -z "$BUILDING_EXAMPLES" ]; then
  BUILDING_EXAMPLES="OFF"
  echo -e "${RED}Building examples disabled.${NC}"
fi

if [ -z "$BUILD_TYPE" ]; then
  BUILD_TYPE="RELEASE"
  echo -e "${GREEN}Building in release mode${NC}"
fi
if [ -z "$USE_CUDA" ]; then
  USE_CUDA="OFF"
  echo -e "${RED}Using CUDA disabled${NC}"
fi

if [ -z "$USE_MPI" ]; then
  USE_MPI="OFF"
  echo -e "${RED}Using MPI disabled${NC}"
fi

if [ -z "$USE_OMP" ]; then
  USE_OMP="OFF"
  echo -e "${RED}Using OMP disabled${NC}"
fi

if [ -z "$USE_SYCL" ]; then
  USE_SYCL="OFF"
  echo -e "${RED}Using SYCL disabled${NC}"
fi

if [ -z "$HICMAPP_USE_TIMER" ]; then
  HICMAPP_USE_TIMER="OFF"
  echo -e "${RED}Using Timer for debugging disabled${NC}"
fi


echo ""
echo -e "${YELLOW}Use -h to print the usages of hicmapp flags.${NC}"
echo ""
rm -rf bin/
mkdir -p bin/installdir

cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DHICMAPP_INSTALL_PREFIX="$INSTALL_PREFIX" \
  -DHICMAPP_BUILD_TESTS="$BUILDING_TESTS" \
  -DHICMAPP_BUILD_EXAMPLES="$BUILDING_EXAMPLES" \
  -DCMAKE_VERBOSE_MAKEFILE:BOOL=$VERBOSE \
  -DHICMAPP_SCHED=$HICMAPP_SCHED \
  -DUSE_OMP="$USE_OMP" \
  -DUSE_CUDA="$USE_CUDA" \
  -DUSE_SYCL="$USE_SYCL" \
  -DHICMAPP_USE_MPI="$USE_MPI" \
  -H"${PROJECT_SOURCE_DIR}" \
  -B"${PROJECT_SOURCE_DIR}/bin" \
  -G "Unix Makefiles" \
  -DHICMAPP_USE_TIMER=$HICMAPP_USE_TIMER
