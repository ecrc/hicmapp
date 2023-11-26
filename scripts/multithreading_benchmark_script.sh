#!/bin/bash
#
# @copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
#                     All rights reserved.
#
if [[ $# -eq 0 ]] ; then
    echo 'This script needs a single argument that is the hicmapp binary to benchmark'
    exit 0
fi

acc="1e-8"

Threads=(1 4 8 16 32 64)

export HICMAPP_VERBOSE=ON
TileCount=(8)

for threads in ${Threads[@]}; do
  cat /dev/null > benchmark_ts1024_${threads}.csv
  export MKL_NUM_THREADS=1
  for tile_count in ${TileCount[@]}; do
        $1 $tile_count $acc 1024 1 $threads >> benchmark_ts1024_${threads}.csv
        unset HICMAPP_VERBOSE
  done
done
