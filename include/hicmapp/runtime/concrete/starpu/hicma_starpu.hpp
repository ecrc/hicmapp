/**
 * @copyright (c) 2017-2022 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
 */
/**
 *
 * @file hicma_starpu.hpp
 *
 * @copyright 2009-2014 The University of Tennessee and The University of
 *                      Tennessee Research Foundation. All rights reserved.
 * @copyright 2012-2016 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 ***
 *
 * @brief Chameleon StarPU runtime header
 *
 * @version 1.0.0
 * @author Mathieu Faverge
 * @author Cedric Castagnede
 * @author Florent Pruvost
 * @date 2011-06-01
 *
 */
#ifndef _HICMA_STARPU_H_
#define _HICMA_STARPU_H_

/* StarPU options */
/* #undef HAVE_STARPU_FXT_PROFILING */
/* #undef HAVE_STARPU_IDLE_PREFETCH */
/* #undef HAVE_STARPU_ITERATION_PUSH */
/* #undef HAVE_STARPU_DATA_WONT_USE */
/* #undef HAVE_STARPU_DATA_SET_COORDINATES */
/* #undef HAVE_STARPU_MALLOC_ON_NODE_SET_DEFAULT_FLAGS */
/* #undef HAVE_STARPU_MPI_DATA_MIGRATE */
/* #undef HAVE_STARPU_MPI_DATA_REGISTER */
/* #undef HAVE_STARPU_MPI_COMM_RANK */
/* #undef HAVE_STARPU_MPI_CACHED_RECEIVE */
/* #undef HAVE_STARPU_MPI_COMM_GET_ATTR */

#if defined(HICMAPP_USE_MPI)
#include <starpu_mpi.h>
#else
#include <starpu.h>
#endif

#include <starpu_profiling.h>

#if defined(USE_CUDA)
#include <starpu_scheduler.h>
#include <starpu_cuda.h>

#include <cublas.h>
#include <starpu_cublas.h>
#if defined(HICMA_USE_CUBLAS_V2)
#include <cublas_v2.h>
#include <starpu_cublas_v2.h>
#endif
#endif

#if defined(HICMA_SIMULATION)
# if !defined(STARPU_SIMGRID)
#  error "Starpu was not built with simgrid support (--enable-simgrid). Can not run Hicma with simulation support."
# endif
#else
# if defined(STARPU_SIMGRID)
#  warning "Starpu was built with simgrid support. Better build Hicma with simulation support (-DHICMA_SIMULATION=YES) NOT SUPPORTED YET."
# endif
#endif

#include <hicmapp/runtime/concrete/starpu/hicma_runtime_workspace.hpp>

typedef struct starpu_conf starpu_conf_t;

/**/

/*
 * MPI Redefinitions
 */
#if defined(HICMAPP_USE_MPI)
#undef STARPU_REDUX
#define starpu_insert_task(...) starpu_mpi_insert_task(MPI_COMM_WORLD, __VA_ARGS__)
#endif

/*
 * cuBlasAPI v2 - StarPU enable the support for cublas handle
 */
#if defined(USE_CUDA) && defined(HICMA_USE_CUBLAS_V2)
#define RUNTIME_getStream(_stream_)                             \
    cublasHandle_t _stream_ = starpu_cublas_get_local_handle();
#else
#define RUNTIME_getStream(_stream_)                             \
    cudaStream_t _stream_ = starpu_cuda_get_local_stream();     \
    cublasSetKernelStream( stream );

#endif

/*
 * Enable codelets names
 */
#if (STARPU_MAJOR_VERSION > 1) || ((STARPU_MAJOR_VERSION == 1) && (STARPU_MINOR_VERSION > 1))
#define CHAMELEON_CODELETS_HAVE_NAME
#endif

#endif /* _HICMA_STARPU_H_ */
