/**
 * @copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
 */

#ifndef HICMAPP_USE_MPI
#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>
#endif


#ifdef HICMAPP_USE_MPI
#define CATCH_CONFIG_RUNNER

#include <catch2/catch_all.hpp>

#include <mpi.h>

bool is_init = false;

int main(int argc, char *argv[]) {

    if (!is_init) {
        int required = 0;
        MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &required);
        is_init = true;
    }
    int result = Catch::Session().run(argc, argv);
    MPI_Finalize();
    return result;
}

#endif
