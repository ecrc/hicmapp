#ifndef HICMAPP_RUNTIME_HICMA_COMMUNICATOR_HPP
#define HICMAPP_RUNTIME_HICMA_COMMUNICATOR_HPP

#ifdef HICMAPP_USE_MPI
#include <hicmapp/runtime/communicator/mpi/Communicator.hpp>
#else
#include <hicmapp/runtime/communicator/serial/Communicator.hpp>
#endif

#endif //HICMAPP_RUNTIME_HICMA_COMMUNICATOR_HPP

