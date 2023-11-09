#ifndef HICMAPP_RUNTIME_SERIAL_HICMA_COMMUNICATOR_HPP
#define HICMAPP_RUNTIME_SERIAL_HICMA_COMMUNICATOR_HPP

#include <stdexcept>

namespace hicmapp {
    namespace runtime {
/***
 * Dummy Communicator Class if MPI is disabled
 */
        class HicmaCommunicator {
        public:

        };
    }
}

#endif //HICMAPP_RUNTIME_SERIAL_HICMA_COMMUNICATOR_HPP