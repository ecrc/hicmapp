#ifndef HICMAPP_RUNTIME_MPI_HICMA_COMMUNICATOR_HPP
#define HICMAPP_RUNTIME_MPI_HICMA_COMMUNICATOR_HPP

#include <mpi.h>

namespace hicmapp {
    namespace runtime {

        class HicmaCommunicator {
        public:
            HicmaCommunicator() = default;

            explicit HicmaCommunicator(MPI_Comm aComm) : mCommunicator{aComm} {

            }

            [[nodiscard]] MPI_Comm
            GetMPICommunicatior() const {
                return mCommunicator;
            }

            void
            SetMPICommunicator(MPI_Comm aCommunicator) {
                mCommunicator = aCommunicator;
            }

        private:
            /*** MPI Communicator */
            MPI_Comm mCommunicator = MPI_COMM_WORLD;
        };
    }
}

#endif //HICMAPP_RUNTIME_MPI_HICMA_COMMUNICATOR_HPP
