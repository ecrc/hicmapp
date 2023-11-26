#ifndef HICMAPP_RUNTIME_HICMA_HARDWARE_HPP
#define HICMAPP_RUNTIME_HICMA_HARDWARE_HPP

namespace hicmapp::runtime {

        struct HicmaHardware {
            /***
             * Struct specifying hardware the runtime instance will use during runtime.
             * @param aCPUs
             * @param aGPUs
             * @param aThreadsPerWorker
             */
            HicmaHardware(int aCPUs, int aGPUs, int aThreadsPerWorker) : mCPUs(aCPUs), mGPUs(aGPUs), mThreadsPerWorker(aThreadsPerWorker) {

            }
            int mCPUs = 1;
            int mGPUs = 0;
            int mThreadsPerWorker = -1;
        };
    }

#endif //HICMAPP_RUNTIME_HICMA_HARDWARE_HPP