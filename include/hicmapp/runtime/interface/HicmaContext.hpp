#ifndef HICMAPP_RUNTIME_HICMA_CONTEXT_HPP
#define HICMAPP_RUNTIME_HICMA_CONTEXT_HPP

#include <hcorepp/kernels/RunContext.hpp>
#include <hcorepp/kernels/ContextManager.hpp>
#include <hicmapp/runtime/interface/HicmaCommunicator.hpp>

namespace hicmapp::runtime {

        class HicmaContext {
        public:
            /***
             * HicmaContext Default Constructor
             */
            explicit HicmaContext();

            /***
             * HicmaContext Constructor with a specific communicator
             * @param aComm
             */
            explicit HicmaContext(HicmaCommunicator aComm);

            /***
             * Default HicmaContext Destructor
             */
            ~HicmaContext() = default;

            /***
             * Get number of Hcorepp contexts
             * @return
             */
            size_t
            GetNumOfContexts();

            /***
             * Get the main context
             * @return
             */
            const hcorepp::kernels::RunContext&
            GetMainContext();

            /***
             * Get a specific context
             * @param aIdx index of context to fetch
             * @return
             */
            const hcorepp::kernels::RunContext&
            GetContext(size_t aIdx = 0);

            /***
             * Synchronize main context
             */
            void
            SyncMainContext();

            /***
             * Synchronize context at a specific index
             * @param aIdx
             */
            void
            SyncContext(size_t aIdx = 0);

            /***
             * Synchronize all contexts
             */
            void
            SyncAll();

            /***
             * Get HicmaCommunicator member
             * @return
             */
            HicmaCommunicator&
            GetCommunicator();

            /***
             * Set Communicator to be used
             * @param aCommunicator
             */
            void
            SetCommunicator(HicmaCommunicator& aCommunicator);

        private:
            /*** Communicator member */
            HicmaCommunicator mCommunicator;
        };
    }
#endif //HICMAPP_RUNTIME_HICMA_CONTEXT_HPP
