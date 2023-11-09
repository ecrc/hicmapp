
#ifndef HICMAPP_RUNTIME_RUNTIME_SINGLETON_HPP
#define HICMAPP_RUNTIME_RUNTIME_SINGLETON_HPP

#include <hicmapp/runtime/interface/RunTimeInterface.hpp>
#include <hicmapp/runtime/interface/RunTimeFactory.hpp>
#include <hicmapp/runtime/interface/HicmaHardware.hpp>

namespace hicmapp::runtime {
        template<typename T>
        /** Singleton Class for Runtime Instance to be used across the project */
        class RunTimeSingleton {
        public:
            /***
             * Getter for Runtime Instance. Throws exception if not initialized
             * @return
             */
            static RunTimeInterface<T> *GetRunTimeInstance();

            /***
             * Initialize Runtime Instance. Currently only Default and StarPu are supported
             * @param aHardware Hicma Hardware initialization
             */
            static void setRunTimeInstance(hicmapp::runtime::HicmaHardware& aHardware);

        private:
            /***
             * Private constructor
             */
            RunTimeSingleton();

            /**
             * Singleton member
             */
            static hicmapp::runtime::RunTimeInterface<T> *runtime_instance;
        };
    }

#endif //HICMAPP_RUNTIME_RUNTIME_SINGLETON_HPP