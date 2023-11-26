#ifndef HICMAPP_RUNTIME_RUNTIME_FACTORY_HPP
#define HICMAPP_RUNTIME_RUNTIME_FACTORY_HPP

#include <hicmapp/runtime/interface/RunTimeInterface.hpp>

#ifdef HICMAPP_STARPU
#include <hicmapp/runtime/concrete/starpu/starpu.hpp>
#else
#include <hicmapp/runtime/concrete/default/default_runtime.hpp>
#endif
namespace hicmapp::runtime {
        /***
         * Runtime factory to return runtime instance. This would be extended and refactored if more runtimes are supported
         * @tparam T
         */
        template<typename T>
        class RunTimeFactory {

        public:
            /***
             * Create an instance of a runtime object
             * @param aHardware HicmaHardware for initialization of runtime
             * @return
             */
            static RunTimeInterface<T> *CreateRunTimeInstance(hicmapp::runtime::HicmaHardware& aHardware) {

#ifdef HICMAPP_STARPU
                    return new StarPu<T>(aHardware);
#else
                    return new DefaultRuntime<T>(aHardware);

#endif
            }
        };
        HICMAPP_INSTANTIATE_CLASS(RunTimeFactory);
    }
#endif //HICMAPP_RUNTIME_RUNTIME_FACTORY_HPP