#include <hicmapp/runtime/interface/StarpuCodelet.hpp>
#include <hicmapp/runtime/concrete/starpu/factory.hpp>

#ifndef HICMAPP_GENERATE_CODELET_HPP
#define HICMAPP_GENERATE_CODELET_HPP
namespace hicmapp {
    namespace runtime {
        template<typename T>
        class GenerateCodelet : public StarpuCodelet {

        public:
            GenerateCodelet();

            starpu_codelet *GetCodelet() override;

            ~GenerateCodelet() = default;

        private:

            static bool Register();

            static const bool registered_;

            struct starpu_codelet cl_dhagdm{};

            static void cl_dhagdm_cpu_func(void *descr[], void *cl_arg);
        };

    }
}
#endif //HICMAPP_GENERATE_CODELET_HPP
