#include <hicmapp/runtime/interface/StarpuCodelet.hpp>
#include <hicmapp/runtime/concrete/starpu/factory.hpp>

#ifndef HICMAPP_SYRK_CODELET_HPP
#define HICMAPP_SYRK_CODELET_HPP
namespace hicmapp {
    namespace runtime {
        template<typename T>
        class SyrkCodelet : public StarpuCodelet {

        public:
            SyrkCodelet();

            starpu_codelet *GetCodelet() override;

            ~SyrkCodelet() = default;

        private:

            static bool Register();

            static const bool registered_;

            struct starpu_codelet cl_syrk{};

            static void cl_syrk_func(void *descr[], void *cl_arg);
        };

    }
}
#endif //HICMAPP_SYRK_CODELET_HPP
