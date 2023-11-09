#include <hicmapp/runtime/interface/StarpuCodelet.hpp>
#include <hicmapp/runtime/concrete/starpu/factory.hpp>

#ifndef HICMAPP_GEMM_CODELET_HPP
#define HICMAPP_GEMM_CODELET_HPP
namespace hicmapp {
    namespace runtime {
        template<typename T>
        class GemmCodelet : public StarpuCodelet {

        public:
            GemmCodelet();

            starpu_codelet *GetCodelet() override;

            ~GemmCodelet() = default;

        private:

            static bool Register();

            static const bool registered_;

            struct starpu_codelet cl_dgemm{};

            static void cl_dgemm_func(void *descr[], void *cl_arg);
        };

    }
}
#endif //HICMAPP_GEMM_CODELET_HPP
