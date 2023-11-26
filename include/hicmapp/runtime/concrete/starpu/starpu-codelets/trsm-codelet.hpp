#include <hicmapp/runtime/interface/StarpuCodelet.hpp>
#include <hicmapp/runtime/concrete/starpu/factory.hpp>

#ifndef HICMAPP_TRSM_CODELET_HPP
#define HICMAPP_TRSM_CODELET_HPP
namespace hicmapp {
    namespace runtime {
        template<typename T>
        class TrsmCodelet : public StarpuCodelet {

        public:
            TrsmCodelet();

            starpu_codelet *GetCodelet() override;

            ~TrsmCodelet() = default;

        private:
            static bool Register();

            static const bool registered_;

            struct starpu_codelet cl_trsm{};

            static void cl_trsm_func(void *descr[], void *cl_arg);
        };

    }
}
#endif //HICMAPP_TRSM_CODELET_HPP
