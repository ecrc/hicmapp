#include <hicmapp/runtime/interface/StarpuCodelet.hpp>
#include <hicmapp/runtime/concrete/starpu/factory.hpp>

#ifndef HICMAPP_DGYTLRDIAG_CODELET_HPP
#define HICMAPP_DGYTLRDIAG_CODELET_HPP
namespace hicmapp {
    namespace runtime {
        template<typename T>
        class GenerateDgytlrDiagonalCodelet : public StarpuCodelet {

        public:
            GenerateDgytlrDiagonalCodelet();

            starpu_codelet *GetCodelet() override;

            ~GenerateDgytlrDiagonalCodelet() = default;

        private:
            static bool Register();

            static const bool registered_;

            struct starpu_codelet cl_dgytlr_diag{};

            static void cl_dgytlr_diag_func(void *descr[], void *cl_arg);
        };

    }
}
#endif //HICMAPP_DGYTLRDIAG_CODELET_HPP
