#include <hicmapp/runtime/interface/StarpuCodelet.hpp>
#include <hicmapp/runtime/concrete/starpu/factory.hpp>

#ifndef HICMAPP_DGYTLR_CODELET_HPP
#define HICMAPP_DGYTLR_CODELET_HPP
namespace hicmapp {
    namespace runtime {
        template<typename T>
        class GenerateDgytlrCodelet : public StarpuCodelet {

        public:
            GenerateDgytlrCodelet();

            starpu_codelet *GetCodelet() override;

            ~GenerateDgytlrCodelet() = default;

        private:
            static bool Register();

            static const bool registered_;

            struct starpu_codelet cl_dgytlr{};

            static void cl_dgytlr_func(void *descr[], void *cl_arg);
        };

    }
}
#endif //HICMAPP_DGYTLR_CODELET_HPP
