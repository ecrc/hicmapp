#include <hicmapp/runtime/interface/StarpuCodelet.hpp>
#include <hicmapp/runtime/concrete/starpu/factory.hpp>

#ifndef HICMAPP_POTRF_CODELET_HPP
#define HICMAPP_POTRF_CODELET_HPP
namespace hicmapp {
    namespace runtime {
        template<typename T>
        class PotrfCodelet : public StarpuCodelet {

        public:
            PotrfCodelet();

            starpu_codelet *GetCodelet() override;

            ~PotrfCodelet() = default;

        private:
            static bool Register();

            static const bool registered_;

            struct starpu_codelet cl_potrf{};

            static void cl_potrf_func(void *descr[], void *cl_arg);
        };

    }
}
#endif //HICMAPP_POTRF_CODELET_HPP
