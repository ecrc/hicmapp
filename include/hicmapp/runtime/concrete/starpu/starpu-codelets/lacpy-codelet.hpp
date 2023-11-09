#ifndef HICMAPP_LACPY_CODELET_HPP
#define HICMAPP_LACPY_CODELET_HPP

#include <hicmapp/runtime/interface/StarpuCodelet.hpp>
#include <hicmapp/runtime/concrete/starpu/factory.hpp>

namespace hicmapp::runtime {
        template<typename T>
        class LacpyCodelet : public StarpuCodelet {

        public:
            LacpyCodelet();

            starpu_codelet *GetCodelet() override;

            ~LacpyCodelet() = default;

        private:
            static bool Register();

            static const bool registered_;

            struct starpu_codelet cl_lacpy{};

            static void cl_lacpy_func(void *descr[], void *cl_arg);
        };

    }
#endif //HICMAPP_LACPY_CODELET_HPP
