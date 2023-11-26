
#include <hicmapp/runtime/interface/StarpuCodelet.hpp>
#include <hicmapp/runtime/concrete/starpu/factory.hpp>

#ifndef HICMAPP_GENERATE_COMPRESSED_DATA_CODELET_HPP
#define HICMAPP_GENERATE_COMPRESSED_DATA_CODELET_HPP

namespace hicmapp {
    namespace runtime {
        template<typename T>
        class GenerateCompressedDataCodelet : public StarpuCodelet {

        public:
            GenerateCompressedDataCodelet();

            starpu_codelet *GetCodelet() override;

            ~GenerateCompressedDataCodelet() = default;

        private:

            static bool Register();

            static const bool registered_;

            struct starpu_codelet cl_dhagcm{};

            static void cl_dhagcm_cpu_func(void *descr[], void *cl_arg);
        };

    }
}
#endif //HICMAPP_GENERATE_COMPRESSED_DATA_CODELET_HPP
