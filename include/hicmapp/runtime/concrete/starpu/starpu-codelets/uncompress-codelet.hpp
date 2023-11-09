#include <hicmapp/runtime/interface/StarpuCodelet.hpp>
#include <hicmapp/runtime/concrete/starpu/factory.hpp>

#ifndef HICMAPP_UNCOMPRESS_CODELET_HPP
#define HICMAPP_UNCOMPRESS_CODELET_HPP
namespace hicmapp::runtime {
        template<typename T>
        class UncompressCodelet : public StarpuCodelet {

        public:
            UncompressCodelet();

            starpu_codelet *GetCodelet() override;

            ~UncompressCodelet() = default;

        private:
            static bool Register();

            static const bool registered_;

            struct starpu_codelet cl_uncompress{};

            static void cl_uncompress_func(void *descr[], void *cl_arg);
        };

    }
#endif //HICMAPP_UNCOMPRESS_CODELET_HPP
