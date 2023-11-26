#include <hcorepp/operators/interface/Tile.hpp>
#include <hicmapp/runtime/concrete/starpu/starpu-codelets/generate_codelet.hpp>
#include <hicmapp/common/definitions.h>
#include <hicmapp/runtime/concrete/starpu/factory.hpp>
#include "hicmapp/tile-operations/TileOperations.hpp"
#include "hcorepp/operators/interface/TilePacker.hpp"

using namespace hicmapp::runtime;

namespace hicmapp::runtime {

    template<typename T> const bool GenerateCodelet<T>::registered_ = GenerateCodelet<T>::Register();

    template<typename T>
    bool GenerateCodelet<T>::Register() {
        static auto maker = new hicmapp::runtime::CodeletMaker<GenerateCodelet, T>(CodeletType::GENERATE_DENSE_DATA);
        return true;
    }

    template<typename T>
    starpu_codelet *GenerateCodelet<T>::GetCodelet() {
        return &this->cl_dhagdm;
    }

    template<typename T>
    GenerateCodelet<T>::GenerateCodelet() {
        cl_dhagdm = {.where=STARPU_CPU, .type = STARPU_SEQ,
                .cpu_funcs={cl_dhagdm_cpu_func},
                .cuda_funcs={}, .cuda_flags={(0)}, .nbuffers=((2)),
                .model={}, .name="dhagdm"};
    }

    template<typename T>
    void GenerateCodelet<T>::cl_dhagdm_cpu_func(void **descr, void *cl_arg) {
        auto* metadata = (hcorepp::operators::TileMetadata *) STARPU_VARIABLE_GET_PTR(descr[0]);
        auto *tile_data = (T *) STARPU_MATRIX_GET_PTR(descr[1]);

        size_t tile_col_index;
        size_t tile_row_index;

        starpu_codelet_unpack_args(cl_arg, &tile_row_index,
                                   &tile_col_index);

        hcorepp::kernels::RunContext &context = hcorepp::kernels::ContextManager::GetInstance().GetContext();

        auto *tile = static_cast<hcorepp::operators::DenseTile<T> *>(hcorepp::operators::TilePacker<T>::PackTile(*metadata,
                                                                                                          tile_data,
                                                                                                          context));

        hicmapp::operations::TileOperations<T>::GenerateDenseTile(*tile,
                                                                  tile_row_index,
                                                                  tile_col_index);
    }

    HICMAPP_INSTANTIATE_CLASS(GenerateCodelet)
}

