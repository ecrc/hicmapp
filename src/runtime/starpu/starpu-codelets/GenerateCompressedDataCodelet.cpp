#include <hcorepp/operators/interface/Tile.hpp>
#include <hicmapp/runtime/concrete/starpu/starpu-codelets/generate_compressed_data_codelet.hpp>
#include <hicmapp/common/definitions.h>
#include <hicmapp/runtime/concrete/starpu/factory.hpp>
#include "hicmapp/tile-operations/TileOperations.hpp"
#include "hcorepp/kernels/memory.hpp"
#include "hcorepp/operators/interface/TilePacker.hpp"

using namespace hicmapp::runtime;

namespace hicmapp::runtime {

    template<typename T> const bool GenerateCompressedDataCodelet<T>::registered_ = GenerateCompressedDataCodelet<T>::Register();

    template<typename T>
    bool GenerateCompressedDataCodelet<T>::Register() {
        static auto maker = new hicmapp::runtime::CodeletMaker<GenerateCompressedDataCodelet, T>(
                CodeletType::GENERATE_COMPRESSED_DATA);
        return true;
    }

    template<typename T>
    starpu_codelet *GenerateCompressedDataCodelet<T>::GetCodelet() {
        return &this->cl_dhagcm;
    }

    template<typename T>
    GenerateCompressedDataCodelet<T>::GenerateCompressedDataCodelet() {
        cl_dhagcm = {.where=STARPU_CPU, .type = STARPU_SEQ,
                .cpu_funcs={cl_dhagcm_cpu_func},
                .cuda_funcs={}, .cuda_flags={(0)}, .nbuffers=((2)),
                /// @Todo: add the model used in old hicma..
                .model={}, .name="dhagcm"};
    }

    template<typename T>
    void GenerateCompressedDataCodelet<T>::cl_dhagcm_cpu_func(void **descr, void *cl_arg) {
        auto *metadata = (hcorepp::operators::TileMetadata *) STARPU_VARIABLE_GET_PTR(descr[0]);
        auto *tile_data = (T *) STARPU_MATRIX_GET_PTR(descr[1]);

        size_t tile_col_index;
        size_t tile_row_index;
        hcorepp::operators::CompressionParameters aParams;

        starpu_codelet_unpack_args(cl_arg, &tile_row_index,
                                   &tile_col_index, &aParams);

        hcorepp::kernels::RunContext &context = hcorepp::kernels::ContextManager::GetInstance().GetContext();

        auto *tile = hcorepp::operators::TilePacker<T>::PackTile(*metadata, tile_data, context);

        hicmapp::operations::TileOperations<T>::GenerateCompressedMatrix(*(hcorepp::operators::CompressedTile<T> *) tile,
                                                                         tile_row_index, tile_col_index, aParams);

        metadata->mMatrixRank = ((hcorepp::operators::Tile<T> *) tile)->GetTileRank();
    }

    HICMAPP_INSTANTIATE_CLASS(GenerateCompressedDataCodelet)
}

