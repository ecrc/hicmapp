#include <hcorepp/operators/interface/Tile.hpp>
#include <hcorepp/operators/interface/TilePacker.hpp>
#include <hicmapp/runtime/concrete/starpu/starpu-codelets/uncompress-codelet.hpp>
#include <hicmapp/common/definitions.h>
#include <hicmapp/runtime/concrete/starpu/factory.hpp>
#include "hicmapp/tile-operations/TileOperations.hpp"

using namespace hicmapp::runtime;

namespace hicmapp::runtime {

    template<typename T> const bool UncompressCodelet<T>::registered_ = UncompressCodelet<T>::Register();

    template<typename T>
    bool UncompressCodelet<T>::Register() {
        static auto maker = new hicmapp::runtime::CodeletMaker<UncompressCodelet, T>(CodeletType::UNCOMPRESS);
        return true;
    }

    template<typename T>
    starpu_codelet *UncompressCodelet<T>::GetCodelet() {
        return &this->cl_uncompress;
    }

    template<typename T>
    UncompressCodelet<T>::UncompressCodelet() {
        cl_uncompress = {
#ifdef USE_CUDA
                .where= STARPU_CPU | STARPU_CUDA,
                .cpu_funcs={cl_uncompress_func},
                .cuda_funcs={},
                .cuda_flags={0},
#else
                .where=STARPU_CPU,
                .cpu_funcs={cl_uncompress_func},
                .cuda_funcs={},
                .cuda_flags={(0)},
#endif
                .nbuffers=(6),
                .model={},
                .name="uncompress"};
    }

    template<typename T>
    void UncompressCodelet<T>::cl_uncompress_func(void **descr, void *cl_arg) {
        T alpha = 1;
        T beta = 0;

        hcorepp::common::BlasOperation a_trans, b_trans;
        size_t ncols;

        hcorepp::operators::TileMetadata *metadata_auv, *metadata_dense, *metadata_rk;

        starpu_codelet_unpack_args(cl_arg, &a_trans, &b_trans, &ncols);

        metadata_auv = (hcorepp::operators::TileMetadata *) STARPU_VARIABLE_GET_PTR(descr[0]);
        auto *tile_auv_data = (T *) STARPU_MATRIX_GET_PTR(descr[1]);
        metadata_dense = (hcorepp::operators::TileMetadata *) STARPU_VARIABLE_GET_PTR(descr[2]);
        auto *tile_dense_data = (T *) STARPU_MATRIX_GET_PTR(descr[3]);
        metadata_rk = (hcorepp::operators::TileMetadata *) STARPU_VARIABLE_GET_PTR(descr[4]);
        auto *tile_rk_data = (T *) STARPU_MATRIX_GET_PTR(descr[5]);

        hcorepp::dataunits::MemoryHandler<T> &memory_handler = hcorepp::dataunits::MemoryHandler<T>::GetInstance();
        hcorepp::kernels::RunContext &context = hcorepp::kernels::ContextManager::GetInstance().GetContext();

        auto *tile_uv = hcorepp::operators::TilePacker<T>::PackTile(*metadata_auv, tile_auv_data, context);
        auto *tile_dense = hcorepp::operators::TilePacker<T>::PackTile(*metadata_dense, tile_dense_data, context);
        auto *tile_rk = hcorepp::operators::TilePacker<T>::PackTile(*metadata_rk, tile_rk_data, context);

        auto memory_unit = memory_handler.GetMemoryUnit();

        size_t nrows = tile_dense->GetNumOfRows();
        size_t ldad = tile_dense->GetLeadingDim();
        auto dense_data = tile_dense->GetTileSubMatrix(0);

        auto *tile_comp = static_cast<hcorepp::operators::CompressedTile<T> *>(tile_uv);
        size_t ldauv = tile_comp->GetULeadingDim();
        auto u_data = tile_comp->GetUMatrix();
        auto v_data = tile_comp->GetVMatrix();

        T rk_data = tile_comp->GetTileRank();

        hicmapp::operations::TileOperations<T>::UnCompressTile(nrows, ncols, alpha, u_data, &rk_data,
                                                               ldauv, v_data,
                                                               ldauv, beta, dense_data, ldad);

        memory_unit.FreeAllocations();
    }

    HICMAPP_INSTANTIATE_CLASS(UncompressCodelet)
}

