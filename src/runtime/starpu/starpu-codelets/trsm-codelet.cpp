#include <hcorepp/operators/interface/Tile.hpp>
#include <hcorepp/operators/interface/TilePacker.hpp>
#include <hicmapp/runtime/concrete/starpu/starpu-codelets/trsm-codelet.hpp>
#include <hicmapp/common/definitions.h>
#include <hicmapp/runtime/concrete/starpu/factory.hpp>
#include "hicmapp/tile-operations/TileOperations.hpp"

using namespace hicmapp::runtime;

namespace hicmapp::runtime {

    template<typename T> const bool TrsmCodelet<T>::registered_ = TrsmCodelet<T>::Register();

    template<typename T>
    bool TrsmCodelet<T>::Register() {
        static auto maker = new hicmapp::runtime::CodeletMaker<TrsmCodelet, T>(CodeletType::TRSM);
        return true;
    }

    template<typename T>
    starpu_codelet *TrsmCodelet<T>::GetCodelet() {
        return &this->cl_trsm;
    }

    template<typename T>
    TrsmCodelet<T>::TrsmCodelet() {
        cl_trsm = {
#ifdef USE_CUDA
                .where= STARPU_CPU | STARPU_CUDA,
                .cpu_funcs={cl_trsm_func},
                .cuda_funcs={},
                .cuda_flags={0},
#else
                .where=STARPU_CPU,
                .cpu_funcs={cl_trsm_func},
                .cuda_funcs={},
                .cuda_flags={(0)},
#endif
                .nbuffers=(4),
                .model={},
                .name="trsm"};
    }

    template<typename T>
    void TrsmCodelet<T>::cl_trsm_func(void **descr, void *cl_arg) {
        T alpha;
        blas::Uplo uplo;
        blas::Side side;
        blas::Op trans;
        blas::Diag diag;
        hcorepp::operators::TileMetadata *metadata_a, *metadata_b;

        starpu_codelet_unpack_args(cl_arg, &side, &uplo, &trans, &diag, &alpha);

        metadata_a = (hcorepp::operators::TileMetadata *) STARPU_VARIABLE_GET_PTR(descr[0]);
        auto *tile_a_data = (T *) STARPU_MATRIX_GET_PTR(descr[1]);
        metadata_b = (hcorepp::operators::TileMetadata *) STARPU_VARIABLE_GET_PTR(descr[2]);
        auto *tile_b_data = (T *) STARPU_MATRIX_GET_PTR(descr[3]);

        hcorepp::dataunits::MemoryHandler<T> &memory_handler = hcorepp::dataunits::MemoryHandler<T>::GetInstance();
        hcorepp::kernels::RunContext &context = hcorepp::kernels::ContextManager::GetInstance().GetContext();

        auto *tile_a = hcorepp::operators::TilePacker<T>::PackTile(*metadata_a, tile_a_data, context);
        auto *tile_b = hcorepp::operators::TilePacker<T>::PackTile(*metadata_b, tile_b_data, context);
//        auto memory_unit = memory_handler.GetMemoryUnit();
        hcorepp::dataunits::MemoryUnit<T> *memory_unit = new hcorepp::dataunits::MemoryUnit<T>(context);

        int flops = 0;
        flops += hicmapp::operations::TileOperations<T>::Trsm(side, uplo, trans, diag, alpha, *tile_a, *tile_b,
                                                              context, *memory_unit);

//        metadata_b->mMatrixRank = ((Tile<T> *) tile_b)->GetTileRank();


        memory_unit->FreeAllocations();
        delete memory_unit;
    }

    HICMAPP_INSTANTIATE_CLASS(TrsmCodelet)
}

