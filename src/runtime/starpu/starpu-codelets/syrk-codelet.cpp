#include <hcorepp/operators/interface/Tile.hpp>
#include <hcorepp/operators/interface/TilePacker.hpp>
#include <hicmapp/runtime/concrete/starpu/starpu-codelets/syrk-codelet.hpp>
#include <hicmapp/common/definitions.h>
#include <hicmapp/runtime/concrete/starpu/factory.hpp>
#include "hicmapp/tile-operations/TileOperations.hpp"

using namespace hicmapp::runtime;

namespace hicmapp::runtime {

    template<typename T> const bool SyrkCodelet<T>::registered_ = SyrkCodelet<T>::Register();

    template<typename T>
    bool SyrkCodelet<T>::Register() {
        static auto maker = new hicmapp::runtime::CodeletMaker<SyrkCodelet, T>(CodeletType::SYRK);
        return true;
    }

    template<typename T>
    starpu_codelet *SyrkCodelet<T>::GetCodelet() {
        return &this->cl_syrk;
    }

    template<typename T>
    SyrkCodelet<T>::SyrkCodelet() {
        cl_syrk = {
#ifdef USE_CUDA
                .where= STARPU_CPU | STARPU_CUDA,
                .cpu_funcs={cl_syrk_func},
                .cuda_funcs={},
                .cuda_flags={0},
#else
                .where=STARPU_CPU,
                .cpu_funcs={cl_syrk_func},
                .cuda_funcs={},
                .cuda_flags={(0)},
#endif
                .nbuffers=(4),
                .model={},
                .name="syrk"};
    }

    template<typename T>
    void SyrkCodelet<T>::cl_syrk_func(void **descr, void *cl_arg) {
        T alpha, beta;
        blas::Op AOp;
        blas::Uplo uplo;
        hcorepp::operators::TileMetadata *metadata_a, *metadata_diag;

        starpu_codelet_unpack_args(cl_arg, &alpha, &AOp, &uplo, &beta);

        metadata_a = (hcorepp::operators::TileMetadata *) STARPU_VARIABLE_GET_PTR(descr[0]);
        auto *tile_a_data = (T *) STARPU_MATRIX_GET_PTR(descr[1]);
        metadata_diag = (hcorepp::operators::TileMetadata *) STARPU_VARIABLE_GET_PTR(descr[2]);
        auto *tile_diag_data = (T *) STARPU_MATRIX_GET_PTR(descr[3]);

        hcorepp::dataunits::MemoryHandler<T> &memory_handler = hcorepp::dataunits::MemoryHandler<T>::GetInstance();
        hcorepp::kernels::RunContext &context = hcorepp::kernels::ContextManager::GetInstance().GetContext();

        auto *tile_a = hcorepp::operators::TilePacker<T>::PackTile(*metadata_a, tile_a_data, context);
        auto *tile_diag = hcorepp::operators::TilePacker<T>::PackTile(*metadata_diag, tile_diag_data, context);
//        auto memory_unit = memory_handler.GetMemoryUnit();

        hcorepp::dataunits::MemoryUnit<T> *memory_unit = new hcorepp::dataunits::MemoryUnit<T>(context);

        int flops = 0;
        flops += hicmapp::operations::TileOperations<T>::Syrk(alpha, *tile_a, AOp, uplo, beta, *tile_diag, context,
                                                              *memory_unit);

        memory_unit->FreeAllocations();
        delete memory_unit;
    }

    HICMAPP_INSTANTIATE_CLASS(SyrkCodelet)
}

