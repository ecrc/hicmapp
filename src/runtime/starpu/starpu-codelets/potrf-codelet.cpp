#include <hcorepp/operators/interface/Tile.hpp>
#include <hcorepp/operators/interface/TilePacker.hpp>
#include <hicmapp/runtime/concrete/starpu/starpu-codelets/potrf-codelet.hpp>
#include <hicmapp/common/definitions.h>
#include <hicmapp/runtime/concrete/starpu/factory.hpp>
#include "hicmapp/tile-operations/TileOperations.hpp"

using namespace hicmapp::runtime;

namespace hicmapp::runtime {

    template<typename T> const bool PotrfCodelet<T>::registered_ = PotrfCodelet<T>::Register();

    template<typename T>
    bool PotrfCodelet<T>::Register() {
        static auto maker = new hicmapp::runtime::CodeletMaker<PotrfCodelet, T>(CodeletType::POTRF);
        return true;
    }

    template<typename T>
    starpu_codelet *PotrfCodelet<T>::GetCodelet() {
        return &this->cl_potrf;
    }

    template<typename T>
    PotrfCodelet<T>::PotrfCodelet() {
        cl_potrf = {
#ifdef USE_CUDA
                .where= STARPU_CPU | STARPU_CUDA,
                .cpu_funcs={cl_potrf_func},
                .cuda_funcs={},
                .cuda_flags={0},
#else
                .where=STARPU_CPU,
                .cpu_funcs={cl_potrf_func},
                .cuda_funcs={},
                .cuda_flags={(0)},
#endif
                .nbuffers=(2),
                .model={},
                .name="potrf"};
    }

    template<typename T>
    void PotrfCodelet<T>::cl_potrf_func(void **descr, void *cl_arg) {
        hcorepp::operators::CompressionParameters parameters;
        blas::Uplo uplo;
        hcorepp::operators::TileMetadata *metadata_a;

        starpu_codelet_unpack_args(cl_arg, &uplo);

        metadata_a = (hcorepp::operators::TileMetadata *) STARPU_VARIABLE_GET_PTR(descr[0]);
        auto *tile_a_data = (T *) STARPU_MATRIX_GET_PTR(descr[1]);

        hcorepp::dataunits::MemoryHandler<T> &memory_handler = hcorepp::dataunits::MemoryHandler<T>::GetInstance();
        hcorepp::kernels::RunContext &context = hcorepp::kernels::ContextManager::GetInstance().GetContext();

        auto *tile_a = hcorepp::operators::TilePacker<T>::PackTile(*metadata_a, tile_a_data, context);
//        auto memory_unit = memory_handler.GetMemoryUnit();
        hcorepp::dataunits::MemoryUnit<T> *memory_unit = new hcorepp::dataunits::MemoryUnit<T>(context);

        size_t flops = 0;

        flops += hicmapp::operations::TileOperations<T>::Potrf(*tile_a, uplo, context, *memory_unit);

        memory_unit->FreeAllocations();

        delete memory_unit;
    }

    HICMAPP_INSTANTIATE_CLASS(PotrfCodelet)
}

