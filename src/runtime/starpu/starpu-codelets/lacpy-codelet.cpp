#include <hcorepp/operators/interface/Tile.hpp>
#include <hcorepp/operators/interface/TilePacker.hpp>
#include <hicmapp/runtime/concrete/starpu/starpu-codelets/lacpy-codelet.hpp>
#include <hicmapp/common/definitions.h>
#include <hicmapp/runtime/concrete/starpu/factory.hpp>
#include "hicmapp/tile-operations/TileOperations.hpp"

using namespace hicmapp::runtime;

namespace hicmapp::runtime {

    template<typename T> const bool LacpyCodelet<T>::registered_ = LacpyCodelet<T>::Register();

    template<typename T>
    bool LacpyCodelet<T>::Register() {
        static auto maker = new hicmapp::runtime::CodeletMaker<LacpyCodelet, T>(CodeletType::LACPY);
        return true;
    }

    template<typename T>
    starpu_codelet *LacpyCodelet<T>::GetCodelet() {
        return &this->cl_lacpy;
    }

    template<typename T>
    LacpyCodelet<T>::LacpyCodelet() {
        cl_lacpy = {
#ifdef USE_CUDA
                .where= STARPU_CPU | STARPU_CUDA,
                .cpu_funcs={cl_lacpy_func},
                .cuda_funcs={},
                .cuda_flags={0},
#else
                .where=STARPU_CPU,
                .cpu_funcs={cl_lacpy_func},
                .cuda_funcs={},
                .cuda_flags={(0)},
#endif
                .nbuffers=(4),
                .model={},
                .name="lacpy"};
    }

    template<typename T>
    void LacpyCodelet<T>::cl_lacpy_func(void **descr, void *cl_arg) {
        hcorepp::operators::TileMetadata *metadata_a, *metadata_b;

        metadata_a = (hcorepp::operators::TileMetadata *) STARPU_VARIABLE_GET_PTR(descr[0]);
        auto *tile_a_data = (T *) STARPU_MATRIX_GET_PTR(descr[1]);
        metadata_b = (hcorepp::operators::TileMetadata *) STARPU_VARIABLE_GET_PTR(descr[2]);
        auto *tile_b_data = (T *) STARPU_MATRIX_GET_PTR(descr[3]);

        hcorepp::dataunits::MemoryHandler<T> &memory_handler = hcorepp::dataunits::MemoryHandler<T>::GetInstance();
        hcorepp::kernels::RunContext &context = hcorepp::kernels::ContextManager::GetInstance().GetContext();

        auto *tile_a = hcorepp::operators::TilePacker<T>::PackTile(*metadata_a, tile_a_data,
                                                                   context);
        auto *tile_b = hcorepp::operators::TilePacker<T>::PackTile(*metadata_b, tile_b_data,
                                                                   context);

        auto memory_unit = memory_handler.GetMemoryUnit();

        int rows = tile_a->GetNumOfRows();
        int cols = rows;

        hicmapp::operations::TileOperations<T>::LaCpy(rows, cols, *tile_a, *tile_b,
                                                      context);

        memory_unit.FreeAllocations();
    }

    HICMAPP_INSTANTIATE_CLASS(LacpyCodelet)
}

