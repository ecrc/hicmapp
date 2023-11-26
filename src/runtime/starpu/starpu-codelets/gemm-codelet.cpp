#include <hcorepp/operators/interface/Tile.hpp>
#include <hcorepp/operators/interface/TilePacker.hpp>
#include <hicmapp/runtime/concrete/starpu/starpu-codelets/gemm-codelet.hpp>
#include <hicmapp/common/definitions.h>
#include <hicmapp/runtime/concrete/starpu/factory.hpp>
#include "hicmapp/tile-operations/TileOperations.hpp"

using namespace hicmapp::runtime;

namespace hicmapp::runtime {

    template<typename T> const bool GemmCodelet<T>::registered_ = GemmCodelet<T>::Register();

    template<typename T>
    bool GemmCodelet<T>::Register() {
        static auto maker = new hicmapp::runtime::CodeletMaker<GemmCodelet, T>(CodeletType::GEMM);
        return true;
    }

    template<typename T>
    starpu_codelet *GemmCodelet<T>::GetCodelet() {
        return &this->cl_dgemm;
    }

    template<typename T>
    GemmCodelet<T>::GemmCodelet() {
        cl_dgemm = {
#ifdef USE_CUDA
                .where= STARPU_CPU | STARPU_CUDA,
                .cpu_funcs={cl_dgemm_func},
                .cuda_funcs={},
                .cuda_flags={0},
#else
                .where=STARPU_CPU,
                .cpu_funcs={cl_dgemm_func},
                .cuda_funcs={},
                .cuda_flags={(0)},
#endif
                .nbuffers=(6),
                .model={},
                .name="dgemm"};
    }

    template<typename T>
    void GemmCodelet<T>::cl_dgemm_func(void **descr, void *cl_arg) {
        T alpha, beta;
        hcorepp::operators::CompressionParameters parameters;
        blas::Op AOp, BOp;
        hcorepp::operators::TileMetadata *metadata_a, *metadata_b, *metadata_c;
        bool cholesky;

        starpu_codelet_unpack_args(cl_arg, &alpha, &AOp, &BOp, &beta, &parameters, &cholesky);

        metadata_a = (hcorepp::operators::TileMetadata *) STARPU_VARIABLE_GET_PTR(descr[0]);
        auto *tile_a = (T *) STARPU_MATRIX_GET_PTR(descr[1]);
        metadata_b = (hcorepp::operators::TileMetadata *) STARPU_VARIABLE_GET_PTR(descr[2]);
        auto *tile_b = (T *) STARPU_MATRIX_GET_PTR(descr[3]);
        metadata_c = (hcorepp::operators::TileMetadata *) STARPU_VARIABLE_GET_PTR(descr[4]);
        auto *tile_c = (T *) STARPU_MATRIX_GET_PTR(descr[5]);

        hcorepp::dataunits::MemoryHandler<T> &memory_handler = hcorepp::dataunits::MemoryHandler<T>::GetInstance();
        hcorepp::kernels::RunContext &context = hcorepp::kernels::ContextManager::GetInstance().GetContext();

//        static int index = 0;
        auto *a = hcorepp::operators::TilePacker<T>::PackTile(*metadata_a, tile_a, context);
        auto *b = hcorepp::operators::TilePacker<T>::PackTile(*metadata_b, tile_b, context);
        auto *c = hcorepp::operators::TilePacker<T>::PackTile(*metadata_c, tile_c, context);
//        auto &memory_unit = memory_handler.GetMemoryUnit(index);
        hcorepp::dataunits::MemoryUnit<T> *memory_unit = new hcorepp::dataunits::MemoryUnit<T>(context);
        hicmapp::operations::TileOperations<T>::Gemm(alpha, *a, AOp, *b, BOp, beta, *c, context,
                                                     *memory_unit, parameters, cholesky);
        metadata_c->mMatrixRank = ((hcorepp::operators::Tile<T> *) c)->GetTileRank();
        memory_unit->FreeAllocations();
//        index++;
        delete memory_unit;
//        delete a;
//        delete b;
//        delete c;
    }

    HICMAPP_INSTANTIATE_CLASS(GemmCodelet)
}

