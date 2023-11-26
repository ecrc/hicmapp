//
// Created by mirna on 01/11/23.
//
#include <hcorepp/operators/interface/Tile.hpp>
#include <hcorepp/operators/interface/TilePacker.hpp>
#include <hicmapp/runtime/concrete/starpu/starpu-codelets/generate-dgytlr-diag-codelet.hpp>
#include <hicmapp/common/definitions.h>
#include <hicmapp/runtime/concrete/starpu/factory.hpp>
#include "hicmapp/tile-operations/TileOperations.hpp"

using namespace hicmapp::runtime;

namespace hicmapp::runtime {

    template<typename T> const bool GenerateDgytlrDiagonalCodelet<T>::registered_ = GenerateDgytlrDiagonalCodelet<T>::Register();

    template<typename T>
    bool GenerateDgytlrDiagonalCodelet<T>::Register() {
        static auto maker = new hicmapp::runtime::CodeletMaker<GenerateDgytlrDiagonalCodelet, T>(
                CodeletType::DGYTLR_DIAG);
        return true;
    }

    template<typename T>
    starpu_codelet *GenerateDgytlrDiagonalCodelet<T>::GetCodelet() {
        return &this->cl_dgytlr_diag;
    }

    template<typename T>
    GenerateDgytlrDiagonalCodelet<T>::GenerateDgytlrDiagonalCodelet() {
        cl_dgytlr_diag = {
#ifdef USE_CUDA
                .where= STARPU_CPU | STARPU_CUDA,
                .cpu_funcs={cl_dgytlr_diag_func},
                .cuda_funcs={},
                .cuda_flags={0},
#else
                .where=STARPU_CPU,
                .cpu_funcs={cl_dgytlr_diag_func},
                .cuda_funcs={},
                .cuda_flags={(0)},
#endif
                .nbuffers=(8),
                .model={},
                .name="dgytlr_diag"};
    }

    template<typename T>
    void GenerateDgytlrDiagonalCodelet<T>::cl_dgytlr_diag_func(void **descr, void *cl_arg) {
        hcorepp::operators::TileMetadata *metadata_auv, *metadata_ark, *metadata_dense, *metadata_diag;
        size_t row_idx, col_idx;
        size_t lda, ldu, ldv, rows, cols;
        unsigned long long int seed;
        size_t maxrank;
        double tol;
        size_t compress_diag;

        starpu_codelet_unpack_args(cl_arg, &row_idx, &col_idx, &seed, &maxrank, &tol, &compress_diag, &lda, &ldu, &ldv,
                                   &rows, &cols);

        metadata_auv = (hcorepp::operators::TileMetadata *) STARPU_VARIABLE_GET_PTR(descr[0]);
        auto *tile_auv_data = (T *) STARPU_MATRIX_GET_PTR(descr[1]);
        metadata_ark = (hcorepp::operators::TileMetadata *) STARPU_VARIABLE_GET_PTR(descr[2]);
        auto *tile_ark_data = (T *) STARPU_MATRIX_GET_PTR(descr[3]);
        metadata_dense = (hcorepp::operators::TileMetadata *) STARPU_VARIABLE_GET_PTR(descr[4]);
        auto *tile_dense_data = (T *) STARPU_MATRIX_GET_PTR(descr[5]);
        metadata_diag = (hcorepp::operators::TileMetadata *) STARPU_VARIABLE_GET_PTR(descr[6]);
        auto *tile_diag_data = (T *) STARPU_MATRIX_GET_PTR(descr[7]);

        hcorepp::dataunits::MemoryHandler<T> &memory_handler = hcorepp::dataunits::MemoryHandler<T>::GetInstance();
        hcorepp::kernels::RunContext &context = hcorepp::kernels::ContextManager::GetInstance().GetContext();

        auto *tile_auv = hcorepp::operators::TilePacker<T>::PackTile(*metadata_auv, tile_auv_data, context);
        auto *tile_ark = hcorepp::operators::TilePacker<T>::PackTile(*metadata_ark, tile_ark_data, context);
        auto *tile_dense = hcorepp::operators::TilePacker<T>::PackTile(*metadata_dense, tile_dense_data, context);
        auto *tile_diag = hcorepp::operators::TilePacker<T>::PackTile(*metadata_diag, tile_diag_data, context);

        auto memory_unit = memory_handler.GetMemoryUnit();

        size_t flops = 0;
        flops += hicmapp::operations::TileOperations<T>::GenerateDiagonalTile(tile_auv, tile_ark, tile_dense,
                                                                              tile_diag, row_idx, col_idx,
                                                                              seed, maxrank, tol, compress_diag,
                                                                              lda, ldu, ldv, rows, cols, context);
        metadata_auv->mMatrixRank = ((hcorepp::operators::Tile<T> *) tile_auv)->GetTileRank();

        memory_unit.FreeAllocations();
    }

    HICMAPP_INSTANTIATE_CLASS(GenerateDgytlrDiagonalCodelet)
}

