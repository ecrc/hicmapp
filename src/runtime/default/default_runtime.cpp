#include <hicmapp/runtime/concrete/default/default_runtime.hpp>
#include <hicmapp/tile-operations/TileOperations.hpp>

namespace hicmapp::runtime {

    template<typename T>
    DefaultRuntime<T>::DefaultRuntime([[maybe_unused]] hicmapp::runtime::HicmaHardware &aHardware) {
    }

    template<typename T>
    DefaultRuntime<T>::~DefaultRuntime() = default;

    template<typename T>
    int DefaultRuntime<T>::GenerateDenseMatrix(Matrix<T> &aMatrix, size_t aTileIdxInCols, size_t aTileIdxInRows) {


        auto *tile = static_cast<DenseTile<T> *>(aMatrix.GetTilePointer(aTileIdxInRows, aTileIdxInCols));

        int rc = hicmapp::operations::TileOperations<T>::GenerateDenseTile(*tile, aTileIdxInRows, aTileIdxInCols);

        return rc;
    }

    template<typename T>
    int DefaultRuntime<T>::GenerateCompressedMatrix(Matrix<T> &aMatrix,
                                                    size_t aTileIdxInRows, size_t aTileIdxInCols,
                                                    const CompressionParameters &aSVDArguments) {

        auto *tile = aMatrix.GetTilePointer(aTileIdxInRows, aTileIdxInCols);
        int rc = hicmapp::operations::TileOperations<T>::GenerateCompressedMatrix(*(CompressedTile<T> *) tile,
                                                                                  aTileIdxInRows, aTileIdxInCols,
                                                                                  aSVDArguments);
        return rc;
    }

    template<typename T>
    size_t
    DefaultRuntime<T>::Gemm(T aAlpha, Matrix<T> &aMatrixA, const size_t &aRowIdxA, const size_t &aColIdxA,
                            const blas::Op &aAOp,
                            Matrix<T> &aMatrixB, const size_t &aRowIdxB, const size_t &aColIdxB, const blas::Op &aBOp,
                            T aBeta,
                            Matrix<T> &aMatrixC, const size_t &aRowIdxC, const size_t &aColIdxC,
                            const hcorepp::kernels::RunContext &aContext, const CompressionParameters &aSVDArguments,
                            hcorepp::dataunits::MemoryUnit<T> &aMemoryUnit, bool aCholesky) {

        size_t flops = 0;
        auto *tile_a = aMatrixA.GetTilePointer(aRowIdxA, aColIdxA);
        auto *tile_b = aMatrixB.GetTilePointer(aRowIdxB, aColIdxB);
        auto *tile_c = aMatrixC.GetTilePointer(aRowIdxC, aColIdxC);
        flops += hicmapp::operations::TileOperations<T>::Gemm(aAlpha, *tile_a, aAOp, *tile_b, aBOp,
                                                              aBeta, *tile_c, aContext, aMemoryUnit, aSVDArguments,
                                                              aCholesky);
        return flops;
    }

    template<typename T>
    int DefaultRuntime<T>::Sync() {
        return 0;
    }

    template<typename T>
    void DefaultRuntime<T>::Flush(const Matrix<T> &aMatrix) {

    }

    template<typename T>
    void DefaultRuntime<T>::Finalize() {

    }

    template<typename T>
    void DefaultRuntime<T>::UnRegisterHandles(Matrix<T> &A) {

    }

    template<typename T>
    void DefaultRuntime<T>::RegisterHandles(Matrix<T> &A) {

    }

    template<typename T>
    void DefaultRuntime<T>::Flush(const Matrix<T> &aMatrix, const size_t aRowIdx, const size_t aColIdx) {

    }

    template<typename T>
    size_t DefaultRuntime<T>::Syrk(Matrix<T> &aMatrixA, const size_t &aRowIdxA,
                                   const size_t &aColIdxA, const blas::Op &aAOp, Matrix<T> &aMatrixC,
                                   const size_t &aRowIdxC, const size_t &aColIdxC, const blas::Uplo aUplo, T aAlpha,
                                   T aBeta, const hcorepp::kernels::RunContext &aContext,
                                   hcorepp::dataunits::MemoryUnit<T> &aMemoryUnit) {
        size_t flops = 0;
        auto *tile_a = aMatrixA.GetTilePointer(aRowIdxA, aColIdxA);
        auto *tile_c = aMatrixC.GetTilePointer(aRowIdxC, aColIdxC);

        flops += hicmapp::operations::TileOperations<T>::Syrk(aAlpha, *tile_a, aAOp, aUplo, aBeta, *tile_c, aContext,
                                                              aMemoryUnit);

        return flops;
    }

    template<typename T>
    size_t DefaultRuntime<T>::Potrf(Matrix<T> &aMatrixA, const size_t &aRowIdxA, const size_t &aColIdxA,
                                    const blas::Uplo aUplo,
                                    const hcorepp::kernels::RunContext &aContext,
                                    hcorepp::dataunits::MemoryUnit<T> &aMemoryUnit) {
        size_t flops = 0;
        auto *tile_a = aMatrixA.GetTilePointer(aRowIdxA, aColIdxA);

        flops += hicmapp::operations::TileOperations<T>::Potrf(*tile_a, aUplo, aContext, aMemoryUnit);

        return flops;
    }

    template<typename T>
    size_t DefaultRuntime<T>::Trsm(blas::Side aSide, blas::Uplo aUplo, blas::Op aTrans, blas::Diag aDiag, T aAlpha,
                                   Matrix<T> &aMatrixADiagonal, const size_t &aRowIdxA, const size_t &aColIdxA,
                                   Matrix<T> &aMatrixAUV, const size_t &aRowIdxB, const size_t &aColIdxB,
                                   const hcorepp::kernels::RunContext &aContext,
                                   hcorepp::dataunits::MemoryUnit<T> &aMemoryUnit) {
        size_t flops = 0;
        auto *tile_a = aMatrixADiagonal.GetTilePointer(aRowIdxA, aColIdxA);
        auto *tile_b = aMatrixAUV.GetTilePointer(aRowIdxB, aColIdxB);

        flops += hicmapp::operations::TileOperations<T>::Trsm(aSide, aUplo, aTrans, aDiag, aAlpha, *tile_a, *tile_b,
                                                              aContext, aMemoryUnit);

        return flops;
    }

    template<typename T>
    size_t
    DefaultRuntime<T>::GenerateDiagonalTile(Matrix<T> &aMatrixUV, Matrix<T> &aMatrixDiag, const size_t &aRowIdxDiag,
                                            const size_t &aColIdxDiag, Matrix<T> &aMatrixRK, const size_t &aRowIdx,
                                            const size_t &aColIdx, unsigned long long int seed, size_t maxrank,
                                            double tol,
                                            size_t compress_diag, Matrix<T> &aMatrixDense,
                                            const hcorepp::kernels::RunContext &aContext, bool diagonal_tile) {
        size_t flops = 0;

        auto *tile_auv = aMatrixUV.GetTilePointer(aRowIdx, aColIdx);
        auto *tile_ark = aMatrixRK.GetTilePointer(aRowIdx, aColIdx);
        auto *tile_dense = aMatrixDense.GetTilePointer(aRowIdx, aColIdx);

        Tile<T> *tile_diag = nullptr;

        size_t rows, cols, lda_diag, ld_uv;

        if (diagonal_tile) {
            tile_diag = aMatrixDiag.GetTilePointer(aRowIdxDiag, aColIdxDiag);

            rows = aMatrixDiag.GetNumOfRowsInTile();
            if (aRowIdxDiag == aMatrixDiag.GetNumOfGlobalTilesInRows() - 1) {
                rows = aMatrixDiag.GetGlobalNumOfRowsInMatrix() - aRowIdxDiag * aMatrixDiag.GetNumOfRowsInTile();
            }
            cols = rows;

            lda_diag = rows;// aMatrixDiag.GetTilePointer(aRowIdxDiag, 0).GetLeadingDim();
            ld_uv = 0;

        } else {
            rows = aMatrixDiag.GetNumOfRowsInTile();
            if (aRowIdxDiag == aMatrixDiag.GetNumOfGlobalTilesInRows() - 1) {
                rows = aMatrixDiag.GetGlobalNumOfRowsInMatrix() - aRowIdxDiag * aMatrixDiag.GetNumOfRowsInTile();
            }

            cols = aMatrixUV.GetNumOfRowsInTile();
            if (aRowIdx == aMatrixUV.GetNumOfGlobalTilesInRows() - 1) {
                cols = aMatrixUV.GetGlobalNumOfRowsInMatrix() - aRowIdx * aMatrixUV.GetNumOfRowsInTile();
            }

            lda_diag = rows;//aMatrixDiag.GetTilePointer(aRowIdxDiag, 0).GetLeadingDim();
            ld_uv = cols;//aMatrixUV.GetTilePointer(aRowIdx, 0).GetLeadingDim();

        }

        flops += hicmapp::operations::TileOperations<T>::GenerateDiagonalTile(tile_auv, tile_ark, tile_dense,
                                                                              tile_diag, aRowIdx, aColIdx,
                                                                              seed, maxrank, tol, compress_diag,
                                                                              lda_diag, ld_uv, ld_uv, rows, cols,
                                                                              aContext);

        return flops;
    }

    template<typename T>
    size_t DefaultRuntime<T>::LaCpy(Matrix<T> &aMatrixA, const size_t &aRowIdxA, const size_t &aColIdxA,
                                    Matrix<T> &aMatrixB, const size_t &aRowIdxB, const size_t &aColIdxB,
                                    const hcorepp::kernels::RunContext &aContext) {

        size_t flops = 0;
        auto* tile_a = aMatrixA.GetTilePointer(aRowIdxA, aColIdxA);
        int rows = tile_a->GetNumOfRows();
        int cols = rows;

        auto tile_b = aMatrixB.GetTilePointer(aRowIdxB, aColIdxB);

        flops += hicmapp::operations::TileOperations<T>::LaCpy(rows, cols, *tile_a, *tile_b, aContext);

        return flops;
    }

    template<typename T>
    size_t DefaultRuntime<T>::Uncompress(Matrix<T> &aMatrixUV, Matrix<T> &aMatrixDense, Matrix<T> &aMatrixRk,
                                         const size_t &aRowIdx, const size_t &aColIdx) {

        Tile<T> *tile_d = aMatrixDense.GetTilePointer(aRowIdx, aColIdx);
        size_t nrows = tile_d->GetNumOfRows();
        size_t ldad = tile_d->GetLeadingDim();
        auto dense_data = tile_d->GetTileSubMatrix(0);

        auto *tile_uv = static_cast<CompressedTile<T> *>(aMatrixUV.GetTilePointer(aRowIdx, aColIdx));
        size_t ldauv = tile_uv->GetULeadingDim();
        auto u_data = tile_uv->GetUMatrix();
        auto v_data = tile_uv->GetVMatrix();
        size_t ncols = aMatrixUV.GetNumOfRowsInTile();

        T rk_data = tile_uv->GetTileRank();

        hicmapp::operations::TileOperations<T>::UnCompressTile(nrows, ncols, 1,
                                                               u_data, &rk_data, ldauv,
                                                               v_data, ldauv, 0, dense_data, ldad);

    }

    HICMAPP_INSTANTIATE_CLASS(DefaultRuntime)

}