#ifndef HICMAPP_RUNTIME_CONCRETE_STARPU_HPP
#define HICMAPP_RUNTIME_CONCRETE_STARPU_HPP

#include <hcorepp/api/HCore.hpp>
#include <hicmapp/runtime/interface/RunTimeInterface.hpp>
#include "hicma_starpu.hpp"
#include <hicmapp/runtime/interface/HicmaHardware.hpp>

using namespace hicmapp::primitives;

namespace hicmapp::runtime {

        typedef std::vector<starpu_data_handle_t> TileHandles;
        typedef std::unordered_map<size_t, TileHandles> TileHandlesMap;

        template<typename T>
        class StarPu : public RunTimeInterface<T> {

        public:
            explicit StarPu(hicmapp::runtime::HicmaHardware &aHardware);

            ~StarPu();

            int GenerateDenseMatrix(Matrix<T>& apMatrix, size_t aTileIdxInRows, size_t aTileIdxInCols) override;

            int GenerateCompressedMatrix(Matrix<T> &apMatrix, size_t aTileIdxInRows, size_t aTileIdxInCols,
                                         const CompressionParameters& aSVDArguments) override;

            size_t
            Gemm(T aAlpha, Matrix<T> &apMatrixA, const size_t &aRowIdxA, const size_t &aColIdxA, const blas::Op &aAOp,
                 Matrix<T> &apMatrixB, const size_t &aRowIdxB, const size_t &aColIdxB, const blas::Op &aBOp, T aBeta,
                 Matrix<T> &apMatrixC, const size_t &aRowIdxC, const size_t &aColIdxC,
                 const hcorepp::kernels::RunContext &aContext, const CompressionParameters &aSVDArguments,
                 hcorepp::dataunits::MemoryUnit<T> &aMemoryUnit, bool aCholesky = false) override;

            int Sync() override;

            void Flush(const Matrix<T> &apMatrix, const size_t aRowIdx, const size_t aColIdx) override;

            void Flush(const Matrix<T> &apMatrix) override;

            TileHandlesMap &GetMatrixHandles(size_t aMatrixId);

            void Finalize() override;

            void UnRegisterHandles(Matrix<T> &A) override;

            void UnRegisterTileHandles(TileHandles &aHandles);

            void RegisterHandles(Matrix<T> &A) override;

            common::RunTimeLibrary LibraryType() override {
                return common::RunTimeLibrary::STARPU;
            }

            size_t
            Syrk(Matrix<T> &apMatrixA, const size_t &aRowIdxA,
                 const size_t &aColIdxA, const blas::Op &aAOp, Matrix<T> &apMatrixC,
                 const size_t &aRowIdxC, const size_t &aColIdxC, blas::Uplo aUplo, T aAlpha,
                 T aBeta, const hcorepp::kernels::RunContext &aContext,
                 hcorepp::dataunits::MemoryUnit<T> &aMemoryUnit) override;

            size_t
            Potrf(Matrix<T> &apMatrixA, const size_t &aRowIdxA, const size_t &aColIdxA, blas::Uplo aUplo,
                  const hcorepp::kernels::RunContext &aContext,
                  hcorepp::dataunits::MemoryUnit<T> &aMemoryUnit) override;

            size_t
            Trsm(blas::Side aSide, blas::Uplo aUplo, blas::Op aTrans, blas::Diag aDiag, T aAlpha,
                 Matrix<T> &apMatrixA, const size_t &aRowIdxA, const size_t &aColIdxA,
                 Matrix<T> &apMatrixB, const size_t &aRowIdxB, const size_t &aColIdxB,
                 const hcorepp::kernels::RunContext &aContext,
                 hcorepp::dataunits::MemoryUnit<T> &aMemoryUnit) override;

            size_t
            GenerateDiagonalTile(Matrix<T> &apMatrixUV, Matrix<T> &apMatrixDiag, const size_t &aRowIdxDiag,
                                 const size_t &aColIdxDiag, Matrix<T> &apMatrixRK, const size_t &aRowIdx,
                                 const size_t &aColIdx, unsigned long long int seed, size_t maxrank, double tol,
                                 size_t compress_diag, Matrix<T> &apMatrixDense,
                                 const hcorepp::kernels::RunContext &aContext, bool diagonal_tile) override;

            size_t
            LaCpy(Matrix<T> &apMatrixA, const size_t &aRowIdxA, const size_t &aColIdxA,
                  Matrix<T> &apMatrixB, const size_t &aRowIdxB, const size_t &aColIdxB,
                  const hcorepp::kernels::RunContext &aContext) override;

            size_t
            Uncompress(Matrix<T> &apMatrixUV, Matrix<T> &apMatrixDense, Matrix<T> &apMatrixRk, const size_t &aRowIdx,
                       const size_t &aColIdx) override;

        private:
            void
            RegisterTileHandles(Matrix<T> &A, size_t aM, size_t aN);

            TileHandles&
            GetTileHandles(Matrix<T> &A, size_t aM, size_t aN);

            TileHandles &
            GetTileHandles(TileHandlesMap &aHandlesMap, size_t aHandleIdx);

        private:
            /*** RunTimeHandles Map, Each matrix has one TileHandlesMap. MatrixID -> TileHandlesMap */
            std::unordered_map<size_t, TileHandlesMap> mRunTimeHandles{};
            /*** TileMetadata Map, Tile Idx -> (row_idx, col_idx, tile_metadata) */
            std::unordered_map<size_t, std::vector<std::tuple<size_t, size_t, TileMetadata *>>> mTileMetadata{};
            /*** StarPu Configurations object */
            starpu_conf_t *mConf;
        };
    }
#endif //HICMAPP_RUNTIME_CONCRETE_STARPU_HPP
