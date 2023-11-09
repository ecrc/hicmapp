#ifndef HICMAPP_DEFAULT_RUNTIME_HPP
#define HICMAPP_DEFAULT_RUNTIME_HPP

#include <hcorepp/api/HCore.hpp>
#include <hicmapp/runtime/interface/RunTimeInterface.hpp>

using namespace hicmapp::primitives;

namespace hicmapp {
    namespace runtime {

        template<typename T>
        class DefaultRuntime : public RunTimeInterface<T> {
        public:
            explicit DefaultRuntime([[maybe_unused]] hicmapp::runtime::HicmaHardware &aHardware);

            ~DefaultRuntime();

            int GenerateDenseMatrix(Matrix<T> &aMatrix, size_t aTileIdxInCols,
                                    size_t aTileIdxInRows) override;

            int GenerateCompressedMatrix(Matrix<T> &aMatrix,
                                         size_t aTileIdxInRows, size_t aTileIdxInCols,
                                         const CompressionParameters &aSVDArguments) override;

            size_t
            Gemm(T aAlpha, Matrix<T> &aMatrixA, const size_t &aRowIdxA, const size_t &aColIdxA, const blas::Op &aAOp,
                 Matrix<T> &aMatrixB, const size_t &aRowIdxB, const size_t &aColIdxB, const blas::Op &aBOp, T aBeta,
                 Matrix<T> &aMatrixC, const size_t &aRowIdxC, const size_t &aColIdxC,
                 const hcorepp::kernels::RunContext &aContext, const CompressionParameters &aSVDArguments,
                 hcorepp::dataunits::MemoryUnit<T> &aMemoryUnit, bool aCholesky = false) override;

            int Sync() override;

            void Flush(const Matrix<T> &aMatrix) override;

            void Finalize() override;

            void UnRegisterHandles(Matrix<T> &A) override;

            void RegisterHandles(Matrix<T> &A) override;

            void Flush(const Matrix<T> &aMatrix, const size_t aRowIdx, const size_t aColIdx) override;

            common::RunTimeLibrary LibraryType() override {
                return common::RunTimeLibrary::DEFAULT;
            }

            size_t
            Syrk(Matrix<T> &aMatrixA, const size_t &aRowIdxA,
                 const size_t &aColIdxA, const blas::Op &aAOp, Matrix<T> &aMatrixC,
                 const size_t &aRowIdxC, const size_t &aColIdxC, const blas::Uplo aUplo, T aAlpha,
                 T aBeta, const hcorepp::kernels::RunContext &aContext,
                 hcorepp::dataunits::MemoryUnit<T> &aMemoryUnit) override;

            size_t
            Potrf(Matrix<T> &aMatrixA, const size_t &aRowIdxA, const size_t &aColIdxA, const blas::Uplo aUplo,
                  const hcorepp::kernels::RunContext &aContext,
                  hcorepp::dataunits::MemoryUnit<T> &aMemoryUnit) override;

            size_t
            Trsm(blas::Side aSide, blas::Uplo aUplo, blas::Op aTrans, blas::Diag aDiag, T aAlpha,
                 Matrix<T> &aMatrixA, const size_t &aRowIdxA, const size_t &aColIdxA,
                 Matrix<T> &aMatrixB, const size_t &aRowIdxB, const size_t &aColIdxB,
                 const hcorepp::kernels::RunContext &aContext,
                 hcorepp::dataunits::MemoryUnit<T> &aMemoryUnit) override;

            size_t
            GenerateDiagonalTile(Matrix<T> &aMatrixUV, Matrix<T> &aMatrixDiag, const size_t &aRowIdxDiag,
                                 const size_t &aColIdxDiag, Matrix<T> &aMatrixRK, const size_t &aRowIdx,
                                 const size_t &aColIdx, unsigned long long int seed, size_t maxrank, double tol,
                                 size_t compress_diag, Matrix<T> &aMatrixDense,
                                 const hcorepp::kernels::RunContext &aContext, bool diagonal_tile) override;

            size_t
            LaCpy(Matrix<T> &aMatrixA, const size_t &aRowIdxA, const size_t &aColIdxA,
                  Matrix<T> &aMatrixB, const size_t &aRowIdxB, const size_t &aColIdxB,
                  const hcorepp::kernels::RunContext &aContext) override;

            size_t
            Uncompress(Matrix<T> &aMatrixUV, Matrix<T> &aMatrixDense, Matrix<T> &aMatrixRk, const size_t &aRowIdx,
                       const size_t &aColIdx) override;

        };

    }
}

#endif //HICMAPP_DEFAULT_RUNTIME_HPP