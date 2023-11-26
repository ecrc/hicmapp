#ifndef HICMAPP_OPERATIONS_MATRIX_OPERATIONS_HPP
#define HICMAPP_OPERATIONS_MATRIX_OPERATIONS_HPP

#include <hicmapp/primitives/matrix.hpp>

using namespace hicmapp::primitives;

namespace hicmapp {
    namespace operations {

        template<typename T>
        class MatrixOperations {

        public:

            static int
            GenerateCompressedMatrix(common::Uplo auplo, Matrix<T> &apAUV,
                                     const CompressionParameters &aSVDArguments, bool aAsync);


            static int
            GenerateDenseMatrix(common::Uplo auplo, Matrix <T> &aMatrix, bool aASync);

            static int
            UncompressMatrix(common::Uplo auplo, Matrix <T> &aMatrixUV, Matrix <T> &aMatrixRK, Matrix <T> &aMatrixD);

            static size_t
            Gemm(Matrix <T> &aMatrixA, const blas::Op &aAOp, Matrix <T> &aMatrixB,
                 const blas::Op &aBOp, Matrix <T> &aMatrixC, T &aAlpha, T &aBeta,
                 runtime::HicmaContext &aContext, const CompressionParameters &aSVDArguments,
                 const std::vector<std::vector<size_t>> &aRanks, bool aAllocatePool = false);

            static size_t
            Cholesky(common::Uplo aUpperLower, Matrix <T> &aMatrixAUV, Matrix <T> &aMatrixADiagonal,
                     Matrix <T> &aMatrixARK, int aRank, int aMaxRank, double aAccuracy,
                     runtime::HicmaContext &aContext);

            static size_t
            DiagVecToMat(Matrix <T> &aMatrixDiag, Matrix <T> &aMatrixDense, runtime::HicmaContext &aContext);

            static size_t
            GenerateDiagonalTiles(common::Uplo auplo, Matrix <T> &aMatrixUV, Matrix <T> &aMatrixRK,
                                  Matrix <T> &aMatrixD, unsigned long long int seed,
                                  int maxrank, double tol, int compress_diag, Matrix <T> &aMatrixDense,
                                  runtime::HicmaContext &aContext);

            static std::vector<size_t>
            CalculateGemmPoolSize(Matrix<T> &aMatrixA, const blas::Op &aAOp, Matrix<T> &aMatrixB,
                                  const blas::Op &aBOp, Matrix<T> &aMatrixC, T &aAlpha, T &aBeta,
                                  runtime::HicmaContext& aContext,
                                  const CompressionParameters &aSVDArguments,
                                  const std::vector<std::vector<size_t>> &aRanks);

        private:
            /**
             * @brief
             * Prevent Class Instantiation for Operations Wrapper Class.
             */
            MatrixOperations() = default;

        };

    }
}
#endif //HICMAPP_OPERATIONS_MATRIX_OPERATIONS_HPP
