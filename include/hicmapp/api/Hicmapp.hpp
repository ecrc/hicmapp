#ifndef HICMAPP_API_HICMAPP_HPP
#define HICMAPP_API_HICMAPP_HPP

#include <hicmapp/primitives/matrix.hpp>
#include <hicmapp/matrix-operations/interface/MatrixOperations.hpp>
#include <hicmapp/runtime/interface/HicmaHardware.hpp>

using namespace hicmapp::primitives;

namespace hicmapp::api {
        template<typename T>
        class Hicmapp {
        public:
            static void
            GenerateDenseMatrix(common::Uplo auplo, Matrix<T> &apMatrix, bool aASync);

            static void
            Init(int aCPUs = 1, int aGPUs = 0, int aThreadsPerWorker = -1);

            static void
            Finalize();

            static void
            GenerateCompressedMatrix(common::Uplo auplo, Matrix<T> &apMatrix,
                                     const CompressionParameters &aSVDArguments, bool aASync);

            static size_t
            Gemm(Matrix<T> &apMatrixA, const blas::Op &aAOp, Matrix<T> &apMatrixB,
                 const blas::Op &aBOp, Matrix<T> &apMatrixC, T &aAlpha, T &aBeta,
                 runtime::HicmaContext &aContext, bool aAllocatePool = false,
                 const CompressionParameters &aSVDArguments = {1e-9},
                 const std::vector<std::vector<size_t>> &aRanks = {});

            static void
            UncompressMatrix(common::Uplo auplo, Matrix<T> &apMatrixUV, Matrix<T> &apMatrixRK, Matrix<T> &apMatrixD);

            static size_t
            GenerateDiagonalTiles(common::Uplo auplo, Matrix<T> &aMatrixUV, Matrix<T> &aMatrixRK,
                                  Matrix<T> &apMatrixD, unsigned long long int seed,
                                  int maxrank, double tol, int compress_diag, Matrix<T> &apMatrixDense,
                                  runtime::HicmaContext &aContext);

            static size_t
            Cholesky(common::Uplo aUpperLower, Matrix<T> &aMatrixAUV, Matrix<T> &aMatrixAD,
                     Matrix<T> &aMatrixARK, int aRank, int aMaxRank, double aAccuracy,
                     runtime::HicmaContext &aContext);

            static size_t
            DiagVecToMat(Matrix<T> &aMatrixDiag, Matrix<T> &aMatrixDense, runtime::HicmaContext &aContext);

        private:
            Hicmapp() = default;
        };
    }
#endif //HICMAPP_API_HICMAPP_HPP