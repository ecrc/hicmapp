#include <hicmapp/api/Hicmapp.hpp>
#include "hicmapp/runtime/interface/RunTimeSingleton.hpp"
#include "hicmapp/problem-manager/StarshManager.hpp"

namespace hicmapp::api {
        template<typename T>
        void Hicmapp<T>::GenerateDenseMatrix(common::Uplo auplo, Matrix<T> &aMatrix, bool aASync) {
            hicmapp::operations::MatrixOperations<T>::GenerateDenseMatrix(auplo, aMatrix, aASync);
        }

        template<typename T>
        void Hicmapp<T>::Init(int aCPUs, int aGPUs, int aThreadsPerWorker) {
            auto hardware = hicmapp::runtime::HicmaHardware(aCPUs, aGPUs, aThreadsPerWorker);
            hicmapp::runtime::RunTimeSingleton<T>::setRunTimeInstance(hardware);
        }

        template<typename T>
        void Hicmapp<T>::Finalize() {
            hicmapp::runtime::RunTimeSingleton<T>::GetRunTimeInstance()->Finalize();
            hicmapp::operations::StarsHManager::DestroyStarsHManager();
        }

        template<typename T>
        void Hicmapp<T>::GenerateCompressedMatrix(common::Uplo auplo, Matrix<T> &aMatrix,
                                                  const CompressionParameters &aSVDArguments, bool aASync) {

            hicmapp::operations::MatrixOperations<T>::GenerateCompressedMatrix(auplo, aMatrix, aSVDArguments, aASync);
        }

        template<typename T>
        void Hicmapp<T>::UncompressMatrix(common::Uplo auplo, Matrix<T> &aMatrixUV, Matrix<T> &aMatrixRK,
                                          Matrix<T> &aMatrixD) {
            hicmapp::operations::MatrixOperations<T>::UncompressMatrix(auplo, aMatrixUV, aMatrixRK, aMatrixD);
        }

        template<typename T>
        size_t Hicmapp<T>::Gemm(Matrix<T> &aMatrixA, const blas::Op &aAOp, Matrix<T> &aMatrixB, const blas::Op &aBOp,
                                Matrix<T> &aMatrixC, T &aAlpha, T &aBeta, runtime::HicmaContext &aContext,
                                bool aAllocatePool, const CompressionParameters &aSVDArguments,
                                const std::vector<std::vector<size_t>> &aRanks) {
            return hicmapp::operations::MatrixOperations<T>::Gemm(aMatrixA, aAOp, aMatrixB, aBOp, aMatrixC, aAlpha,
                                                                  aBeta, aContext, aSVDArguments, aRanks,
                                                                  aAllocatePool);
        }

        template<typename T>
        size_t Hicmapp<T>::GenerateDiagonalTiles(common::Uplo auplo, Matrix<T> &aMatrixUV, Matrix<T> &aMatrixRK,
                                                 Matrix<T> &aMatrixD, unsigned long long int aSeed, int aMaxRank,
                                                 double aTol, int aCompressDiag, Matrix<T> &aMatrixDense,
                                                 runtime::HicmaContext &aContext) {
            return hicmapp::operations::MatrixOperations<T>::GenerateDiagonalTiles(auplo, aMatrixUV, aMatrixRK,
                                                                                   aMatrixD, aSeed, aMaxRank, aTol,
                                                                                   aCompressDiag, aMatrixDense,
                                                                                   aContext);

        }

        template<typename T>
        size_t Hicmapp<T>::Cholesky(common::Uplo aUpperLower, Matrix<T> &aMatrixAUV, Matrix<T> &aMatrixADiagonal,
                                    Matrix<T> &aMatrixARK, int aRank, int aMaxRank, double aAccuracy,
                                    runtime::HicmaContext &aContext) {

            return hicmapp::operations::MatrixOperations<T>::Cholesky(aUpperLower, aMatrixAUV, aMatrixADiagonal,
                                                                      aMatrixARK, aRank, aMaxRank, aAccuracy,
                                                                      aContext);

        }

        template<typename T>
        size_t
        Hicmapp<T>::DiagVecToMat(Matrix<T> &aMatrixDiag, Matrix<T> &aMatrixDense, runtime::HicmaContext &aContext) {

            return hicmapp::operations::MatrixOperations<T>::DiagVecToMat(aMatrixDiag, aMatrixDense, aContext);
        }

        HICMAPP_INSTANTIATE_CLASS(Hicmapp)
    }
