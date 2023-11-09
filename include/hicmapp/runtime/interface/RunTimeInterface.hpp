#ifndef HICMAPP_RUNTIME_RUN_TIME_INTERFACE_HPP
#define HICMAPP_RUNTIME_RUN_TIME_INTERFACE_HPP

#include <hicmapp/primitives/matrix.hpp>
#include <hicmapp/runtime/interface/HicmaHardware.hpp>

using namespace hicmapp::primitives;

namespace hicmapp::runtime {

    /***
     * This is the interface that should be supported by all runtime libraries
     * @tparam T
     */
        template<typename T>
        class RunTimeInterface {

        public:
            /***
             * Default Constructor
             */
            RunTimeInterface() = default;
            /***
             * Default Destructor
             */
            ~RunTimeInterface() = default;

            /***
             * Dense Matrix Generation
             * @param aMatrix Full Matrix
             * @param aTileIdxInRows Index of Tile in Rows to be Generated
             * @param aTileIdxInCols Index of Tile in Cols to be Generated
             * @return Error Code
             */
            virtual int
            GenerateDenseMatrix(Matrix<T> &aMatrix, size_t aTileIdxInRows, size_t aTileIdxInCols) = 0;

            /***
             * Compressed Matrix Generation
             * @param aMatrix Full Matrix
             * @param aTileIdxInRows Index of Tile in Rows to be Generated
             * @param aTileIdxInCols Index of Tile in Cols to be Generated
             * @param aSVDArguments Compression Parameters
             * @return
             */
            virtual int
            GenerateCompressedMatrix(Matrix<T> &aMatrix, size_t aTileIdxInRows, size_t aTileIdxInCols,
                                     const CompressionParameters &aSVDArguments) = 0;

            /***
             * Gemm Operation
             * @param aAlpha
             * @param aMatrixA
             * @param aRowIdxA
             * @param aColIdxA
             * @param aAOp
             * @param aMatrixB
             * @param aRowIdxB
             * @param aColIdxB
             * @param aBOp
             * @param aBeta
             * @param aMatrixC
             * @param aRowIdxC
             * @param aColIdxC
             * @param aContext
             * @param aSVDArguments
             * @param aMemoryUnit
             * @param aCholesky
             * @return
             */
            virtual size_t
            Gemm(T aAlpha, Matrix<T> &aMatrixA, const size_t &aRowIdxA, const size_t &aColIdxA, const blas::Op &aAOp,
                 Matrix<T> &aMatrixB, const size_t &aRowIdxB, const size_t &aColIdxB, const blas::Op &aBOp, T aBeta,
                 Matrix<T> &aMatrixC, const size_t &aRowIdxC, const size_t &aColIdxC,
                 const hcorepp::kernels::RunContext &aContext, const CompressionParameters &aSVDArguments,
                 hcorepp::dataunits::MemoryUnit<T> &aMemoryUnit, bool aCholesky = false) = 0;

            /***
             * Syrk Operation
             * @param aMatrixA
             * @param aRowIdxA
             * @param aColIdxA
             * @param aAOp
             * @param aMatrixC
             * @param aRowIdxC
             * @param aColIdxC
             * @param aUplo
             * @param aAlpha
             * @param aBeta
             * @param aContext
             * @param aMemoryUnit
             * @return
             */
            virtual size_t
            Syrk(Matrix<T> &apMatrixA, const size_t &aRowIdxA,
                 const size_t &aColIdxA, const blas::Op &aAOp, Matrix<T> &apMatrixC,
                 const size_t &aRowIdxC, const size_t &aColIdxC, blas::Uplo aUplo, T aAlpha,
                 T aBeta, const hcorepp::kernels::RunContext &aContext,
                 hcorepp::dataunits::MemoryUnit<T> &aMemoryUnit) = 0;

            /***
             *
             * @param aMatrixA
             * @param aRowIdxA
             * @param aColIdxA
             * @param aUplo
             * @param aContext
             * @param aMemoryUnit
             * @return
             */
            virtual size_t
            Potrf(Matrix<T> &aMatrixA, const size_t &aRowIdxA, const size_t &aColIdxA, blas::Uplo aUplo,
                  const hcorepp::kernels::RunContext &aContext, hcorepp::dataunits::MemoryUnit<T> &aMemoryUnit) = 0;

            /***
             * Trsm operation
             * @param aSide
             * @param aUplo
             * @param aTrans
             * @param aDiag
             * @param aAlpha
             * @param aMatrixA
             * @param aRowIdxA
             * @param aColIdxA
             * @param aMatrixB
             * @param aRowIdxB
             * @param aColIdxB
             * @param aContext
             * @param aMemoryUnit
             * @return
             */
            virtual size_t
            Trsm(blas::Side aSide, blas::Uplo aUplo, blas::Op aTrans, blas::Diag aDiag, T aAlpha,
                 Matrix<T> &aMatrixA, const size_t &aRowIdxA, const size_t &aColIdxA,
                 Matrix<T> &aMatrixB, const size_t &aRowIdxB, const size_t &aColIdxB,
                 const hcorepp::kernels::RunContext &aContext,
                 hcorepp::dataunits::MemoryUnit<T> &aMemoryUnit) = 0;

            /***
             * Generation of Diagonal Tiles
             * @param aMatrixUV
             * @param aMatrixDiag
             * @param aRowIdxDiag
             * @param aColIdxDiag
             * @param apMatrixRK
             * @param aRowIdx
             * @param aColIdx
             * @param seed
             * @param maxrank
             * @param tol
             * @param compress_diag
             * @param aMatrixDense
             * @param aContext
             * @param diagonal_tile
             * @return
             */
            virtual size_t
            GenerateDiagonalTile(Matrix<T>& aMatrixUV, Matrix<T>& aMatrixDiag, const size_t &aRowIdxDiag,
                                 const size_t &aColIdxDiag, Matrix<T>& apMatrixRK, const size_t &aRowIdx,
                                 const size_t &aColIdx, unsigned long long int seed, size_t maxrank, double tol,
                                 size_t compress_diag, Matrix<T>& aMatrixDense,
                                 const hcorepp::kernels::RunContext &aContext, bool diagonal_tile) = 0;

            /***
             * Copy Operation
             * @param aMatrixA
             * @param aRowIdxA
             * @param aColIdxA
             * @param aMatrixB
             * @param aRowIdxB
             * @param aColIdxB
             * @param aContext
             * @return
             */
            virtual size_t
            LaCpy(Matrix<T> &aMatrixA, const size_t &aRowIdxA, const size_t &aColIdxA,
                  Matrix<T> &aMatrixB, const size_t &aRowIdxB, const size_t &aColIdxB,
                  const hcorepp::kernels::RunContext &aContext) = 0;

            /***
             * Uncompress Operation
             * @param aMatrixUV
             * @param aMatrixDense
             * @param pMatrixRk
             * @param aRowIdx
             * @param aColIdx
             * @return
             */
            virtual size_t
            Uncompress(Matrix<T> &aMatrixUV, Matrix<T> &aMatrixDense, Matrix<T> &pMatrixRk, const size_t &aRowIdx,
                       const size_t &aColIdx) = 0;

            /***
             * Synchronize. Waits on all Tasks and places an MPI barrier if MPI is enabled (for StarPU)
             * @return
             */
            virtual int Sync() = 0;

            /***
             * Flush specific tile data
             * @param aMatrix
             * @param aRowIdx
             * @param aColIdx
             */
            virtual void Flush(const Matrix<T> &aMatrix, size_t aRowIdx, size_t aColIdx) = 0;

            /***
             * Flush entire matrix data
             * @param aMatrix
             */
            virtual void Flush(const Matrix<T> &apMatrix) = 0;

            /***
             * Finalize Runtime
             */
            virtual void Finalize() = 0;

            /***
             * Register all tiles of Matrix A
             * @param A
             */
            virtual void RegisterHandles(Matrix<T> &A) = 0;

            /***
             * Unregister all tiles of Matrix A
             * @param A
             */
            virtual void UnRegisterHandles(Matrix<T> &A) = 0;

            /***
             * Getter for Library Type
             * @return
             */
            virtual common::RunTimeLibrary LibraryType() = 0;
        };
    }
#endif //HICMAPP_RUNTIME_RUN_TIME_INTERFACE_HPP
