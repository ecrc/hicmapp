#ifndef HICMAPP_TILE_OPERATIONS_HPP
#define HICMAPP_TILE_OPERATIONS_HPP

#include <hcorepp/operators/concrete/Compressed.hpp>
#include <hcorepp/operators/concrete/Dense.hpp>

namespace hicmapp::operations {
    template<typename T>
    class TileOperations {

    public:

        /**
         * Generates a single Compressed Tile based on specific Compression Parameters (Tolerance, Fixed Rank etc.)
         * @param[out] aCompressedTile CompressedTile to be Generated
         * @param[in] aTileRowIdx Index of tile in global matrix rows needed by STARSH generation
         * @param[in] aTileColIdx Index of tile in global matrix cols needed by STARSH generation
         * @param[in] aSVDArguments Compression Parameters
         * @return
         */
        static int
        GenerateCompressedMatrix(hcorepp::operators::CompressedTile<T> &aCompressedTile,
                                 size_t aTileRowIdx, size_t aTileColIdx,
                                 const hcorepp::operators::CompressionParameters &aSVDArguments);

        /**
         * Generates a single Dense Tile
         * @param[out] aDenseTile Dense Tile to be generated
         * @param[in] aTileRowIdx Index of tile in global matrix rows needed by STARSH generation
         * @param[in] aTileColIdx Index of tile in global matrix cols needed by STARSH generation
         * @return
         */
        static int
        GenerateDenseTile(hcorepp::operators::DenseTile<T> &DenseTile, size_t aTileRowIdx, size_t aTileColIdx);

        /***
         * Uncompress U and V matrices into a single dense buffer
         * @param aNumOfRows Num of Rows
         * @param aNumOfCols Num of Cols
         * @param aAlpha alpha factor
         * @param apAU U submatrix
         * @param apArk Rank matrix
         * @param aLeadingDimA Leading dim of U
         * @param apBV V submatrix
         * @param aLeadingDimB Leading Dimension of B
         * @param aBeta beta factor
         * @param apC Allocated dense buffer
         * @param aLeadingDimC Leading dim of dense buffer
         * @return
         */
        static int
        UnCompressTile(size_t aNumOfRows, size_t aNumOfCols, double aAlpha, const T *apAU,
                       const T *apArk,
                       size_t aLeadingDimA, const T *apBV, size_t aLeadingDimB,
                       double aBeta,
                       T *apC, size_t aLeadingDimC);


        /**
         * Computes Gemm between two tile objects A and B into the output tile object C where
         * C = alpha * A * B + beta * C
         * @param aAlpha alpha factor
         * @param aTileA Input Tile A
         * @param aAOp Operation to be performed on Tile A
         * @param aTileB Input Tile B
         * @param aBOp Operation to be performed on Tile B
         * @param aBeta beta factor
         * @param[out] aTileC Output Tile
         * @param aContext Hcorepp context
         * @param aMemoryUnit Memory Unit for intermediate allocations
         * @param aSVDArguments Compression Parameters for Compressed Tiles
         * @param aCholesky Flag specifying if the gemm oepration was done in the context of cholesky
         * @return Flops
         */
        static size_t
        Gemm(T aAlpha, hcorepp::operators::Tile<T> const &aTileA, blas::Op const &aAOp,
             hcorepp::operators::Tile<T> const &aTileB, blas::Op const &aBOp, T aBeta,
             hcorepp::operators::Tile<T> &aTileC, const hcorepp::kernels::RunContext &aContext,
             hcorepp::dataunits::MemoryUnit<T> &aMemoryUnit,
             const hcorepp::operators::CompressionParameters &aSVDArguments = {1e-9}, bool aCholesky = false);


        /***
         * Syrk operation to be performed on tiles A and B
         * @param aAlpha alpha factor
         * @param aA Input tile A
         * @param aAOp Operation to be performed on Tile A
         * @param aUplo Upper, Lower , or UpperLower
         * @param aBeta beta factor
         * @param[out] aB Output Tile B
         * @param aContext Hcorepp context to be used
         * @param aMemoryUnit Memory unit for intermediate allocations
         * @return flops
         */
        static size_t
        Syrk(T aAlpha, const hcorepp::operators::Tile<T> &aA, const blas::Op &aAOp, blas::Uplo aUplo,
             T aBeta, hcorepp::operators::Tile<T> &aB, const hcorepp::kernels::RunContext &aContext,
             hcorepp::dataunits::MemoryUnit<T> &aMemoryUnit);


        /***
         * Potrf operation to be performed on tiles A
         * @param aAlpha alpha factor
         * @param aA Input tile A
         * @param aAOp Operation to be performed on Tile A
         * @param aUplo Upper, Lower , or UpperLower
         * @param aBeta beta factor
         * @param[out] aB Output Tile B
         * @param aContext Hcorepp context to be used
         * @param aMemoryUnit Memory unit for intermediate allocations
         * @return flops
         */
        static size_t
        Potrf(hcorepp::operators::Tile<T> &aA, blas::Uplo aUplo, const hcorepp::kernels::RunContext &aContext,
              hcorepp::dataunits::MemoryUnit<T> &aMemoryUnit);


        /***
         * Trsm operation to be performed on Tiles A and B
         * @param aSide Left or Right Sided Operation
         * @param aUplo Upper, Lower or UpperLower
         * @param aTrans Transpose operation to be performed
         * @param aDiag NonUnit or Unit Diagonal
         * @param aAlpha alpha factor
         * @param aA Tile A
         * @param aB Tile B
         * @param aContext Hcorepp context to be used
         * @param aMemoryUnit  Memory unit for intermediate allocations
         * @return flops
         */
        static size_t
        Trsm(blas::Side aSide, blas::Uplo aUplo, blas::Op aTrans, blas::Diag aDiag, T aAlpha,
             hcorepp::operators::Tile<T> &aA, hcorepp::operators::Tile<T> &aB,
             const hcorepp::kernels::RunContext &aContext, hcorepp::dataunits::MemoryUnit<T> &aMemoryUnit);

        /***
         * Generate Diagonal Tiles
         * @param aAUV Compressed AUV tile
         * @param aRanks Ranks of A matrix
         * @param aDenseA Dense A Tile
         * @param aDiagonalA Diagonal Tiles in A
         * @param aTileRowIdx Row Index of Tile A
         * @param aTileColIdx Column Index of Tile A
         * @param aSeed Seed for Generation
         * @param aMaxRank Maximum Rank
         * @param aTolerance Tolerance and Accuracy for Compression
         * @param aCompressDiagonal Boolean whether diagonal to be compressed or not
         * @param aLeadingDimA Leading Dimension of A tile
         * @param aLeadingDimU Leading Dimension of U submatrix
         * @param aLeadingDimV Leading Dimension of V submatrix
         * @param aRows Number of rows
         * @param aCols Number of columns in A
         * @param aContext Hcorepp Context
         * @return Flops
         */
        static size_t
        GenerateDiagonalTile(hcorepp::operators::Tile<T> *aAUV, hcorepp::operators::Tile<T> *aRanks,
                             hcorepp::operators::Tile<T> *aDenseA, hcorepp::operators::Tile<T> *aDiagonalA,
                             int aTileRowIdx, int aTileColIdx, unsigned long long int aSeed,
                             int aMaxRank, double aTolerance, int aCompressDiagonal,
                             int aLeadingDimA, int aLeadingDimU, int aLeadingDimV, int aRows,
                             int aCols, const hcorepp::kernels::RunContext &aContext);


        /***
         * Copies Data from Tile A into Tile B
         * @param aRows Number of rows to copy
         * @param aCols Number of Columns to copy
         * @param aA Tile to be copied from
         * @param aB Tile be copied into
         * @param aContext Hcorepp context
         * @return
         */
        static size_t
        LaCpy(int aRows, int aCols, const hcorepp::operators::Tile<T> &aA, hcorepp::operators::Tile<T> &aB,
              const hcorepp::kernels::RunContext &aContext);

    private:
        /**
         * @brief
         * Prevent Class Instantiation for Operations Wrapper Class.
         */
        TileOperations() = default;

    };
}
#endif //HICMAPP_TILE_OPERATIONS_HPP
