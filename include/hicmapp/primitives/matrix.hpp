
#ifndef HICMAPP_PRIMITIVES_MATRIX_HPP
#define HICMAPP_PRIMITIVES_MATRIX_HPP

#include <cstddef>
#include <vector>

#include <hicmapp/common/definitions.h>
#include <hicmapp/runtime/interface/HicmaContext.hpp>
#include "submatrix.hpp"
#include <hicmapp/primitives/decomposer/matrix_decomposer.hpp>
#include "hicmapp/primitives/decomposer/concrete/slowest_dimension_decomposer.hpp"
#include <hcorepp/helpers/RawMatrix.hpp>
#include <memory>

namespace hicmapp::primitives {
/**
 *  Tile matrix descriptor
 *
 *  Matrices are stored in a contiguous data chunk containning in order
 *  A11, A21, A12, A22 with :
 *
 *           n1      n2
 *      +----------+---+
 *      |          |   |    With m1 = lm - (lm%mb)
 *      |          |   |         m2 = lm%mb
 *  m1  |    A11   |A12|         n1 = ln - (ln%nb)
 *      |          |   |         n2 = ln%nb
 *      |          |   |
 *      +----------+---+
 *  m2  |    A21   |A22|
 *      +----------+---+
 *
 */

        template<typename T>
        class Matrix {
        public:

            /**
             * Matrix Class constructor.
             *
             * @param apMatrixData
             * Pointer to Matrix data elements, data will be distributed across submatrices, each containing a 2d array of
             * tiles and size of each tile.
             * @param aTotalGlobalNumOfRows
             * Total Num of Rows in Global Matrix.
             * @param aTotalGlobalNumOfCols
             * Total NUm of Cols in Global Matrix.
             * @param aTileNumOfRows
             * Num of Rows in each tile, except the remainder tile if it exists.
             * @param aTileNumOfCols
             * Num of cols in each tile, except the remainder tile if it exists.
             * @param aStorageLayout
             * Storage Layout in memory.
             * @param aMatrixDecomposer
             * Customized matrix decomposer.
             * @param aRank
             * Matrix rank.
             * @param aCommunicator
             */
            Matrix(T *apMatrixData, size_t aTotalGlobalNumOfRows, size_t aTotalGlobalNumOfCols, size_t aTileNumOfRows,
                   size_t aTileNumOfCols, common::StorageLayout aStorageLayout, MatrixDecomposer &aMatrixDecomposer,
                   hicmapp::runtime::HicmaContext &aContext,
                   size_t aRank = 0, bool aDiagonalMatrix = false);

            /**
                 * Matrix Class constructor, uses a default SlowestDimDecomposer.
                 *
                 * @param apMatrixData
                 * Pointer to Matrix data elements, data will be distributed across submatrices, each containing a 2d array of
                 * tiles and size of each tile.
                 * @param aTotalGlobalNumOfRows
                 * Total Num of Rows in Global Matrix.
                 * @param aTotalGlobalNumOfCols
                 * Total NUm of Cols in Global Matrix.
                 * @param aTileNumOfRows
                 * Num of Rows in each tile, except the remainder tile if it exists.
                 * @param aTileNumOfCols
                 * Num of cols in each tile, except the remainder tile if it exists.
                 * @param aStorageLayout
                 * Storage Layout in memory.
                 * @param aRank
                 * Matrix rank.
                 * @param aCommunicator
                 */
            Matrix(T *apMatrixData, size_t aTotalGlobalNumOfRows, size_t aTotalGlobalNumOfCols, size_t aTileNumOfRows,
                   size_t aTileNumOfCols, common::StorageLayout aStorageLayout, runtime::HicmaContext &aContext,
                   size_t aRank = 0, bool aDiagonalMatrix = false);

            /**
             * Matrix Class constructor, uses a default SlowestDimDecomposer.
             *
             * @param apMatrixData
             * Pointer to Matrix data elements, data will be distributed across submatrices, each containing a 2d array of
             * tiles and size of each tile.
             * @param aTotalGlobalNumOfRows
             * Total Num of Rows in Global Matrix.
             * @param aTotalGlobalNumOfCols
             * Total NUm of Cols in Global Matrix.
             * @param aTileNumOfRows
             * Num of Rows in each tile, except the remainder tile if it exists.
             * @param aTileNumOfCols
             * Num of cols in each tile, except the remainder tile if it exists.
             * @param aStorageLayout
             * Storage Layout in memory.
             * @param aRank
             * Matrix rank.
             * @param aCommunicator
             */
            Matrix(T *apMatrixData, size_t aTotalGlobalNumOfRows, size_t aTotalGlobalNumOfCols, size_t aTileNumOfRows,
                   size_t aTileNumOfCols, common::StorageLayout aStorageLayout, runtime::HicmaContext &aContext,
                   const CompressionParameters &aParams);

            /**
             * Matrix Class constructor.
             *
             * @param apMatrixData
             * Pointer to Matrix data elements, data will be distributed across submatrices, each containing a 2d array of
             * tiles and size of each tile.
             * @param aTotalGlobalNumOfRows
             * Total Num of Rows in Global Matrix.
             * @param aTotalGlobalNumOfCols
             * Total NUm of Cols in Global Matrix.
             * @param aTileNumOfRows
             * Num of Rows in each tile, except the remainder tile if it exists.
             * @param aTileNumOfCols
             * Num of cols in each tile, except the remainder tile if it exists.
             * @param aStorageLayout
             * Storage Layout in memory.
             * @param aMatrixDecomposer
             * Customized matrix decomposer.
             * @param aRank
             * Matrix rank.
             */
            Matrix(T *apMatrixData, size_t aTotalGlobalNumOfRows, size_t aTotalGlobalNumOfCols, size_t aTileNumOfRows,
                   size_t aTileNumOfCols, common::StorageLayout aStorageLayout, MatrixDecomposer &aMatrixDecomposer,
                   hicmapp::runtime::HicmaContext &aContext, const CompressionParameters &aParams);

            /**
             * Matrix destructor.
             */
            ~Matrix();

            /**
             * Get Current Matrix Id.
             *
             * @return
             * Matrix Id.
             */
            [[nodiscard]] size_t
            GetMatrixId() const;

            /**
             * Get number of sub-matrices assigned to current process
             *
             * @return
             * Number of Sub matrices.
             */
            [[nodiscard]] size_t
            GetNumOfSubMatrices() const;

            /**
             * Get total number of sub-matrices decomposed across all processes
             *
             * @return
             * Total number of Sub matrices.
             */
            [[nodiscard]] size_t
            GetTotalNumOfSubMatrices() const;

            /**
             * Get vector of pointers to Sub matrices.
             *
             * @return
             *
             */
            std::vector<SubMatrix<T> *> &
            GetSubMatrices();

            SubMatrix<T> &
            GetSubMatrix(size_t aSubMatrixIndex) const;

            [[nodiscard]] size_t
            GetNumOfGlobalTilesInRows() const;

            [[nodiscard]] size_t
            GetNumOfGlobalTilesInCols() const;

            [[nodiscard]] bool
            ContainsTile(size_t aTileIdxInRows, size_t aTileIdxInCols) const;

            Tile<T> *
            GetTilePointer(size_t aTileIdxInRows, size_t aTileIdxInCols);

            [[nodiscard]] common::StorageLayout
            GetStorageLayout() const;

            [[nodiscard]] int
            GetSubMatrixOwnerId(size_t aTileIdxInRows, size_t aTileIdxInCols) const;

            [[nodiscard]] int
            GetTileOwnerId(size_t aTileIdxInRows, size_t aTileIdxInCols) const;

            [[nodiscard]] runtime::HicmaContext &
            GetContext() const;

            [[nodiscard]] bool
            IsMatrixValid() const;

            [[nodiscard]] size_t GetNumOfRowsInTile() const;

            [[nodiscard]] size_t GetNumOfColsInTile() const;

            [[nodiscard]] size_t GetGlobalNumOfRowsInMatrix() const;

            [[nodiscard]] size_t GetGlobalNumOfColsInMatrix() const;

            size_t GetTileLeadingDim(size_t aTileIdx);

            hcorepp::helpers::RawMatrix<T>
            ToRawMatrix(runtime::HicmaContext &aContext);

            size_t
            GetMemoryFootprint();

            [[nodiscard]] const std::vector<MatrixSpecifications> &GetMatrixSpecs() const {
                return mSpecs;
            }

            [[nodiscard]] size_t GetMatrixFixedRank() const {
                return mFixedRank;
            }

            [[nodiscard]] TileType GetMatrixTileType() const {
                return mTileType;
            }

            TileMetadata *GetTileMetadata(size_t aTileRowIdx, size_t aTileColIdx);

            void Print(std::ostream &aOutStream);

        private:
            void
            Initialize(T *apMatrixData, size_t aTotalGlobalNumOfRows, size_t aTotalGlobalNumOfCols,
                       size_t aTileNumOfRows, size_t aTileNumOfCols, common::StorageLayout aStorageLayout,
                       MatrixDecomposer &aMatrixDecomposer, hicmapp::runtime::HicmaContext &aContext, size_t aRank = 0);

            void
            Initialize(T *apMatrixData, size_t aTotalGlobalNumOfRows, size_t aTotalGlobalNumOfCols,
                       size_t aTileNumOfRows, size_t aTileNumOfCols, common::StorageLayout aStorageLayout,
                       MatrixDecomposer &aMatrixDecomposer, hicmapp::runtime::HicmaContext &aContext,
                       const CompressionParameters &aParams = {1e-9});

        private:
            // 2d array of sub matrices representing the GLobal matrix.
            std::vector<SubMatrix<T> *> mSubMatrices;
            // number of rows in a tile.
            size_t mGlobalNumOfRowsInTile;
            // number of columns in a tile.
            size_t mGlobalNumOfColsInTile;
            // Total number of rows in global matrix.
            size_t mGlobalNumOfRowsInMatrix;
            // Total number of columns in global matrix.
            size_t mGlobalNumOfColsInMatrix;
            // Matrix Id
            size_t mMatrixId;
            // Global number of Tiles in Rows;
            size_t mGlobalNumOfTilesInRows;
            // Global number of Tiles in Cols;
            size_t mGlobalNumOfTilesInCols;
            // Storage layout
            common::StorageLayout mStorageLayout;
            //Hicma context
            runtime::HicmaContext &mContext;
            //Memory Footprint
            size_t mMemory;
            // MatrixDecomposerType
            DecomposerType mDecomposerType = SLOWESTDIM;
            // Decomposition Specs
            std::vector<MatrixSpecifications> mSpecs;
            size_t mFixedRank = -1;
            TileType mTileType;
            bool mDiagonalMatrix;
        };
    }
#endif //HICMAPP_PRIMITIVES_MATRIX_HPP
