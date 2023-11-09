
#ifndef HICMAPP_PRIMITIVES_SUBMATRIX_HPP
#define HICMAPP_PRIMITIVES_SUBMATRIX_HPP

#include <hcorepp/operators/concrete/Dense.hpp>
#include <hcorepp/operators/concrete/Compressed.hpp>
#include <hicmapp/common/definitions.h>
#include <hicmapp/runtime/interface/HicmaContext.hpp>

using namespace hcorepp::operators;

namespace hicmapp::primitives {

        template<typename T>
        class SubMatrix {
        public:

            /**
             * SubMatrix Class constructor.
             *
             * @param apSubMatrixData
             * Pointer to the submatrix data elements, containing a 2d array of tiles to be holding the data.
             * @param aTileNumOfRows
             * Num of Rows in each tile, except the remainder tile if it exists.
             * @param aTileNumOfCols
             * Num of cols in each tile, except the remainder tile if it exists.
             * @param aGlobalMatrixRows
             * Num of global matrix rows.
             * @param aGlobalMatrixCols
             * Num of global matrix cols.
             * @param aSubMatrixNumOfRows
             * Num of sub-matrix rows.
             * @param aSubMatrixNumOfCols
             * Num of sub-matrix cols.
             * @param aTilesGlobalStIdxInRows
             * Tile's global index across the matrix in rows.
             * @param aTilesGlobalStIdxInCols
             * Tile's global index across the matrix in cols.
             * @param aStorageLayout
             * Storage Layout in memory.
             * @param aOwnerId
             * Sub-matrix' owner ID.
             * @param aRank
             * Sub-matrix' rank.
             */
            SubMatrix(T *apSubMatrixData, size_t aTileNumOfRows, size_t aTileNumOfCols, size_t aGlobalMatrixRows,
                      size_t aGlobalMatrixCols, size_t aSubMatrixNumOfRows, size_t aSubMatrixNumOfCols,
                      size_t aTilesGlobalStIdxInRows, size_t aTilesGlobalStIdxInCols,
                      common::StorageLayout aStorageLayout, size_t aOwnerId, runtime::HicmaContext& aContext, size_t aRank = 0);

            /**
             * SubMatrix Class constructor.
             *
             * @param apSubMatrixData
             * Pointer to the submatrix data elements, containing a 2d array of tiles to be holding the data.
             * @param aTileNumOfRows
             * Num of Rows in each tile, except the remainder tile if it exists.
             * @param aTileNumOfCols
             * Num of cols in each tile, except the remainder tile if it exists.
             * @param aGlobalMatrixRows
             * Num of global matrix rows.
             * @param aGlobalMatrixCols
             * Num of global matrix cols.
             * @param aSubMatrixNumOfRows
             * Num of sub-matrix rows.
             * @param aSubMatrixNumOfCols
             * Num of sub-matrix cols.
             * @param aTilesGlobalStIdxInRows
             * Tile's global index across the matrix in rows.
             * @param aTilesGlobalStIdxInCols
             * Tile's global index across the matrix in cols.
             * @param aStorageLayout
             * Storage Layout in memory.
             * @param aOwnerId
             * Sub-matrix' owner ID.
             * @param aRank
             * Sub-matrix' rank.
             */
            SubMatrix(T *apSubMatrixData, size_t aTileNumOfRows, size_t aTileNumOfCols, size_t aGlobalMatrixRows,
                      size_t aGlobalMatrixCols, size_t aSubMatrixNumOfRows, size_t aSubMatrixNumOfCols,
                      size_t aTilesGlobalStIdxInRows, size_t aTilesGlobalStIdxInCols,
                      common::StorageLayout aStorageLayout, size_t aOwnerId, runtime::HicmaContext& aContext, const CompressionParameters& aParams);

            /**
             * Sub Matrix destructor.
             */
            ~SubMatrix();

            /**
             * Get number of tiles in a sub-matrix
             *
             * @return
             * Number of tiles
             */
            size_t
            GetNumberofTiles();

            /**
             * Get reference to sub-matrix tiles.
             *
             * @return
             * Sub matrix tiles
             */
            std::vector<Tile<T> *> &
            GetTiles();

            bool
            ContainsTile(size_t aTileIdxInRows, size_t aTileIdxInCols);

            Tile<T> *
            GetTilePointer(size_t aTileIdxInRows, size_t aTileIdxInCols);

            int
            GetSubMatrixOwnerId();

            bool
            IsValid();

            size_t
            GetNumOfTilesinRows();

            size_t
            GetNumOfTilesinCols();

            size_t
            GetTileRows();

            size_t
            GetTileCols();

            size_t
            GetTilesGlobalStIdxInRows();

            size_t
            GetTilesGlobalStIdxInCols();

            size_t
            GetMemoryFootprint();

        private:
            // 2d array of tiles representing the matrix.
            std::vector<Tile<T> *> mTiles;
            // storage layout.
            common::StorageLayout mStorageLayout;
            // number of rows in a single tile.
            size_t mTileRows;
            // number of columns in a single tile.
            size_t mTileCols;
            // Total number of rows in sub-matrix.
            size_t mSubMatrixRows;
            // Total number of columns in sub-matrix.
            size_t mSubMatrixCols;

            size_t mGlobalMatrixRows;
            size_t mGlobalMatrixCols;

            size_t mTilesGlobalStIdxInRows;
            size_t mTilesGlobalStIdxInCols;

            size_t mSubMatrixOwnerId;

            size_t mNumOfTilesinRows;
            size_t mNumOfTilesinCols;

            size_t mMemory;
        };
    }
#endif //HICMAPP_PRIMITIVES_SUBMATRIX_HPP
