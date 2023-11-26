
#ifndef HICMAPP_PRIMITIVES_DECOMPOSER_MATRIX_DECOMPOSER_HPP
#define HICMAPP_PRIMITIVES_DECOMPOSER_MATRIX_DECOMPOSER_HPP

#include <cstddef>
#include <utility>
#include <vector>

namespace hicmapp::primitives {

        enum DecomposerType {
            SLOWESTDIM = 0,
            CYCLIC2D = 1,
        };

        class MatrixSpecifications {

        public:
            MatrixSpecifications(size_t aNumOfTilesInRow, size_t aNumOfTilesInCol, size_t aStartingIndexInRows,
                                 size_t aStartingIndexInCols, size_t aNumberOfProcessInRow,
                                 size_t aNumberOfProcessInCol, int aOwnerId) : mNumOfTilesInRow(aNumOfTilesInRow),
                                                                               mNumOfTilesInCol(aNumOfTilesInCol),
                                                                               mStartingIndexInRows(
                                                                                       aStartingIndexInRows),
                                                                               mStartingIndexInCols(
                                                                                       aStartingIndexInCols),
                                                                               mNumberOfProcessInRow(
                                                                                       aNumberOfProcessInRow),
                                                                               mNumberOfProcessInCol(
                                                                                       aNumberOfProcessInCol),
                                                                               mOwnerId(aOwnerId) {

            }

            MatrixSpecifications() = default;

            ~MatrixSpecifications() = default;

            [[nodiscard]] size_t GetNumOfTilesInRow() const {
                return mNumOfTilesInRow;
            }

            [[nodiscard]] size_t GetNumOfTilesInCol() const {
                return mNumOfTilesInCol;
            }

            [[nodiscard]] size_t GetStartingIndexInRows() const {
                return mStartingIndexInRows;
            }

            [[nodiscard]] size_t GetStartingIndexInCols() const {
                return mStartingIndexInCols;
            }

            [[nodiscard]] size_t GetNumberOfProcessInRow() const {
                return mNumberOfProcessInRow;
            }

            [[nodiscard]] size_t GetNumberOfProcessInCol() const {
                return mNumberOfProcessInCol;
            }

            void SetNumOfTilesInRow(size_t aNumOfTileRows) {
                mNumOfTilesInRow = aNumOfTileRows;
            }

            void SetNumOfTilesInCol(size_t aNumOfTileCols) {
                mNumOfTilesInCol = aNumOfTileCols;
            }

            void SetStartingIndexInRows(size_t aStartingIndexInRows) {
                mStartingIndexInRows = aStartingIndexInRows;
            }

            void SetStartingIndexInCols(size_t aStartingIndexInCols) {
                mStartingIndexInCols = aStartingIndexInCols;
            }

            void SetNumberOfProcessInRow(size_t aNumberOfProcessInRow) {
                mNumberOfProcessInRow = aNumberOfProcessInRow;
            }

            void SetNumberOfProcessInCol(size_t aNumberOfProcessInCol) {
                mNumberOfProcessInCol = aNumberOfProcessInCol;
            }

            void SetOwnerId(int aOwnerId) {
                mOwnerId = aOwnerId;
            }

            [[nodiscard]] int GetOwnerId() const {
                return mOwnerId;
            }

            void SetTotalSubMatrixNumOfElementsInRows(size_t aTotalSubMatrixNumOfElementsInRows) {
                mTotalSubMatrixNumOfElementsInRows = aTotalSubMatrixNumOfElementsInRows;
            }

            void SetTotalSubMatrixNumOfElementsInCols(size_t aTotalSubMatrixNumOfElementsInCols) {
                mTotalSubMatrixNumOfElementsInCols = aTotalSubMatrixNumOfElementsInCols;
            }

            [[nodiscard]] size_t GetTotalSubMatrixNumOfElementsInRows() const {
                return mTotalSubMatrixNumOfElementsInRows;
            }

            [[nodiscard]] size_t GetTotalSubMatrixNumOfElementsInCols() const {
                return mTotalSubMatrixNumOfElementsInCols;
            }

        private:
            size_t mNumOfTilesInRow{};
            size_t mNumOfTilesInCol{};
            /// Tile index in rows.
            size_t mStartingIndexInRows{};
            /// Tile index in columns.
            size_t mStartingIndexInCols{};
            size_t mNumberOfProcessInRow = 0;
            size_t mNumberOfProcessInCol = 0;
            int mOwnerId = 0;
            size_t mTotalSubMatrixNumOfElementsInRows{};
            size_t mTotalSubMatrixNumOfElementsInCols{};
        };

        class MatrixDecomposer {
        public:

            MatrixDecomposer() = default;

            virtual std::vector<MatrixSpecifications>
            Decompose(size_t aGlobalMatrixTilesInRows, size_t aGlobalMatrixTilesinCols,
                      bool aDiagonalMatrix = false) = 0;

            virtual
            DecomposerType
            GetType() = 0;
        };
    }
#endif //HICMAPP_PRIMITIVES_DECOMPOSER_MATRIX_DECOMPOSER_HPP
