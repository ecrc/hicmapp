
#include <hicmapp/primitives/submatrix.hpp>
#include <hcorepp/kernels/memory.hpp>

namespace hicmapp::primitives {

    template<typename T>
    SubMatrix<T>::SubMatrix(T *apSubMatrixData, size_t aTileNumOfRows, size_t aTileNumOfCols, size_t aGlobalMatrixRows,
                            size_t aGlobalMatrixCols, size_t aSubMatrixNumOfRows, size_t aSubMatrixNumOfCols,
                            size_t aTilesGlobalStIdxInRows, size_t aTilesGlobalStIdxInCols,
                            common::StorageLayout aStorageLayout, size_t aOwnerId, runtime::HicmaContext &aContext,
                            size_t aRank) {
        mStorageLayout = aStorageLayout;
        mSubMatrixRows = aSubMatrixNumOfRows;
        mSubMatrixCols = aSubMatrixNumOfCols;
        mTileRows = aTileNumOfRows;
        mTileCols = aTileNumOfCols;
        mGlobalMatrixRows = aGlobalMatrixRows;
        mGlobalMatrixCols = aGlobalMatrixCols;
        // with respect to the tiles decomposed within the global matrix
        mTilesGlobalStIdxInRows = aTilesGlobalStIdxInRows;
        mTilesGlobalStIdxInCols = aTilesGlobalStIdxInCols;

        if (aRank < 0 || !IsValid()) {
            throw std::invalid_argument("Matrix::Initialize Invalid Matrix Initialization");
        }

        if (mTileCols == 0 || mTileRows == 0) {
            throw std::runtime_error("SubMatrix::SubMatrix, Division by zero.\n");
        }
        mNumOfTilesinRows = (mSubMatrixRows + mTileRows - 1) / mTileRows;
        mNumOfTilesinCols = (mSubMatrixCols + mTileCols - 1) / mTileCols;

        mSubMatrixOwnerId = aOwnerId;

        size_t data_offset = 0;

        auto slow_dim_total_num_of_elements_in_matrix = 0;
        auto fast_dim_total_num_of_elements_in_matrix = 0;
        auto slow_dim_total_num_of_elements_in_submatrix = 0;
        auto fast_dim_total_num_of_elements_in_submatrix = 0;
        auto slow_dim_tile_num_of_elements = 0;
        auto fast_dim_tile_num_of_elements = 0;
        auto num_of_rows_in_tile = mTileRows;
        auto num_of_cols_in_tile = mTileCols;
        auto layout = blas::Layout::RowMajor;
        auto leading_dim = num_of_rows_in_tile;
        size_t initial_global_offset = 0;

        if (mStorageLayout == common::StorageLayout::HicmaCM) {
            slow_dim_total_num_of_elements_in_submatrix = mSubMatrixCols;
            fast_dim_total_num_of_elements_in_submatrix = mSubMatrixRows;
            slow_dim_tile_num_of_elements = mTileCols;
            fast_dim_tile_num_of_elements = mTileRows;
            layout = blas::Layout::ColMajor;
            slow_dim_total_num_of_elements_in_matrix = mGlobalMatrixCols;
            fast_dim_total_num_of_elements_in_matrix = mGlobalMatrixRows;
            initial_global_offset =
                    mTilesGlobalStIdxInCols * mTileCols * mGlobalMatrixRows + mTilesGlobalStIdxInRows * mTileRows;
        } else if (mStorageLayout == common::StorageLayout::HicmaRM) {
            slow_dim_total_num_of_elements_in_submatrix = mSubMatrixRows;
            fast_dim_total_num_of_elements_in_submatrix = mSubMatrixCols;
            slow_dim_tile_num_of_elements = mTileRows;
            fast_dim_tile_num_of_elements = mTileCols;
            layout = blas::Layout::RowMajor;
            slow_dim_total_num_of_elements_in_matrix = mGlobalMatrixRows;
            fast_dim_total_num_of_elements_in_matrix = mGlobalMatrixCols;
            initial_global_offset =
                    mTilesGlobalStIdxInRows * mTileRows * mGlobalMatrixCols + mTilesGlobalStIdxInCols * mTileCols;
        }

        size_t global_offset = 0;

        for (size_t i = 0; i < slow_dim_total_num_of_elements_in_submatrix; i += slow_dim_tile_num_of_elements) {
            global_offset = (i * fast_dim_total_num_of_elements_in_matrix) + initial_global_offset;
            data_offset = global_offset;

            for (size_t j = 0; j < fast_dim_total_num_of_elements_in_submatrix; j += fast_dim_tile_num_of_elements) {
                if (mStorageLayout == common::StorageLayout::HicmaCM) {
                    num_of_cols_in_tile = std::min(mTileCols, slow_dim_total_num_of_elements_in_submatrix - i);
                    num_of_rows_in_tile = std::min(mTileRows, fast_dim_total_num_of_elements_in_submatrix - j);
                    leading_dim = num_of_rows_in_tile;
                } else if (mStorageLayout == common::StorageLayout::HicmaRM) {
                    num_of_cols_in_tile = std::min(mTileCols, fast_dim_total_num_of_elements_in_submatrix - j);
                    num_of_rows_in_tile = std::min(mTileRows, slow_dim_total_num_of_elements_in_submatrix - i);
                    leading_dim = num_of_cols_in_tile;
                }

                if (apSubMatrixData == nullptr) {
                    if (aRank > 0) {
                        mTiles.push_back(
                                new CompressedTile<T>(num_of_rows_in_tile, num_of_cols_in_tile, nullptr, leading_dim,
                                                      aRank, layout,
                                                      aContext.GetMainContext()));
                    } else {
                        mTiles.push_back(
                                new DenseTile<T>(num_of_rows_in_tile, num_of_cols_in_tile, nullptr, leading_dim, layout,
                                                 aContext.GetMainContext()));
                    }
                    continue;
                }

                if (apSubMatrixData != nullptr) {
                    auto data_array = hcorepp::memory::AllocateArray<T>(num_of_rows_in_tile * num_of_cols_in_tile,
                                                                        aContext.GetMainContext());
                    auto array_off = 0;
                    auto temp_offset = data_offset;
                    if (mStorageLayout == common::StorageLayout::HicmaCM) {
                        for (size_t col = 0; col < num_of_cols_in_tile; col++) {
                            hcorepp::memory::Memcpy<T>(&data_array[array_off], &apSubMatrixData[temp_offset],
                                                       num_of_rows_in_tile, aContext.GetMainContext(),
                                                       hcorepp::memory::MemoryTransfer::DEVICE_TO_DEVICE);
                            temp_offset += mGlobalMatrixRows;
                            array_off += num_of_rows_in_tile;
                        }
                    } else if (mStorageLayout == common::StorageLayout::HicmaRM) {
                        for (size_t row = 0; row < num_of_rows_in_tile; row++) {
                            hcorepp::memory::Memcpy<T>(&data_array[array_off], &apSubMatrixData[temp_offset],
                                                       num_of_cols_in_tile, aContext.GetMainContext(),
                                                       hcorepp::memory::MemoryTransfer::DEVICE_TO_DEVICE);
                            temp_offset += mGlobalMatrixCols;
                            array_off += num_of_cols_in_tile;
                        }
                    }

                    if (aRank > 0) {
                        mTiles.push_back(
                                new CompressedTile<T>(num_of_rows_in_tile, num_of_cols_in_tile, data_array, leading_dim,
                                                      aRank,
                                                      layout, aContext.GetMainContext()));
                    } else {
                        mTiles.push_back(
                                new DenseTile<T>(num_of_rows_in_tile, num_of_cols_in_tile, data_array, leading_dim,
                                                 layout,
                                                 aContext.GetMainContext()));
                    }

                    data_offset += leading_dim;
                    hcorepp::memory::DestroyArray(data_array, aContext.GetMainContext());
                }
            }
        }

        mMemory = 0;
        if (aRank > 0) {
            for (size_t i = 0; i < mNumOfTilesinCols; i++) {
                for (size_t j = 0; j < mNumOfTilesinRows; j++) {
                    auto tile_cols = std::min(mTileCols, mSubMatrixCols - i * mTileCols);
                    auto tile_rows = std::min(mTileRows, mSubMatrixRows - j * mTileRows);
                    auto tile_idx = 0;
                    if (mStorageLayout == common::StorageLayout::HicmaCM) {
                        tile_idx = i * mNumOfTilesinRows + j;
                    } else if (mStorageLayout == common::StorageLayout::HicmaRM) {
                        tile_idx = j * mNumOfTilesinCols + i;
                    }
                    mMemory += ((tile_rows + tile_cols) *
                                this->mTiles[tile_idx]->GetTileRank() * sizeof(T));
                }
            }
        } else {
            mMemory = mSubMatrixRows * mSubMatrixCols * sizeof(T);
        }
    }

    template<typename T>
    SubMatrix<T>::SubMatrix(T *apSubMatrixData, size_t aTileNumOfRows, size_t aTileNumOfCols,
                            size_t aGlobalMatrixRows, size_t aGlobalMatrixCols, size_t aSubMatrixNumOfRows,
                            size_t aSubMatrixNumOfCols, size_t aTilesGlobalStIdxInRows,
                            size_t aTilesGlobalStIdxInCols, common::StorageLayout aStorageLayout, size_t aOwnerId,
                            runtime::HicmaContext &aContext, const CompressionParameters &aParams) {
        mStorageLayout = aStorageLayout;
        mSubMatrixRows = aSubMatrixNumOfRows;
        mSubMatrixCols = aSubMatrixNumOfCols;
        mTileRows = aTileNumOfRows;
        mTileCols = aTileNumOfCols;
        mGlobalMatrixRows = aGlobalMatrixRows;
        mGlobalMatrixCols = aGlobalMatrixCols;
        // with respect to the tiles decomposed within the global matrix
        mTilesGlobalStIdxInRows = aTilesGlobalStIdxInRows;
        mTilesGlobalStIdxInCols = aTilesGlobalStIdxInCols;

        if (aParams.GetFixedRank() < 0 || !IsValid()) {
            throw std::invalid_argument("Matrix::Initialize Invalid Matrix Initialization");
        }


        if (mTileCols == 0 || mTileRows == 0) {
            throw std::runtime_error("SubMatrix::SubMatrix, Division by zero.\n");
        }
        mNumOfTilesinRows = (mSubMatrixRows + mTileRows - 1) / mTileRows;
        mNumOfTilesinCols = (mSubMatrixCols + mTileCols - 1) / mTileCols;


        mSubMatrixOwnerId = aOwnerId;

        size_t data_offset = 0;

        auto slow_dim_total_num_of_elements = 0;
        auto fast_dim_total_num_of_elements = 0;
        auto slow_dim_tile_num_of_elements = 0;
        auto fast_dim_tile_num_of_elements = 0;
        auto num_of_rows = mTileRows;
        auto num_of_cols = mTileCols;
        auto layout = blas::Layout::RowMajor;
        auto leading_dim = num_of_rows;
        size_t initial_global_offset = 0;

        if (mStorageLayout == common::StorageLayout::HicmaCM) {
            slow_dim_total_num_of_elements = mSubMatrixCols;
            fast_dim_total_num_of_elements = mSubMatrixRows;
            slow_dim_tile_num_of_elements = mTileCols;
            fast_dim_tile_num_of_elements = mTileRows;
            layout = blas::Layout::ColMajor;
            initial_global_offset =
                    mTilesGlobalStIdxInCols * mTileCols * mGlobalMatrixRows + mTilesGlobalStIdxInRows * mTileRows;

        } else if (mStorageLayout == common::StorageLayout::HicmaRM) {
            slow_dim_total_num_of_elements = mSubMatrixRows;
            fast_dim_total_num_of_elements = mSubMatrixCols;
            slow_dim_tile_num_of_elements = mTileRows;
            fast_dim_tile_num_of_elements = mTileCols;
            layout = blas::Layout::RowMajor;
            initial_global_offset =
                    mTilesGlobalStIdxInRows * mTileRows * mGlobalMatrixCols + mTilesGlobalStIdxInCols * mTileCols;
        }

        size_t global_offset = 0;
        for (size_t i = 0; i < slow_dim_total_num_of_elements; i += slow_dim_tile_num_of_elements) {
            global_offset = (i * fast_dim_total_num_of_elements) + initial_global_offset;
            data_offset = global_offset;

            for (size_t j = 0; j < fast_dim_total_num_of_elements; j += fast_dim_tile_num_of_elements) {
                if (mStorageLayout == common::StorageLayout::HicmaCM) {
                    num_of_cols = std::min(mTileCols, slow_dim_total_num_of_elements - i);
                    num_of_rows = std::min(mTileRows, fast_dim_total_num_of_elements - j);
                    leading_dim = num_of_rows;
                } else if (mStorageLayout == common::StorageLayout::HicmaRM) {
                    num_of_cols = std::min(mTileCols, fast_dim_total_num_of_elements - j);
                    num_of_rows = std::min(mTileRows, slow_dim_total_num_of_elements - i);
                    leading_dim = num_of_cols;
                }

                if (apSubMatrixData == nullptr) {
                    mTiles.push_back(
                            new CompressedTile<T>(num_of_rows, num_of_cols, nullptr, leading_dim, aParams, layout,
                                                  aContext.GetMainContext()));

                    continue;
                }


                if (apSubMatrixData != nullptr) {
                    auto data_array = hcorepp::memory::AllocateArray<T>(num_of_rows * num_of_cols,
                                                                        aContext.GetMainContext());
                    auto array_off = 0;
                    auto temp_offset = data_offset;

                    if (mStorageLayout == common::StorageLayout::HicmaCM) {
                        for (size_t col = 0; col < num_of_cols; col++) {
                            hcorepp::memory::Memcpy<T>(&data_array[array_off], &apSubMatrixData[temp_offset],
                                                       num_of_rows, aContext.GetMainContext(),
                                                       hcorepp::memory::MemoryTransfer::DEVICE_TO_DEVICE);
                            temp_offset += mGlobalMatrixRows;
                            array_off += num_of_rows;
                        }
                    } else if (mStorageLayout == common::StorageLayout::HicmaRM) {
                        for (size_t row = 0; row < num_of_rows; row++) {
                            hcorepp::memory::Memcpy<T>(&data_array[array_off], &apSubMatrixData[temp_offset],
                                                       num_of_cols, aContext.GetMainContext(),
                                                       hcorepp::memory::MemoryTransfer::DEVICE_TO_DEVICE);
                            temp_offset += mGlobalMatrixCols;
                            array_off += num_of_cols;
                        }
                    }

                    mTiles.push_back(
                            new CompressedTile<T>(num_of_rows, num_of_cols, data_array, leading_dim, aParams,
                                                  layout, aContext.GetMainContext()));

                    data_offset += leading_dim;
                    hcorepp::memory::DestroyArray(data_array, aContext.GetMainContext());
                }
            }


        }

        mMemory = 0;
        if (aParams.GetFixedRank() > 0) {
            for (size_t i = 0; i < mNumOfTilesinCols; i++) {
                for (size_t j = 0; j < mNumOfTilesinRows; j++) {
                    auto tile_cols = std::min(mTileCols, mSubMatrixCols - i * mTileCols);
                    auto tile_rows = std::min(mTileRows, mSubMatrixRows - j * mTileRows);
                    mMemory += ((tile_rows + tile_cols) *
                                this->mTiles[i][j].GetTileRank() * sizeof(T));
                }
            }
        } else {
            mMemory = mSubMatrixRows * mSubMatrixCols * sizeof(T);
        }

    }

    template<typename T>
    SubMatrix<T>::~SubMatrix<T>() {
        for (auto tile: mTiles) {
            delete tile;
        }
        mTiles.clear();
    }

    template<typename T>
    size_t SubMatrix<T>::GetNumberofTiles() {
        return mTiles.size();
    }

    template<typename T>
    std::vector<Tile<T> *> &
    SubMatrix<T>::GetTiles() {
        return mTiles;
    }

    template<typename T>
    bool SubMatrix<T>::ContainsTile(size_t aTileIdxInRows, size_t aTileIdxInCols) {
        size_t sub_matrix_tile_end_idx_row = mTilesGlobalStIdxInRows + mNumOfTilesinRows;
        size_t sub_matrix_tile_end_idx_col = mTilesGlobalStIdxInCols + mNumOfTilesinCols;

        bool row_check = (aTileIdxInRows < sub_matrix_tile_end_idx_row && aTileIdxInRows >= mTilesGlobalStIdxInRows);
        bool column_check = (aTileIdxInCols < sub_matrix_tile_end_idx_col && aTileIdxInCols >= mTilesGlobalStIdxInCols);
        return (row_check && column_check);
    }


    template<typename T>
    Tile<T> *SubMatrix<T>::GetTilePointer(size_t aTileIdxInRows, size_t aTileIdxInCols) {

        auto requested_tile_idx = 0;

        size_t relative_tile_row_idx = aTileIdxInRows - mTilesGlobalStIdxInRows;
        size_t relative_tile_col_idx = aTileIdxInCols - mTilesGlobalStIdxInCols;

        if (mStorageLayout == common::StorageLayout::HicmaCM) {
            requested_tile_idx =
                    relative_tile_col_idx * mNumOfTilesinRows + relative_tile_row_idx;
        } else if (mStorageLayout == common::StorageLayout::HicmaRM) {
            requested_tile_idx =
                    relative_tile_row_idx * mNumOfTilesinCols + relative_tile_col_idx;
        }

        size_t index = requested_tile_idx;

        return mTiles[index];
    }

    template<typename T>
    int SubMatrix<T>::GetSubMatrixOwnerId() {
        return mSubMatrixOwnerId;
    }

    template<typename T>
    bool SubMatrix<T>::IsValid() {

        if (mSubMatrixRows <= 0 || mSubMatrixCols < 0) {
            return false;
        }
        if ((mGlobalMatrixRows < mSubMatrixRows) || (mGlobalMatrixCols < mSubMatrixCols)) {
            return false;
        }
        if ((mTilesGlobalStIdxInRows > 0 && mTilesGlobalStIdxInRows >= mGlobalMatrixRows) ||
            (mTilesGlobalStIdxInCols > 0 && mTilesGlobalStIdxInCols >= mGlobalMatrixCols)) {
            return false;
        }
        if (mTilesGlobalStIdxInRows + mSubMatrixRows > mGlobalMatrixRows ||
            mTilesGlobalStIdxInCols + mSubMatrixCols > mGlobalMatrixCols) {
            return false;
        }
        return true;
    }

    template<typename T>
    size_t SubMatrix<T>::GetNumOfTilesinRows() {
        return mNumOfTilesinRows;
    }

    template<typename T>
    size_t SubMatrix<T>::GetNumOfTilesinCols() {
        return mNumOfTilesinCols;
    }

    template<typename T>
    size_t SubMatrix<T>::GetTileRows() {
        return mTileRows;
    }

    template<typename T>
    size_t SubMatrix<T>::GetTileCols() {
        return mTileCols;
    }

    template<typename T>
    size_t SubMatrix<T>::GetTilesGlobalStIdxInRows() {
        return mTilesGlobalStIdxInRows;
    }

    template<typename T>
    size_t SubMatrix<T>::GetTilesGlobalStIdxInCols() {
        return mTilesGlobalStIdxInCols;
    }

    template<typename T>
    size_t SubMatrix<T>::GetMemoryFootprint() {
        return mMemory;
    }

    HICMAPP_INSTANTIATE_CLASS(SubMatrix)
}