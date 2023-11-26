#include <hicmapp/utils/MatrixHelpers.hpp>
#include <hcorepp/api/HCore.hpp>
#include <hcorepp/kernels/memory.hpp>


namespace hicmapp::utils {
    template<typename T>
    void MatrixHelpers<T>::MatrixToArray(primitives::Matrix<T> &aMatrix, T *&aArray) {

        size_t array_offset = 0;
        for (auto sub_matrix: aMatrix.GetSubMatrices()) {
            auto &tiles = sub_matrix->GetTiles();
            for (size_t row = 0; row < sub_matrix->GetNumOfTilesinRows(); row++) {
                for (size_t col = 0; col < sub_matrix->GetNumOfTilesinCols(); col++) {
                    auto index = col * sub_matrix->GetNumOfTilesinRows() + row;
                    auto rows = row * sub_matrix->GetTileRows() +
                                sub_matrix->GetTilesGlobalStIdxInRows() * aMatrix.GetNumOfRowsInTile();
                    auto cols = col * sub_matrix->GetTileCols() +
                                sub_matrix->GetTilesGlobalStIdxInCols() * aMatrix.GetNumOfColsInTile();
                    auto &tile = tiles[index];
                    auto tile_rows = tile->GetNumOfRows();
                    auto tile_cols = tile->GetNumOfCols();
                    auto tile_data = tile->GetTileSubMatrix(0);
                    for (size_t i = 0; i < tile_cols; i++) {
                        for (size_t j = 0; j < tile_rows; j++) {
                            auto index_in_tile = i * tile_rows + j;
                            auto full_array_index = rows + j + ((cols + i) * aMatrix.GetGlobalNumOfRowsInMatrix());
                            aArray[full_array_index] = tile_data[index_in_tile];
                        }
                    }
                }
            }
        }
    }

    template<typename T>
    void MatrixHelpers<T>::PrintArray(T *&aArray, size_t aRows, size_t aCols, hicmapp::common::StorageLayout aLayout) {
        if (aLayout == common::StorageLayout::HicmaRM) {
            for (size_t row = 0; row < aRows; row++) {
                for (size_t col = 0; col < aCols; col++) {
                    auto index = row * aCols + col;
                    std::cout << "data[" << row << "][" << col << "]= " << aArray[index] << "\t";
                }
                std::cout << " \n";
            }
        } else if (aLayout == common::StorageLayout::HicmaCM) {
            for (size_t row = 0; row < aRows; row++) {
                for (size_t col = 0; col < aCols; col++) {
                    auto index = col * aRows + row;
                    std::cout << "data[" << row << "][" << col << "]= " << aArray[index] << "\t";
                }
                std::cout << " \n";
            }
        }
    }

    HICMAPP_INSTANTIATE_CLASS(MatrixHelpers)

}