#include <stdexcept>
#include <hicmapp/primitives/decomposer/concrete/slowest_dimension_decomposer.hpp>
#include <iostream>

namespace hicmapp::primitives {

    SlowestDimDecomposer::SlowestDimDecomposer(size_t aNumOfSubMatrices,
                                               common::StorageLayout aStorageLayout) : mStorageLayout(
            aStorageLayout), mNumOfSubMatrices(aNumOfSubMatrices) {
    }


    std::vector<MatrixSpecifications>
    SlowestDimDecomposer::Decompose(size_t aGlobalMatrixTilesInRows, size_t aGlobalMatrixTilesInCols,
                                    bool aDiagonalMatrix) {

/** This assumes a one-to-one process to sub-matrix mapping. Should be revised to support more general mappings */
        size_t sub_matrix_num_of_tiles_in_cols = aGlobalMatrixTilesInCols;
        size_t rem_sub_matrix_num_of_tiles_in_cols = 0;
        size_t sub_matrix_num_of_tiles_in_rows = aGlobalMatrixTilesInRows;
        size_t rem_sub_matrix_num_of_tiles_in_rows = 0;
        auto actual_num_of_submatrices = mNumOfSubMatrices;

        if (mNumOfSubMatrices == 0) {
            throw std::runtime_error("SlowestDecomposer::Decompose, Division by zero.\n");
        }

        if (mStorageLayout == common::StorageLayout::HicmaCM) {
            if (aDiagonalMatrix) {
                sub_matrix_num_of_tiles_in_cols = 1;//aGlobalMatrixTilesInCols / mNumOfSubMatrices;
                rem_sub_matrix_num_of_tiles_in_cols = 0;//aGlobalMatrixTilesInCols % mNumOfSubMatrices;
                sub_matrix_num_of_tiles_in_rows = aGlobalMatrixTilesInRows / mNumOfSubMatrices;
                rem_sub_matrix_num_of_tiles_in_rows = aGlobalMatrixTilesInRows % mNumOfSubMatrices;

            } else {
                sub_matrix_num_of_tiles_in_cols = aGlobalMatrixTilesInCols / mNumOfSubMatrices;
                rem_sub_matrix_num_of_tiles_in_cols = aGlobalMatrixTilesInCols % mNumOfSubMatrices;
                if (sub_matrix_num_of_tiles_in_cols == 0) {
                    throw std::runtime_error(
                            "SlowestDecomposer::Decompose, Number of tiles per column < Number of submatrices.\n");
                }
                if (sub_matrix_num_of_tiles_in_cols == 0 && rem_sub_matrix_num_of_tiles_in_cols > 0) {
                    actual_num_of_submatrices = 1;
                } else {
                    actual_num_of_submatrices = mNumOfSubMatrices;
                }
            }
        } else if (mStorageLayout == common::StorageLayout::HicmaRM) {
            sub_matrix_num_of_tiles_in_rows = aGlobalMatrixTilesInRows / mNumOfSubMatrices;
            rem_sub_matrix_num_of_tiles_in_rows = aGlobalMatrixTilesInRows % mNumOfSubMatrices;
            if (sub_matrix_num_of_tiles_in_rows == 0) {
                throw std::runtime_error(
                        "SlowestDecomposer::Decompose, Number of tiles per row < Number of submatrices.\n");
            }
            if (sub_matrix_num_of_tiles_in_rows == 0 && rem_sub_matrix_num_of_tiles_in_rows > 0) {
                actual_num_of_submatrices = 1;
            } else {
                actual_num_of_submatrices = mNumOfSubMatrices;
            }
        }

        std::vector<MatrixSpecifications> specs(actual_num_of_submatrices);
        int owner_id = 0;
        for (size_t i = 0; i < actual_num_of_submatrices; i++) {

            specs[i].SetNumOfTilesInCol(sub_matrix_num_of_tiles_in_cols);
            specs[i].SetNumOfTilesInRow(sub_matrix_num_of_tiles_in_rows);
            if (i == (actual_num_of_submatrices - 1)) {
                specs[i].SetNumOfTilesInCol(rem_sub_matrix_num_of_tiles_in_cols + sub_matrix_num_of_tiles_in_cols);
                specs[i].SetNumOfTilesInRow(rem_sub_matrix_num_of_tiles_in_rows + sub_matrix_num_of_tiles_in_rows);
            }

            //starting index of the sub-matrix tiles in rows and columns.
            if (mStorageLayout == common::StorageLayout::HicmaCM) {
                specs[i].SetStartingIndexInCols(i * sub_matrix_num_of_tiles_in_cols);
                specs[i].SetStartingIndexInRows(0);
                if(aDiagonalMatrix) {
                    specs[i].SetStartingIndexInCols(0);
                    specs[i].SetStartingIndexInRows(i * sub_matrix_num_of_tiles_in_rows);
                }
            } else if (mStorageLayout == common::StorageLayout::HicmaRM) {
                specs[i].SetStartingIndexInCols(0);
                specs[i].SetStartingIndexInRows(i * sub_matrix_num_of_tiles_in_rows);
            }


            specs[i].SetOwnerId(owner_id);
            owner_id++;
        }

        return specs;
    }


}
