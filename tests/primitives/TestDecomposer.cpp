#include <iostream>

#include <catch2/catch_all.hpp>

#include <hicmapp/primitives/decomposer/matrix_decomposer.hpp>
#include <hicmapp/primitives/decomposer/concrete/slowest_dimension_decomposer.hpp>
#include <hicmapp/primitives/decomposer/concrete/two_dimension_cyclic_decomposer.hpp>
using namespace hicmapp::common;
using namespace hicmapp::primitives;

template<typename T>
void TEST_MATRIX_DECOMPOSER() {
    SECTION("Slowest Dimension Decomposer, CM, 1 sub-matrix") {
        /**
         * __________________________
         * |tile_0 | tile_2 | tile_4|
         * |tile_1 | tile_3 | tile_5|
         * --------------------------
         */
        size_t number_of_sub_matrices = 1;
        StorageLayout storage_layout = StorageLayout::HicmaCM;
        size_t global_matrix_tiles_in_rows = 2;
        size_t global_matrix_tiles_in_cols = 3;
        hicmapp::runtime::HicmaContext context;

        SlowestDimDecomposer slowest_dim_decomposer(number_of_sub_matrices, storage_layout);
        std::vector<MatrixSpecifications> matrix_specs = slowest_dim_decomposer.Decompose(global_matrix_tiles_in_rows,
                                                                                          global_matrix_tiles_in_cols);

        REQUIRE(matrix_specs.size() == number_of_sub_matrices);

        for (int i = 0; i < number_of_sub_matrices; i++) {
            REQUIRE(matrix_specs[i].GetNumOfTilesInCol() == 3);
            REQUIRE(matrix_specs[i].GetNumOfTilesInRow() == 2);
            REQUIRE(matrix_specs[i].GetStartingIndexInCols() == 0);
            REQUIRE(matrix_specs[i].GetStartingIndexInRows() == 0);
        }

    }

    SECTION("Slowest Dimension Decomposer, CM, 2 sub-matrices, non-divisible") {
        /**
         * ____________________________
         * | tile_0 || tile_2 | tile_4 |
         * | tile_1 || tile_3 | tile_5 |
         * ----------------------------
         */
        size_t number_of_sub_matrices = 2;
        StorageLayout storage_layout = StorageLayout::HicmaCM;
        size_t global_matrix_tiles_in_rows = 2;
        size_t global_matrix_tiles_in_cols = 3;
        hicmapp::runtime::HicmaContext context;

        SlowestDimDecomposer slowest_dim_decomposer(number_of_sub_matrices, storage_layout);
        std::vector<MatrixSpecifications> matrix_specs = slowest_dim_decomposer.Decompose(global_matrix_tiles_in_rows,
                                                                                          global_matrix_tiles_in_cols);

        REQUIRE(matrix_specs.size() == number_of_sub_matrices);

        REQUIRE(matrix_specs[0].GetNumOfTilesInCol() == 1);
        REQUIRE(matrix_specs[0].GetNumOfTilesInRow() == 2);
        REQUIRE(matrix_specs[0].GetStartingIndexInCols() == 0);
        REQUIRE(matrix_specs[0].GetStartingIndexInRows() == 0);

        REQUIRE(matrix_specs[1].GetNumOfTilesInCol() == 2);
        REQUIRE(matrix_specs[1].GetNumOfTilesInRow() == 2);
        REQUIRE(matrix_specs[1].GetStartingIndexInCols() == 1);
        REQUIRE(matrix_specs[1].GetStartingIndexInRows() == 0);

    }

    SECTION("Slowest Dimension Decomposer, CM, 2 sub-matrices, sub-matrices > number of tiles") {
        /**
         * ____________________________
         * | tile_0 | tile_2 | tile_4 |
         * | tile_1 | tile_3 | tile_5 |
         * ----------------------------
         */
        size_t number_of_sub_matrices = 2;
        StorageLayout storage_layout = StorageLayout::HicmaCM;
        size_t global_matrix_tiles_in_rows = 2;
        size_t global_matrix_tiles_in_cols = 3;
        hicmapp::runtime::HicmaContext context;

        SlowestDimDecomposer slowest_dim_decomposer(number_of_sub_matrices, storage_layout);
        std::vector<MatrixSpecifications> matrix_specs = slowest_dim_decomposer.Decompose(global_matrix_tiles_in_rows,
                                                                                          global_matrix_tiles_in_cols);

        REQUIRE(matrix_specs.size() == number_of_sub_matrices);

        //TODO: test for number of sub mtarices > number of tiles per row/col error handling
    }

    SECTION("Slowest Dimension Decomposer, CM, 3 sub-matrices") {
        /**
         * ____________________________
         * | tile_0 || tile_2 || tile_4 |
         * | tile_1 || tile_3 || tile_5 |
         * ----------------------------
         */
        size_t number_of_sub_matrices = 3;
        StorageLayout storage_layout = StorageLayout::HicmaCM;
        size_t global_matrix_tiles_in_rows = 2;
        size_t global_matrix_tiles_in_cols = 3;
        hicmapp::runtime::HicmaContext context;

        SlowestDimDecomposer slowest_dim_decomposer(number_of_sub_matrices, storage_layout);
        std::vector<MatrixSpecifications> matrix_specs = slowest_dim_decomposer.Decompose(global_matrix_tiles_in_rows,
                                                                                          global_matrix_tiles_in_cols);

        REQUIRE(matrix_specs.size() == number_of_sub_matrices);

        for (int i = 0; i < number_of_sub_matrices; i++) {
            REQUIRE(matrix_specs[i].GetNumOfTilesInCol() == 1);
            REQUIRE(matrix_specs[i].GetNumOfTilesInRow() == 2);
            REQUIRE(matrix_specs[i].GetStartingIndexInCols() == i);
            REQUIRE(matrix_specs[i].GetStartingIndexInRows() == 0);
        }
    }

    SECTION("Slowest Dimension Decomposer, RM, 1 sub-matrix") {
        /**
         * __________________________
         * |tile_0 | tile_1 | tile_2|
         * |tile_3 | tile_4 | tile_5|
         * --------------------------
         */
        size_t number_of_sub_matrices = 1;
        StorageLayout storage_layout = StorageLayout::HicmaRM;
        size_t global_matrix_tiles_in_rows = 2;
        size_t global_matrix_tiles_in_cols = 3;
        hicmapp::runtime::HicmaContext context;

        SlowestDimDecomposer slowest_dim_decomposer(number_of_sub_matrices, storage_layout);
        std::vector<MatrixSpecifications> matrix_specs = slowest_dim_decomposer.Decompose(global_matrix_tiles_in_rows,
                                                                                          global_matrix_tiles_in_cols);

        REQUIRE(matrix_specs.size() == number_of_sub_matrices);

        for (int i = 0; i < number_of_sub_matrices; i++) {
            REQUIRE(matrix_specs[i].GetNumOfTilesInCol() == 3);
            REQUIRE(matrix_specs[i].GetNumOfTilesInRow() == 2);
            REQUIRE(matrix_specs[i].GetStartingIndexInCols() == 0);
            REQUIRE(matrix_specs[i].GetStartingIndexInRows() == 0);
        }

    }

    SECTION("Slowest Dimension Decomposer, RM, 2 sub-matrices") {
        /**
         * __________________________
         * |tile_0 | tile_1 | tile_2|
         * ==========================
         * |tile_3 | tile_4 | tile_5|
         * --------------------------
         */
        size_t number_of_sub_matrices = 2;
        StorageLayout storage_layout = StorageLayout::HicmaRM;
        size_t global_matrix_tiles_in_rows = 2;
        size_t global_matrix_tiles_in_cols = 3;
        hicmapp::runtime::HicmaContext context;

        SlowestDimDecomposer slowest_dim_decomposer(number_of_sub_matrices, storage_layout);
        std::vector<MatrixSpecifications> matrix_specs = slowest_dim_decomposer.Decompose(global_matrix_tiles_in_rows,
                                                                                          global_matrix_tiles_in_cols);

        REQUIRE(matrix_specs.size() == number_of_sub_matrices);

        for (int i = 0; i < number_of_sub_matrices; i++) {
            REQUIRE(matrix_specs[i].GetNumOfTilesInCol() == 3);
            REQUIRE(matrix_specs[i].GetNumOfTilesInRow() == 1);
            REQUIRE(matrix_specs[i].GetStartingIndexInCols() == 0);
            REQUIRE(matrix_specs[i].GetStartingIndexInRows() == i);
        }

    }

    SECTION("Slowest Dimension Decomposer, RM, 2 sub-matrices, sub-matrices > number of tiles") {
        /**
         * __________________________
         * |tile_0 | tile_1 | tile_2|
         * ==========================
         * |tile_3 | tile_4 | tile_5|
         * --------------------------
         */
        size_t number_of_sub_matrices = 3;
        StorageLayout storage_layout = StorageLayout::HicmaRM;
        size_t global_matrix_tiles_in_rows = 2;
        size_t global_matrix_tiles_in_cols = 3;
        hicmapp::runtime::HicmaContext context;

        SlowestDimDecomposer slowest_dim_decomposer(number_of_sub_matrices, storage_layout);
        REQUIRE_THROWS(slowest_dim_decomposer.Decompose(global_matrix_tiles_in_rows,
                                                        global_matrix_tiles_in_cols));

    }

    SECTION("Slowest Dimension Decomposer, RM, 2 sub-matrices, non-divisible") {
        /**
         * __________________________
         * |tile_0 | tile_1 | tile_2|
         * ==========================
         * |tile_3 | tile_4 | tile_5|
         * ==========================
         * |tile_3 | tile_4 | tile_5|
         * --------------------------
         * |tile_3 | tile_4 | tile_5|
         */

        size_t number_of_sub_matrices = 3;
        StorageLayout storage_layout = StorageLayout::HicmaRM;
        size_t global_matrix_tiles_in_rows = 4;
        size_t global_matrix_tiles_in_cols = 3;
        hicmapp::runtime::HicmaContext context;

        SlowestDimDecomposer slowest_dim_decomposer(number_of_sub_matrices, storage_layout);
        std::vector<MatrixSpecifications> matrix_specs = slowest_dim_decomposer.Decompose(global_matrix_tiles_in_rows,
                                                                                          global_matrix_tiles_in_cols);

        REQUIRE(matrix_specs.size() == number_of_sub_matrices);
        int i = 0;
        for (; i < number_of_sub_matrices - 1; i++) {
            REQUIRE(matrix_specs[i].GetNumOfTilesInCol() == 3);
            REQUIRE(matrix_specs[i].GetNumOfTilesInRow() == 1);
            REQUIRE(matrix_specs[i].GetStartingIndexInCols() == 0);
            REQUIRE(matrix_specs[i].GetStartingIndexInRows() == i);
        }
        REQUIRE(matrix_specs[i].GetNumOfTilesInCol() == 3);
        REQUIRE(matrix_specs[i].GetNumOfTilesInRow() == 2);
        REQUIRE(matrix_specs[i].GetStartingIndexInCols() == 0);
        REQUIRE(matrix_specs[i].GetStartingIndexInRows() == i);

    }

    SECTION("Slowest Dimension Decomposer, CM, 2 sub-matrices, non-divisible") {
        /**
         * ____________________________
         * | tile_0 || tile_2 | tile_4 |
         * | tile_1 || tile_3 | tile_5 |
         * ----------------------------
         */
        size_t number_of_sub_matrices = 2;
        StorageLayout storage_layout = StorageLayout::HicmaCM;
        size_t global_matrix_tiles_in_rows = 2;
        size_t global_matrix_tiles_in_cols = 3;
        hicmapp::runtime::HicmaContext context;

        SlowestDimDecomposer slowest_dim_decomposer(number_of_sub_matrices, storage_layout);
        std::vector<MatrixSpecifications> matrix_specs = slowest_dim_decomposer.Decompose(global_matrix_tiles_in_rows,
                                                                                          global_matrix_tiles_in_cols);

        REQUIRE(matrix_specs.size() == number_of_sub_matrices);

        REQUIRE(matrix_specs[0].GetNumOfTilesInCol() == 1);
        REQUIRE(matrix_specs[0].GetNumOfTilesInRow() == 2);
        REQUIRE(matrix_specs[0].GetStartingIndexInCols() == 0);
        REQUIRE(matrix_specs[0].GetStartingIndexInRows() == 0);

        REQUIRE(matrix_specs[1].GetNumOfTilesInCol() == 2);
        REQUIRE(matrix_specs[1].GetNumOfTilesInRow() == 2);

        REQUIRE(matrix_specs[1].GetStartingIndexInCols() == 1);
        REQUIRE(matrix_specs[1].GetStartingIndexInRows() == 0);

    }

    SECTION("Slowest Dimension Decomposer, CM, 2 sub-matrices, sub-matrices > number of tiles") {
        /**
         * ____________________________
         * | tile_0 | tile_2 | tile_4 |
         * | tile_1 | tile_3 | tile_5 |
         * ----------------------------
         */
        size_t number_of_sub_matrices = 2;
        StorageLayout storage_layout = StorageLayout::HicmaCM;
        size_t global_matrix_tiles_in_rows = 2;
        size_t global_matrix_tiles_in_cols = 3;
        hicmapp::runtime::HicmaContext context;

        SlowestDimDecomposer slowest_dim_decomposer(number_of_sub_matrices, storage_layout);
        std::vector<MatrixSpecifications> matrix_specs = slowest_dim_decomposer.Decompose(global_matrix_tiles_in_rows,
                                                                                          global_matrix_tiles_in_cols);

        REQUIRE(matrix_specs.size() == number_of_sub_matrices);

        //TODO: test for number of sub mtarices > number of tiles per row/col error handling
    }

    SECTION("Slowest Dimension Decomposer, CM, 3 sub-matrices") {
        /**
         * ____________________________
         * | tile_0 || tile_2 || tile_4 |
         * | tile_1 || tile_3 || tile_5 |
         * ----------------------------
         */
        size_t number_of_sub_matrices = 3;
        StorageLayout storage_layout = StorageLayout::HicmaCM;
        size_t global_matrix_tiles_in_rows = 2;
        size_t global_matrix_tiles_in_cols = 3;
        hicmapp::runtime::HicmaContext context;

        SlowestDimDecomposer slowest_dim_decomposer(number_of_sub_matrices, storage_layout);
        std::vector<MatrixSpecifications> matrix_specs = slowest_dim_decomposer.Decompose(global_matrix_tiles_in_rows,
                                                                                          global_matrix_tiles_in_cols);

        REQUIRE(matrix_specs.size() == number_of_sub_matrices);

        for (int i = 0; i < number_of_sub_matrices; i++) {
            REQUIRE(matrix_specs[i].GetNumOfTilesInCol() == 1);
            REQUIRE(matrix_specs[i].GetNumOfTilesInRow() == 2);
            REQUIRE(matrix_specs[i].GetStartingIndexInCols() == i);
            REQUIRE(matrix_specs[i].GetStartingIndexInRows() == 0);
        }
    }

    SECTION("Slowest Dimension Decomposer, RM, 1 sub-matrix") {
        /**
         * __________________________
         * |tile_0 | tile_1 | tile_2|
         * |tile_3 | tile_4 | tile_5|
         * --------------------------
         */
        size_t number_of_sub_matrices = 1;
        StorageLayout storage_layout = StorageLayout::HicmaRM;
        size_t global_matrix_tiles_in_rows = 2;
        size_t global_matrix_tiles_in_cols = 3;
        hicmapp::runtime::HicmaContext context;

        SlowestDimDecomposer slowest_dim_decomposer(number_of_sub_matrices, storage_layout);
        std::vector<MatrixSpecifications> matrix_specs = slowest_dim_decomposer.Decompose(global_matrix_tiles_in_rows,
                                                                                          global_matrix_tiles_in_cols);

        REQUIRE(matrix_specs.size() == number_of_sub_matrices);

        for (int i = 0; i < number_of_sub_matrices; i++) {
            REQUIRE(matrix_specs[i].GetNumOfTilesInCol() == 3);
            REQUIRE(matrix_specs[i].GetNumOfTilesInRow() == 2);
            REQUIRE(matrix_specs[i].GetStartingIndexInCols() == 0);
            REQUIRE(matrix_specs[i].GetStartingIndexInRows() == 0);
        }

    }

    SECTION("Slowest Dimension Decomposer, RM, 2 sub-matrices") {
        /**
         * __________________________
         * |tile_0 | tile_1 | tile_2|
         * ==========================
         * |tile_3 | tile_4 | tile_5|
         * --------------------------
         */
        size_t number_of_sub_matrices = 2;
        StorageLayout storage_layout = StorageLayout::HicmaRM;
        size_t global_matrix_tiles_in_rows = 2;
        size_t global_matrix_tiles_in_cols = 3;
        hicmapp::runtime::HicmaContext context;

        SlowestDimDecomposer slowest_dim_decomposer(number_of_sub_matrices, storage_layout);
        std::vector<MatrixSpecifications> matrix_specs = slowest_dim_decomposer.Decompose(global_matrix_tiles_in_rows,
                                                                                          global_matrix_tiles_in_cols);

        REQUIRE(matrix_specs.size() == number_of_sub_matrices);

        for (int i = 0; i < number_of_sub_matrices; i++) {
            REQUIRE(matrix_specs[i].GetNumOfTilesInCol() == 3);
            REQUIRE(matrix_specs[i].GetNumOfTilesInRow() == 1);
            REQUIRE(matrix_specs[i].GetStartingIndexInCols() == 0);
            REQUIRE(matrix_specs[i].GetStartingIndexInRows() == i);
        }

    }

    SECTION("Slowest Dimension Decomposer, RM, 2 sub-matrices, sub-matrices > number of tiles") {
        /**
         * __________________________
         * |tile_0 | tile_1 | tile_2|
         * ==========================
         * |tile_3 | tile_4 | tile_5|
         * --------------------------
         */
        size_t number_of_sub_matrices = 3;
        StorageLayout storage_layout = StorageLayout::HicmaRM;
        size_t global_matrix_tiles_in_rows = 2;
        size_t global_matrix_tiles_in_cols = 3;
        hicmapp::runtime::HicmaContext context;

        SlowestDimDecomposer slowest_dim_decomposer(number_of_sub_matrices, storage_layout);
        REQUIRE_THROWS(slowest_dim_decomposer.Decompose(global_matrix_tiles_in_rows, global_matrix_tiles_in_cols));

    }

    SECTION("Slowest Dimension Decomposer, RM, 2 sub-matrices, non-divisible") {
        /**
         * __________________________
         * |tile_0 | tile_1 | tile_2|
         * ==========================
         * |tile_3 | tile_4 | tile_5|
         * ==========================
         * |tile_3 | tile_4 | tile_5|
         * --------------------------
         * |tile_3 | tile_4 | tile_5|
         */

        size_t number_of_sub_matrices = 3;
        StorageLayout storage_layout = StorageLayout::HicmaRM;
        size_t global_matrix_tiles_in_rows = 4;
        size_t global_matrix_tiles_in_cols = 3;
        hicmapp::runtime::HicmaContext context;

        SlowestDimDecomposer slowest_dim_decomposer(number_of_sub_matrices, storage_layout);
        std::vector<MatrixSpecifications> matrix_specs = slowest_dim_decomposer.Decompose(global_matrix_tiles_in_rows,
                                                                                          global_matrix_tiles_in_cols);

        REQUIRE(matrix_specs.size() == number_of_sub_matrices);
        int i = 0;
        for (; i < number_of_sub_matrices - 1; i++) {
            REQUIRE(matrix_specs[i].GetNumOfTilesInCol() == 3);
            REQUIRE(matrix_specs[i].GetNumOfTilesInRow() == 1);
            REQUIRE(matrix_specs[i].GetStartingIndexInCols() == 0);
            REQUIRE(matrix_specs[i].GetStartingIndexInRows() == i);
        }
        REQUIRE(matrix_specs[i].GetNumOfTilesInCol() == 3);
        REQUIRE(matrix_specs[i].GetNumOfTilesInRow() == 2);
        REQUIRE(matrix_specs[i].GetStartingIndexInCols() == 0);
        REQUIRE(matrix_specs[i].GetStartingIndexInRows() == i);

    }

    /** These Test Cases need to be revised. A more general approach regardless of number of processes should be taken here */
//    SECTION("Two Dimension Decomposer, CM, 6 sub-matrices") {
//        /**
//         * __________________________
//         * |tile_0 || tile_2 || tile_4|
//         * |tile_1 || tile_3 || tile_5|
//         * --------------------------
//         */
//        StorageLayout storage_layout = StorageLayout::HicmaCM;
//        hicmapp::runtime::HicmaContext context;
//
//        size_t global_matrix_tiles_in_rows = 2;
//        size_t global_matrix_tiles_in_cols = 3;
//
//
//        int id = 0;
//        int size = 1;
//#ifdef HICMAPP_USE_MPI
//        MPI_Comm_rank(MPI_COMM_WORLD, &id);
//        MPI_Comm_size(MPI_COMM_WORLD, &size);
//#endif
//
//        size_t number_of_row_processes = std::max(size/2, 1);
//        size_t number_of_col_processes = std::max(size/2, 1);
//        TwoDimCyclicDecomposer two_dim_cyclic_decomposer(number_of_row_processes, number_of_col_processes);
//        std::vector<MatrixSpecifications> matrix_specs = two_dim_cyclic_decomposer.Decompose(
//                global_matrix_tiles_in_rows,
//                global_matrix_tiles_in_cols);
//
//        size_t number_of_sub_matrices = size;
//        size_t matrix_of_row_indexes[] = {0, 0, 0};
//        size_t matrix_of_col_indexes[] = {0, 1, 2,};
//
//        if (size == 1) {
//            number_of_sub_matrices = 1;
//            size_t matrix_of_row_indexes_one_proc[] = {0, 0, 0, 1, 1, 1};
//            size_t matrix_of_col_indexes_one_proc[] = {0, 1, 2, 0, 1, 2};
//            for (int i = 0; i < number_of_sub_matrices; i++) {
//                REQUIRE(matrix_specs[i].GetStartingIndexInRows() == matrix_of_row_indexes_one_proc[i]);
//                REQUIRE(matrix_specs[i].GetStartingIndexInCols() == matrix_of_col_indexes_one_proc[i]);
//            }
//        } else {
//            for (int i = 0; i < number_of_sub_matrices; i++) {
//                REQUIRE(matrix_specs[i].GetStartingIndexInRows() == matrix_of_row_indexes[i]);
//                REQUIRE(matrix_specs[i].GetStartingIndexInCols() == matrix_of_col_indexes[i]);
//            }
//        }
//        REQUIRE(number_of_sub_matrices == matrix_specs.size());
//
//    }
//
//    SECTION("Two Dimension Decomposer, RM, 12 sub-matrices, non divisible by col") {
//        /**
//         * __________________________
//         * |tile_0 | tile_1 || tile_4|
//         * |tile_2 | tile_3 || tile_5|
//         * ---------------------------
//         * |tile_6 | tile_7 || tile_10|
//         * |tile_8 | tile_9 || tile_11|
//         * --------------------------
//         */
//        size_t matrix_of_row_indexes[] = {0, 0, 2, 2};
//        size_t matrix_of_col_indexes[] = {0, 2, 0, 2};
//
//        StorageLayout storage_layout = StorageLayout::HicmaCM;
//        size_t number_of_row_processes = 2;
//        size_t number_of_col_processes = 2;
//
//        size_t global_matrix_tiles_in_rows = 4;
//        size_t global_matrix_tiles_in_cols = 3;
//
//        int id = 0;
//        int size = 1;
//#ifdef HICMAPP_USE_MPI
//        MPI_Comm_rank(MPI_COMM_WORLD, &id);
//        MPI_Comm_size(MPI_COMM_WORLD, &size);
//#endif
//        hicmapp::runtime::HicmaContext context;
//
//        TwoDimCyclicDecomposer two_dim_cyclic_decomposer(number_of_row_processes, number_of_col_processes);
//        std::vector<MatrixSpecifications> matrix_specs = two_dim_cyclic_decomposer.Decompose(
//                global_matrix_tiles_in_rows,
//                global_matrix_tiles_in_cols);
//
//        size_t number_of_sub_matrices = 4;
//        if (size == 1) {
//            number_of_sub_matrices = global_matrix_tiles_in_rows * global_matrix_tiles_in_cols;
//            size_t matrix_of_row_indexes_one_proc[] = {0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3};
//            size_t matrix_of_col_indexes_one_proc[] = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2};
//            for (int i = 0; i < number_of_sub_matrices; i++) {
//                REQUIRE(matrix_specs[i].GetStartingIndexInRows() == matrix_of_row_indexes_one_proc[i]);
//                REQUIRE(matrix_specs[i].GetStartingIndexInCols() == matrix_of_col_indexes_one_proc[i]);
//            }
//        } else {
//            for (int i = 0; i < number_of_sub_matrices; i++) {
//                REQUIRE(matrix_specs[i].GetStartingIndexInRows() == matrix_of_row_indexes[i]);
//                REQUIRE(matrix_specs[i].GetStartingIndexInCols() == matrix_of_col_indexes[i]);
//            }
//        }
//        REQUIRE(number_of_sub_matrices == matrix_specs.size());
//
//    }
//
//    SECTION("Two Dimension Decomposer, RM, 12 sub-matrices, non divisible by both col and row") {
//        /**
//         * __________________________
//         * |tile_0 | tile_1 || tile_6|
//         * |tile_2 | tile_3 || tile_7|
//         * |tile_4 | tile_5 || tile_8|
//         * ---------------------------
//         * |tile_9 | tile_10 || tile_11|
//         * --------------------------
//         */
//
//        size_t matrix_of_row_indexes[] = {0, 0, 3, 3};
//        size_t matrix_of_col_indexes[] = {0, 2, 0, 2};
//
//        StorageLayout storage_layout = StorageLayout::HicmaCM;
//        size_t number_of_row_processes = 3;
//        size_t number_of_col_processes = 2;
//
//        size_t global_matrix_tiles_in_rows = 4;
//        size_t global_matrix_tiles_in_cols = 3;
//        hicmapp::runtime::HicmaContext context;
//
//        TwoDimCyclicDecomposer two_dim_cyclic_decomposer(number_of_row_processes, number_of_col_processes);
//        std::vector<MatrixSpecifications> matrix_specs = two_dim_cyclic_decomposer.Decompose(
//                global_matrix_tiles_in_rows,
//                global_matrix_tiles_in_cols);
//
//        int id = 0;
//        int size = 1;
//#ifdef HICMAPP_USE_MPI
//        MPI_Comm_rank(MPI_COMM_WORLD, &id);
//        MPI_Comm_size(MPI_COMM_WORLD, &size);
//#endif
//        size_t number_of_sub_matrices = 4;
//        if (size == 1) {
//            number_of_sub_matrices = global_matrix_tiles_in_rows * global_matrix_tiles_in_cols;
//            size_t matrix_of_row_indexes_one_proc[] = {0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3};
//            size_t matrix_of_col_indexes_one_proc[] = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2};
//            for (int i = 0; i < number_of_sub_matrices; i++) {
//                REQUIRE(matrix_specs[i].GetStartingIndexInRows() == matrix_of_row_indexes_one_proc[i]);
//                REQUIRE(matrix_specs[i].GetStartingIndexInCols() == matrix_of_col_indexes_one_proc[i]);
//            }
//        } else {
//            for (int i = 0; i < number_of_sub_matrices; i++) {
//                REQUIRE(matrix_specs[i].GetStartingIndexInRows() == matrix_of_row_indexes[i]);
//                REQUIRE(matrix_specs[i].GetStartingIndexInCols() == matrix_of_col_indexes[i]);
//            }
//        }
//        REQUIRE(number_of_sub_matrices == matrix_specs.size());
//    }

}

TEMPLATE_TEST_CASE("MatrixDecomposerTest", "[MatrixDecomposer]", float, double) {
    TEST_MATRIX_DECOMPOSER<TestType>();
}
