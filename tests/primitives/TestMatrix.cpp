#include <catch2/catch_all.hpp>
#include <algorithm>

#include <hicmapp/primitives/matrix.hpp>
#include <hicmapp/primitives/decomposer/matrix_decomposer.hpp>
#include <hicmapp/primitives/decomposer/concrete/two_dimension_cyclic_decomposer.hpp>
#include <hcorepp/kernels/memory.hpp>

using namespace hicmapp::common;
using namespace hicmapp::primitives;

template<typename T>
void TEST_MATRIX() {

    SECTION("Test CM, slowestdimdecomposer 2 submatrices") {
        int id = 0, size = 1;
        hicmapp::runtime::HicmaCommunicator communicator;
#ifdef HICMAPP_USE_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &id);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        communicator.SetMPICommunicator(MPI_COMM_WORLD);
#endif
        hicmapp::runtime::HicmaContext context(communicator);
        printf(" SECTION 1 PROCESS # %d\n", id);

        /**
               *      matrix
               * 0    3 || 6    9
               * ------ || -------
               * 1    4 || 7    10
               * ------ || -------
               * 2    5 || 8   11
               */
        T matrix_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        T matrix_data_expected[] = {0, 3, 1, 4, 2, 5, 6, 9, 7, 10, 8, 11};
        size_t matrix_rows = 3;
        size_t matrix_cols = 4;
        size_t tile_rows = 1;
        size_t tile_cols = 2;
        float eps = 1e-6;
        StorageLayout storage_layout = StorageLayout::HicmaCM;
        size_t rank = 0;
        size_t number_of_sub_matrices = size;
        size_t number_of_tile_per_sub_matrix = 3;

        SlowestDimDecomposer matrix_decomposer(number_of_sub_matrices, storage_layout);
        Matrix<T> matrix(matrix_data, matrix_rows, matrix_cols, tile_rows, tile_cols, storage_layout,
                         matrix_decomposer, context,
                         rank);
        /* TEST GET NUMBER OF SUB MATRICES */
        if (id < number_of_sub_matrices) {

            ///for MPI support
//            number_of_sub_matrices = number_of_sub_matrices / size == 0 ? 1 : number_of_sub_matrices / size;

            REQUIRE(matrix.GetNumOfSubMatrices() == number_of_sub_matrices);

            /* TEST GET NUMBER OF TILES IN ROW */
            REQUIRE(matrix.GetNumOfGlobalTilesInRows() == 3);

            /* TEST GET NUMBER OF TILES IN ROW */
            REQUIRE(matrix.GetNumOfGlobalTilesInCols() == 2);

            //TODO: test contains tile when fixed.

            /* TEST GET STORAGE LAYOUT */
            REQUIRE(matrix.GetStorageLayout() == storage_layout);

            /* TEST IS MATRIX VALID */
            REQUIRE(matrix.IsMatrixValid() == true);

            /* TEST NUMBER OF ROWS IN TILE */
            REQUIRE(matrix.GetNumOfRowsInTile() == tile_rows);

            /* TEST NUMBER OF COLS IN TILE */
            REQUIRE(matrix.GetNumOfColsInTile() == tile_cols);

            /* TEST NUMBER OF ROWS IN MATRIX */
            REQUIRE(matrix.GetGlobalNumOfRowsInMatrix() == matrix_rows);

            /* TEST NUMBER OF COLS IN MATRIX */
            REQUIRE(matrix.GetGlobalNumOfColsInMatrix() == matrix_cols);

            /* TEST GET SUBMATRICES */
            auto sub_matrices = matrix.GetSubMatrices();
            size_t offset = 0;
            auto *host_mem = new T[tile_rows * tile_cols];
            for (int i = id; i < number_of_sub_matrices; i++) {
                size_t number_of_tiles_in_sub_matrix = sub_matrices[i]->GetNumberofTiles();
                for (int j = 0; j < number_of_tiles_in_sub_matrix; j++) {
                    hcorepp::memory::Memcpy(host_mem,
                                            sub_matrices[i]->GetTiles()[j]->GetTileSubMatrix(0),
                                            tile_rows * tile_cols, context.GetMainContext(),
                                            hcorepp::memory::MemoryTransfer::DEVICE_TO_HOST);
                    context.SyncMainContext();
                    for (int k = 0; k < tile_cols * tile_rows; k++) {
                        REQUIRE(std::abs(host_mem[k] - matrix_data_expected[k + offset]) <= std::abs(eps));
                    }
                    offset += tile_rows * tile_cols;
                }
            }
            /* TEST GET SUBMATRIX */
            offset = 0;
            for (int i = id; i < number_of_sub_matrices; i += size) {
                auto sub_matrix = &matrix.GetSubMatrix(i);
                size_t number_of_tiles_in_sub_matrix = sub_matrix->GetNumberofTiles();
                for (int j = 0; j < number_of_tiles_in_sub_matrix; j++) {
                    hcorepp::memory::Memcpy(host_mem, sub_matrix->GetTiles()[j]->GetTileSubMatrix(0),
                                            tile_rows * tile_cols, context.GetMainContext(),
                                            hcorepp::memory::MemoryTransfer::DEVICE_TO_HOST);
                    context.SyncMainContext();
                    for (int k = 0; k < tile_cols * tile_rows; k++) {
                        REQUIRE(std::abs(host_mem[k] - matrix_data_expected[k + offset]) <= std::abs(eps));
                    }
                    offset += tile_rows * tile_cols;
                }
            }
            /* TEST GET TILE POINTER */
            offset = id * tile_cols * tile_rows * number_of_tile_per_sub_matrix;
            for (int i = 0; i < number_of_sub_matrices; i++) {
                size_t col_idx = id + i;
                for (int j = 0; j < number_of_tile_per_sub_matrix; j++) {
                    size_t row_idx = j;
                    auto tile = matrix.GetTilePointer(row_idx, col_idx);
                    hcorepp::memory::Memcpy(host_mem, tile->GetTileSubMatrix(0), tile_rows * tile_cols,
                                            context.GetMainContext(),
                                            hcorepp::memory::MemoryTransfer::DEVICE_TO_HOST);
                    context.SyncMainContext();
                    for (int k = 0; k < tile_cols * tile_rows; k++) {
                        REQUIRE(std::abs(host_mem[k] - matrix_data_expected[k + offset]) <= std::abs(eps));
                    }
                    offset += tile_rows * tile_cols;

                }
            }
            delete[] host_mem;
        } else {
            REQUIRE(matrix.GetNumOfSubMatrices() == 0);

        }

    }

    SECTION("Test CM, default decomposer") {
        /**
        * matrix
        * 0    3 | 6     9
        * ------ | -------
        * 1    4 | 7    10
        * ------ | -------
        * 2    5 | 8    11
        */
        int id = 0, size = 1;
        hicmapp::runtime::HicmaCommunicator communicator;
#ifdef HICMAPP_USE_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &id);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        communicator.SetMPICommunicator(MPI_COMM_WORLD);
#endif

        hicmapp::runtime::HicmaContext context(communicator);

        T matrix_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        T matrix_data_expected[] = {0, 3, 1, 4, 2, 5, 6, 9, 7, 10, 8, 11};
        size_t matrix_rows = 3;
        size_t matrix_cols = 4;
        size_t global_number_of_tiles_in_col = 2;
        size_t global_number_of_tiles_in_row = 3;
        size_t tile_rows = 1;
        size_t tile_cols = 2;
        float eps = 1e-6;
        StorageLayout storage_layout = StorageLayout::HicmaCM;
        size_t rank = 0;
        size_t number_of_sub_matrices = 1;
        size_t global_number_of_tiles = 6;
        size_t number_of_tiles_per_sub_matrix = 3; /**Slowest Dim Decomposer in the case where total number of processes <= global_number_of_tiles_in_col number of submatrices = 2*/
        Matrix<T> matrix(matrix_data, matrix_rows, matrix_cols, tile_rows, tile_cols, storage_layout, context,
                         rank);

        if (global_number_of_tiles_in_col >= size || id == size - 1) {

            ///for MPI support
//            number_of_sub_matrices = number_of_sub_matrices / size == 0 ? 1 : number_of_sub_matrices / size;
            /** TEST GET NUMBER OF SUB MATRICES **/
            REQUIRE(matrix.GetNumOfSubMatrices() == number_of_sub_matrices);

            /** TEST GET NUMBER OF TILES IN ROW **/
            REQUIRE(matrix.GetNumOfGlobalTilesInRows() == global_number_of_tiles_in_row);

            /** TEST GET NUMBER OF TILES IN ROW **/
            REQUIRE(matrix.GetNumOfGlobalTilesInCols() == global_number_of_tiles_in_col);

            ///TODO: test contains tile when fixed.

            /** TEST GET STORAGE LAYOUT **/
            REQUIRE(matrix.GetStorageLayout() == storage_layout);

            /** TEST IS MATRIX VALID **/
            REQUIRE(matrix.IsMatrixValid() == true);

            /** TEST NUMBER OF ROWS IN TILE **/
            REQUIRE(matrix.GetNumOfRowsInTile() == tile_rows);

            /** TEST NUMBER OF COLS IN TILE **/
            REQUIRE(matrix.GetNumOfColsInTile() == tile_cols);

            /** TEST NUMBER OF ROWS IN MATRIX **/
            REQUIRE(matrix.GetGlobalNumOfRowsInMatrix() == matrix_rows);

            /** TEST NUMBER OF COLS IN MATRIX **/
            REQUIRE(matrix.GetGlobalNumOfColsInMatrix() == matrix_cols);

            /** TEST GET SUBMATRICES **/
            auto sub_matrices = matrix.GetSubMatrices();
            size_t offset = (id - size + 1) * tile_cols * tile_rows * global_number_of_tiles;
            if (id == 0) {
                offset = 0;
            } else if (id == 1) {
                offset = 6;
            }
            auto *host_mem = new T[tile_rows * tile_cols];
            size_t number_of_tiles_in_sub_matrix;
            for (int i = 0; i < number_of_sub_matrices; i++) {
                number_of_tiles_in_sub_matrix = sub_matrices[i]->GetNumberofTiles();
                for (int j = 0; j < number_of_tiles_in_sub_matrix; j++) {
                    hcorepp::memory::Memcpy(host_mem,
                                            sub_matrices[i]->GetTiles()[j]->GetTileSubMatrix(0),
                                            tile_rows * tile_cols, context.GetMainContext(),
                                            hcorepp::memory::MemoryTransfer::DEVICE_TO_HOST);
                    context.SyncMainContext();
                    for (int k = 0; k < tile_cols * tile_rows; k++) {
                        REQUIRE(std::abs(host_mem[k] - matrix_data_expected[k + offset]) <= std::abs(eps));
                    }
                    offset += tile_rows * tile_cols;
                }
            }

            /** TEST GET SUBMATRIX **/
            offset = (id - size + 1) * tile_cols * tile_rows * global_number_of_tiles;
            if (id == 0) {
                offset = 0;
            } else if (id == 1) {
                offset = 6;
            }
            for (int i = 0; i < number_of_sub_matrices; i++) {
                auto sub_matrix = &matrix.GetSubMatrix(i);
                number_of_tiles_in_sub_matrix = sub_matrix->GetNumberofTiles();
                for (int j = 0; j < number_of_tiles_in_sub_matrix; j++) {
                    hcorepp::memory::Memcpy(host_mem, sub_matrix->GetTiles()[j]->GetTileSubMatrix(0),
                                            tile_rows * tile_cols, context.GetMainContext(),
                                            hcorepp::memory::MemoryTransfer::DEVICE_TO_HOST);
                    context.SyncMainContext();
                    for (int k = 0; k < tile_cols * tile_rows; k++) {
                        REQUIRE(std::abs(host_mem[k] - matrix_data_expected[k + offset]) <= std::abs(eps));
                    }
                    offset += tile_rows * tile_cols;
                }
            }


            offset = (id - size + 1) * tile_cols * tile_rows * global_number_of_tiles;
            if (id == 0) {
                offset = 0;
            } else if (id == 1) {
                offset = 6;
            }
            number_of_sub_matrices = global_number_of_tiles_in_col >= size ? 1 : 2;

            for (int i = 0; i < number_of_sub_matrices; i++) {
                size_t col_idx = id - size + 1 + i;
                number_of_tiles_in_sub_matrix = 3;

                if (id == 0) {
                    col_idx = 0;
                } else if (id == 1) {
                    col_idx = 1;
                }
                for (int j = 0; j < number_of_tiles_in_sub_matrix; j++) {
                    size_t row_idx = j;
                    auto tile = matrix.GetTilePointer(row_idx, col_idx);
                    hcorepp::memory::Memcpy(host_mem, tile->GetTileSubMatrix(0), tile_rows * tile_cols,
                                            context.GetMainContext(),
                                            hcorepp::memory::MemoryTransfer::DEVICE_TO_HOST);
                    context.SyncMainContext();
                    for (int k = 0; k < tile_cols * tile_rows; k++) {

                        REQUIRE(std::abs(host_mem[k] - matrix_data_expected[k + offset]) <= std::abs(eps));
                    }
                    offset += tile_rows * tile_cols;
                }
            }
            delete[] host_mem;

        } else {
            REQUIRE(matrix.GetNumOfSubMatrices() == 0);
        }
    }

    SECTION("Test RM, slowestdimdecomposer 2 submatrices") {
        int id = 0, size = 1;
        /**
        * matrix
        * 0    1  | 2    3
        * ------  | -------
        * ================
        * 4    5  | 6    7
        * ------  | -------
        * 8    9  | 10  11
        */
        hicmapp::runtime::HicmaCommunicator communicator;
#ifdef HICMAPP_USE_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &id);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        communicator.SetMPICommunicator(MPI_COMM_WORLD);
#endif

        hicmapp::runtime::HicmaContext context;
        T matrix_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        T matrix_data_expected[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        T tiles_data_expected[][6] = {{0,  1},
                                      {2,  3},
                                      {4,  5},
                                      {6,  7},
                                      {8,  9},
                                      {10, 11}};
        size_t matrix_rows = 3;
        size_t matrix_cols = 4;
        size_t tile_rows = 1;
        size_t tile_cols = 2;
        float eps = 1e-6;
        StorageLayout storage_layout = StorageLayout::HicmaRM;
        size_t rank = 0;
        size_t total_number_of_sub_matrices = size;
        size_t number_of_sub_matrices = size;
        size_t number_of_tile_per_sub_matrix[] = {2, 4};
        SlowestDimDecomposer matrix_decomposer(number_of_sub_matrices, storage_layout);

        Matrix<T> matrix(matrix_data, matrix_rows, matrix_cols, tile_rows, tile_cols, storage_layout,
                         matrix_decomposer, context,
                         rank);
        context.SyncMainContext();
        /** TEST GET NUMBER OF SUB MATRICES **/
        if (id < number_of_sub_matrices) {

            ///for MPI support
//            number_of_sub_matrices = number_of_sub_matrices / size == 0 ? 1 : number_of_sub_matrices / size;

            REQUIRE(matrix.GetNumOfSubMatrices() == number_of_sub_matrices);

            /** TEST GET NUMBER OF TILES IN ROW **/
            REQUIRE(matrix.GetNumOfGlobalTilesInRows() == 3);

            /** TEST GET NUMBER OF TILES IN ROW **/
            REQUIRE(matrix.GetNumOfGlobalTilesInCols() == 2);

            //TODO: test contains tile when fixed.

            /** TEST GET STORAGE LAYOUT **/
            REQUIRE(matrix.GetStorageLayout() == storage_layout);

            /** TEST IS MATRIX VALID **/
            REQUIRE(matrix.IsMatrixValid() == true);

            /** TEST NUMBER OF ROWS IN TILE **/
            REQUIRE(matrix.GetNumOfRowsInTile() == tile_rows);

            /** TEST NUMBER OF COLS IN TILE **/
            REQUIRE(matrix.GetNumOfColsInTile() == tile_cols);

            /** TEST NUMBER OF ROWS IN MATRIX **/
            REQUIRE(matrix.GetGlobalNumOfRowsInMatrix() == matrix_rows);

            /** TEST NUMBER OF COLS IN MATRIX **/
            REQUIRE(matrix.GetGlobalNumOfColsInMatrix() == matrix_cols);

            /** TEST GET SUBMATRICES **/
            auto sub_matrices = matrix.GetSubMatrices();

            size_t offset = 0;
            if (id == 1) {
                offset = +id * tile_cols * tile_rows * 2;
            }

            auto *host_mem = new T[tile_rows * tile_cols];
            for (int i = 0; i < number_of_sub_matrices; i++) {
                size_t number_of_tiles_in_sub_matrix = sub_matrices[i]->GetNumberofTiles();
                for (int j = 0; j < number_of_tiles_in_sub_matrix; j++) {
                    hcorepp::memory::Memcpy(host_mem,
                                            sub_matrices[i]->GetTiles()[j]->GetTileSubMatrix(0),
                                            tile_rows * tile_cols, context.GetMainContext(),
                                            hcorepp::memory::MemoryTransfer::DEVICE_TO_HOST);
                    context.SyncMainContext();
                    for (int k = 0; k < tile_cols * tile_rows; k++) {
                        REQUIRE(std::abs(host_mem[k] - matrix_data_expected[k + offset]) <= std::abs(eps));
                    }
                    offset += tile_rows * tile_cols;
                }
            }

            /** TEST GET SUBMATRIX **/
            offset = 0;
            if (id == 1) {
                offset = +id * tile_cols * tile_rows * 2;
            }

            for (int i = 0; i < number_of_sub_matrices; i++) {
                auto sub_matrix = &matrix.GetSubMatrix(i);
                size_t number_of_tiles_in_sub_matrix = sub_matrix->GetNumberofTiles();
                for (int j = 0; j < number_of_tiles_in_sub_matrix; j++) {
                    hcorepp::memory::Memcpy(host_mem, sub_matrix->GetTiles()[j]->GetTileSubMatrix(0),
                                            tile_rows * tile_cols, context.GetMainContext(),
                                            hcorepp::memory::MemoryTransfer::DEVICE_TO_HOST);
                    context.SyncMainContext();
                    for (int k = 0; k < tile_cols * tile_rows; k++) {
                        REQUIRE(std::abs(host_mem[k] - matrix_data_expected[k + offset]) <= std::abs(eps));
                    }
                    offset += tile_rows * tile_cols;
                }
            }

            offset = id * 2;
            auto expected_offset = offset;
            for (int i = 0, idx = id; i < number_of_sub_matrices, idx < total_number_of_sub_matrices; i++, idx += size) {
                for (int j = 0; j < number_of_tile_per_sub_matrix[idx]; j++) {
                    size_t row = (offset + j) / 2;
                    size_t col = (offset + j) % 2;

                    auto tile = matrix.GetTilePointer(row, col);
                    hcorepp::memory::Memcpy(host_mem, tile->GetTileSubMatrix(0), tile_rows * tile_cols,
                                            context.GetMainContext(),
                                            hcorepp::memory::MemoryTransfer::DEVICE_TO_HOST);
                    context.SyncMainContext();
                    for (int k = 0; k < tile_cols * tile_rows; k++) {
                        REQUIRE(std::abs(host_mem[k] - matrix_data_expected[k + expected_offset]) <= std::abs(eps));
                    }
                    expected_offset += size * 2;
                }
                offset += size * 2;
            }
            delete[] host_mem;

        } else {
            REQUIRE(matrix.GetNumOfSubMatrices() == 0);
        }
    }

    SECTION("Test RM, default decomposer") {
        int id = 0, size = 1;

        /**
         *      matrix
         * 0    1 | 2    3
         * ------ | -------
         * 4    5 | 6    7
         * ------ | -------
         * 8    9 | 10   11
         */

        hicmapp::runtime::HicmaCommunicator communicator;
#ifdef HICMAPP_USE_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &id);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        communicator.SetMPICommunicator(MPI_COMM_WORLD);
#endif

        hicmapp::runtime::HicmaContext context;
        T matrix_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        T matrix_data_expected[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        T tiles_data_expected[][6] = {{0,  1},
                                      {2,  3},
                                      {4,  5},
                                      {6,  7},
                                      {8,  9},
                                      {11, 10}};
        size_t matrix_rows = 3;
        size_t matrix_cols = 4;
        size_t tile_rows = 1;
        size_t tile_cols = 2;
        float eps = 1e-6;
        size_t global_number_of_tiles = 6;
        size_t global_number_of_tiles_in_rows = 3;
        StorageLayout storage_layout = StorageLayout::HicmaRM;
        size_t rank = 0;
        size_t total_number_of_sub_matrices = 1;
        size_t number_of_sub_matrices = 1;
        size_t number_of_tile_per_sub_matrix = 6;

        Matrix<T> matrix(matrix_data, matrix_rows, matrix_cols, tile_rows, tile_cols, storage_layout, context, rank);
        if (global_number_of_tiles_in_rows >= size || id == size - 1) {

            ///for MPI support
//            number_of_sub_matrices = number_of_sub_matrices / size == 0 ? 1 : number_of_sub_matrices / size;

            /** TEST GET NUMBER OF SUB MATRICES **/
            REQUIRE(matrix.GetNumOfSubMatrices() == number_of_sub_matrices);

            /** TEST GET NUMBER OF TILES IN ROW **/
            REQUIRE(matrix.GetNumOfGlobalTilesInRows() == 3);

            /** TEST GET NUMBER OF TILES IN ROW **/
            REQUIRE(matrix.GetNumOfGlobalTilesInCols() == 2);

            ///TODO: test contains tile when fixed.

            /** TEST GET STORAGE LAYOUT **/
            REQUIRE(matrix.GetStorageLayout() == storage_layout);

            /** TEST IS MATRIX VALID **/
            REQUIRE(matrix.IsMatrixValid() == true);

            /** TEST NUMBER OF ROWS IN TILE **/
            REQUIRE(matrix.GetNumOfRowsInTile() == tile_rows);

            /** TEST NUMBER OF COLS IN TILE **/
            REQUIRE(matrix.GetNumOfColsInTile() == tile_cols);

            /** TEST NUMBER OF ROWS IN MATRIX **/
            REQUIRE(matrix.GetGlobalNumOfRowsInMatrix() == matrix_rows);

            /** TEST NUMBER OF COLS IN MATRIX **/
            REQUIRE(matrix.GetGlobalNumOfColsInMatrix() == matrix_cols);

            /** TEST GET SUBMATRICES **/
            auto sub_matrices = matrix.GetSubMatrices();
            size_t offset = (id - size + 1) * tile_cols * tile_rows * global_number_of_tiles;
            if (size <= global_number_of_tiles_in_rows) {
                offset = 4 * id;
            }
            size_t number_of_tiles_in_sub_matrix;
            auto *host_mem = new T[tile_rows * tile_cols];
            for (int i = 0; i < number_of_sub_matrices; i++) {
                number_of_tiles_in_sub_matrix = sub_matrices[i]->GetNumberofTiles();
                for (int j = 0; j < number_of_tiles_in_sub_matrix; j++) {
                    hcorepp::memory::Memcpy(host_mem,
                                            sub_matrices[i]->GetTiles()[j]->GetTileSubMatrix(0),
                                            tile_rows * tile_cols, context.GetMainContext(),
                                            hcorepp::memory::MemoryTransfer::DEVICE_TO_HOST);
                    context.SyncMainContext();
                    for (int k = 0; k < tile_cols * tile_rows; k++) {
                        REQUIRE(std::abs(host_mem[k] - matrix_data_expected[k + offset]) <= std::abs(eps));
                    }
                    offset += tile_rows * tile_cols;
                }
            }
            /** TEST GET SUBMATRIX **/
            offset = (id - size + 1) * tile_cols * tile_rows * global_number_of_tiles;
            if (size <= global_number_of_tiles_in_rows) {
                offset = 4 * id;
            }
            for (int i = 0; i < number_of_sub_matrices; i++) {
                auto sub_matrix = &matrix.GetSubMatrix(i);
                number_of_tiles_in_sub_matrix = sub_matrix->GetNumberofTiles();
                for (int j = 0; j < number_of_tiles_in_sub_matrix; j++) {
                    hcorepp::memory::Memcpy(host_mem, sub_matrix->GetTiles()[j]->GetTileSubMatrix(0),
                                            tile_rows * tile_cols, context.GetMainContext(),
                                            hcorepp::memory::MemoryTransfer::DEVICE_TO_HOST);
                    context.SyncMainContext();
                    for (int k = 0; k < tile_cols * tile_rows; k++) {
                        REQUIRE(std::abs(host_mem[k] - matrix_data_expected[k + offset]) <= std::abs(eps));
                    }
                    offset += tile_rows * tile_cols;
                }
            }

            offset = (id - size + 1) * tile_cols * tile_rows * global_number_of_tiles;
            if (size <= global_number_of_tiles_in_rows) {
                offset = 4 * id;
            }

            int idx = 0;
            if (size == 2 && id == 1) {
                idx = 1;
            }
            int starting = id;
            int until = 3 / size + id + idx;
            if (size > global_number_of_tiles_in_rows) {
                starting = 0;
                until = 3;
            }
            for (int k = starting; k < until; k++) {
                size_t row_idx = k;
                for (int j = 0; j < matrix.GetNumOfGlobalTilesInCols(); j++) {
                    size_t col_idx = j;
                    auto tile = matrix.GetTilePointer(row_idx, col_idx);
                    hcorepp::memory::Memcpy(host_mem, tile->GetTileSubMatrix(0), tile_rows * tile_cols,
                                            context.GetMainContext(),
                                            hcorepp::memory::MemoryTransfer::DEVICE_TO_HOST);
                    context.SyncMainContext();
                    for (int i = 0; i < tile_rows * tile_cols; i++) {
                        REQUIRE(std::abs(host_mem[i] - matrix_data_expected[i + offset]) <= std::abs(eps));
                    }
                    offset += tile_cols * tile_rows;
                }
            }

            /** TEST GET TILE **/
//            offset = id * 2;
//            for (int i = 0, idx = id;
//                 i < number_of_sub_matrices, idx < total_number_of_sub_matrices; i++, idx += size) {
//                for (int j = 0; j < number_of_tile_per_sub_matrix; j++) {
//                    size_t row = (offset + j) / 2;
//                    size_t col = (offset + j) % 2;
//
//                    auto tile = matrix.GetTilePointer(row, col);
//                    for (int k = 0; k < tile_cols * tile_rows; k++) {
//                                tiles_data_expected[j + offset][k]);
//                    }
//                }
//                offset += size * 2;
//            }
            delete[] host_mem;

        } else {
            REQUIRE(matrix.GetNumOfSubMatrices() == 0);
        }
    }

    /** Decomposer Tests need to be Revised. Assumptions to be corrected */
//    SECTION("Test RM, twodimcyclicdecomposer 4 submatrices") {
//        int id = 0, size = 1;
//        /**
//             * ______________
//             * | 0 | 1 || 6|
//             * | 2 | 3 || 7|
//             * | 4 | 5 || 8|
//             * --------------
//             * | 9 |10 || 11|
//             * --------------
//             */
//        hicmapp::runtime::HicmaCommunicator communicator;
//#ifdef HICMAPP_USE_MPI
//        MPI_Comm_rank(MPI_COMM_WORLD, &id);
//        MPI_Comm_size(MPI_COMM_WORLD, &size);
//        communicator.SetMPICommunicator(MPI_COMM_WORLD);
//#endif
//
//        hicmapp::runtime::HicmaContext context;
//        T matrix_data[] = {0, 1, 6, 2, 3, 7, 4, 5, 8, 9, 10, 11};
//        T matrix_data_expected[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
//        T sub_matrix_1_data[] = {0, 1, 2, 3, 4, 5};
//        T sub_matrix_2_data[] = {6, 7, 8};
//        T sub_matrix_3_data[] = {9, 10};
//        T sub_matrix_4_data[] = {11};
//        T *sub_matrix_data[] = {sub_matrix_1_data, sub_matrix_2_data, sub_matrix_3_data, sub_matrix_4_data};
//        size_t sub_matrix_global_starting_row_index[] = {0, 0, 3, 3};
//        size_t sub_matrix_global_starting_col_index[] = {0, 2, 0, 2};
//        size_t matrix_rows = 4;
//        size_t matrix_cols = 3;
//        size_t tile_rows = 1;
//        size_t tile_cols = 1;
//        float eps = 1e-6;
//        StorageLayout storage_layout = StorageLayout::HicmaRM;
//        size_t rank = 0;
//        size_t number_of_processes_in_row = 3;
//        size_t number_of_processes_in_col = 2;
//        size_t total_number_of_sub_matrices = 4;
//        size_t number_of_sub_matrices = 4;
//
//        TwoDimCyclicDecomposer matrix_decomposer(number_of_processes_in_row, number_of_processes_in_col);
//
//        Matrix<T> matrix(matrix_data, matrix_rows, matrix_cols, tile_rows, tile_cols, storage_layout,
//                         matrix_decomposer, context,
//                         rank);
//        size_t num_sub = matrix.GetNumOfSubMatrices();
//        if (id < number_of_sub_matrices) {
//
//            if (size == 1) {
//                // assuming 1 * 1 tiles.
//                number_of_sub_matrices = matrix_rows * matrix_cols;
//            } else {
//                ///for MPI support
//                number_of_sub_matrices = number_of_sub_matrices / size == 0 ? 1 : number_of_sub_matrices / size;
//                number_of_sub_matrices = (size == 3 && id == 0) ? 2 : number_of_sub_matrices;
//            }
//
//            REQUIRE(matrix.GetNumOfSubMatrices() == number_of_sub_matrices);
//
//            /** TEST GET NUMBER OF TILES IN ROW **/
//            REQUIRE(matrix.GetNumOfGlobalTilesInRows() == 4);
//
//            /** TEST GET NUMBER OF TILES IN ROW **/
//            REQUIRE(matrix.GetNumOfGlobalTilesInCols() == 3);
//
//            /** TEST GET STORAGE LAYOUT **/
//            REQUIRE(matrix.GetStorageLayout() == storage_layout);
//
//            /** TEST IS MATRIX VALID **/
//            REQUIRE(matrix.IsMatrixValid() == true);
//
//            /** TEST NUMBER OF ROWS IN TILE **/
//            REQUIRE(matrix.GetNumOfRowsInTile() == tile_rows);
//
//            /** TEST NUMBER OF COLS IN TILE **/
//            REQUIRE(matrix.GetNumOfColsInTile() == tile_cols);
//
//            /** TEST NUMBER OF ROWS IN MATRIX **/
//            REQUIRE(matrix.GetGlobalNumOfRowsInMatrix() == matrix_rows);
//
//            /** TEST NUMBER OF COLS IN MATRIX **/
//            REQUIRE(matrix.GetGlobalNumOfColsInMatrix() == matrix_cols);
//
//            /** TEST GET SUBMATRICES **/
//            size_t expected_index = id;
//            auto *host_mem = new T;
//            auto sub_matrices = matrix.GetSubMatrices();
//            for (int i = 0; i < number_of_sub_matrices; i++) {
//                size_t number_of_tiles_in_sub_matrix = sub_matrices[i]->GetNumberofTiles();
//                for (int j = 0; j < number_of_tiles_in_sub_matrix; j++) {
//                    hcorepp::memory::Memcpy(host_mem,
//                                            sub_matrices[i]->GetTiles()[j]->GetTileSubMatrix(0), 1,
//                                            context.GetMainContext(),
//                                            hcorepp::memory::MemoryTransfer::DEVICE_TO_HOST);
//                    context.SyncMainContext();
//                    if (size == 1) {
//                        REQUIRE(std::abs(*host_mem - matrix_data[i]) <= std::abs(eps));
//                    } else {
//                        REQUIRE(std::abs(*host_mem - sub_matrix_data[expected_index][j]) <= std::abs(eps));
//                    }
//                }
//                expected_index += size;
//            }
//
//            /** TEST GET SUBMATRIX **/
//            expected_index = id;
//            for (int i = 0; i < number_of_sub_matrices; i++) {
//                auto sub_matrix = &matrix.GetSubMatrix(i);
//                size_t number_of_tiles_in_sub_matrix = sub_matrix->GetNumberofTiles();
//                for (int j = 0; j < number_of_tiles_in_sub_matrix; j++) {
//                    hcorepp::memory::Memcpy(host_mem, sub_matrix->GetTiles()[j]->GetTileSubMatrix(0), 1,
//                                            context.GetMainContext(),
//                                            hcorepp::memory::MemoryTransfer::DEVICE_TO_HOST);
//                    context.SyncMainContext();
//                    if (size == 1) {
//                        REQUIRE(std::abs(*host_mem - matrix_data[i]) <= std::abs(eps));
//                    } else {
//                        REQUIRE(std::abs(*host_mem - sub_matrix_data[expected_index][j]) <= std::abs(eps));
//                    }
//                }
//                expected_index += size;
//            }
//
//            /** TEST GET TILE POINTER **/
//            for (size_t idx_i = id; idx_i < total_number_of_sub_matrices; idx_i += size) {
//
//                size_t i = sub_matrix_global_starting_row_index[idx_i];
//                auto num_of_rows = std::min(number_of_processes_in_row, matrix_rows - i);
//                size_t j = sub_matrix_global_starting_col_index[idx_i];
//                auto num_of_cols = std::min(number_of_processes_in_col, matrix_cols - j);
//
//                for (int r = 0; r < num_of_rows; r++) {
//                    for (int c = 0; c < num_of_cols; c++) {
//                        auto global_idx_row = r + i;
//                        auto global_idx_col = c + j;
//                        auto tile = matrix.GetTilePointer(global_idx_row, global_idx_col);
//                        hcorepp::memory::Memcpy(host_mem, tile->GetTileSubMatrix(0), 1,
//                                                context.GetMainContext(),
//                                                hcorepp::memory::MemoryTransfer::DEVICE_TO_HOST);
//                        context.SyncMainContext();
//                        REQUIRE(std::abs(*host_mem - matrix_data[global_idx_row * matrix_cols + global_idx_col]) <=
//                                std::abs(eps));
//                    }
//                }
//            }
//            delete host_mem;
//
//        } else {
//            REQUIRE(matrix.GetNumOfSubMatrices() == 0);
//
//        }
//    }

    SECTION("Test RM, non-divisible Matrix") {
        /**
         * 0    1  || 2    3
         * ==================
         * 4    5  || 6    7
         * ==================
         * 8    9  || 10   11
         * ==================
         * 12   13 || 14   15
         * --------||--------
         * 16   17 || 18   19
         *
         */
        int id = 0, size = 1;

        hicmapp::runtime::HicmaCommunicator communicator;
#ifdef HICMAPP_USE_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &id);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        communicator.SetMPICommunicator(MPI_COMM_WORLD);
#endif

        hicmapp::runtime::HicmaContext context;
        T matrix_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
        T matrix_data_expected[][4] = {{0,  1,  2,  3},
                                       {4,  5,  6,  7},
                                       {8,  9,  10, 11},
                                       {12, 13, 14, 15},
                                       {16, 17, 18, 19}};
        size_t matrix_rows = 5;
        T tiles_data_expected[][2] = {{0,  1},
                                      {2,  3},
                                      {4,  5},
                                      {6,  7},
                                      {8,  9},
                                      {10, 11},
                                      {12, 13},
                                      {14, 15},
                                      {16, 17},
                                      {18, 19}};
        size_t matrix_cols = 4;
        size_t tile_rows = 1;
        size_t tile_cols = 2;
        StorageLayout storage_layout = StorageLayout::HicmaRM;
        size_t rank = 0;
        size_t total_number_of_tiles = 10;
        float eps = 1e-6;
        size_t total_number_of_sub_matrices = size;
        size_t number_of_sub_matrices = size;
        SlowestDimDecomposer matrix_decomposer(number_of_sub_matrices, storage_layout);

        Matrix<T> matrix(matrix_data, matrix_rows, matrix_cols, tile_rows, tile_cols, storage_layout,
                         matrix_decomposer, context,
                         rank);
        if (id < number_of_sub_matrices) {

            ///for MPI support
//            number_of_sub_matrices = number_of_sub_matrices / size == 0 ? 1 : number_of_sub_matrices / size;
            size_t remainder_sub_matrices = total_number_of_sub_matrices - (number_of_sub_matrices * size);

            if (id == 0 && size < total_number_of_sub_matrices) {
                number_of_sub_matrices += remainder_sub_matrices;
            }
            /** TEST GET NUMBER OF SUB MATRICES **/
            REQUIRE(matrix.GetNumOfSubMatrices() == number_of_sub_matrices);

            /** TEST GET NUMBER OF SUB MATRICES **/
            REQUIRE(matrix.GetNumOfSubMatrices() == number_of_sub_matrices);

            /** TEST GET NUMBER OF TILES IN ROW **/
            REQUIRE(matrix.GetNumOfGlobalTilesInRows() == 5);

            /** TEST GET NUMBER OF TILES IN ROW **/
            REQUIRE(matrix.GetNumOfGlobalTilesInCols() == 2);

            ///TODO: test contains tile when fixed.

            /** TEST GET STORAGE LAYOUT **/
            REQUIRE(matrix.GetStorageLayout() == storage_layout);

            /** TEST IS MATRIX VALID **/
            REQUIRE(matrix.IsMatrixValid() == true);

            /** TEST NUMBER OF ROWS IN TILE **/
            REQUIRE(matrix.GetNumOfRowsInTile() == tile_rows);

            /** TEST NUMBER OF COLS IN TILE **/
            REQUIRE(matrix.GetNumOfColsInTile() == tile_cols);

            /** TEST NUMBER OF ROWS IN MATRIX **/
            REQUIRE(matrix.GetGlobalNumOfRowsInMatrix() == matrix_rows);

            /** TEST NUMBER OF COLS IN MATRIX **/
            REQUIRE(matrix.GetGlobalNumOfColsInMatrix() == matrix_cols);
        }
    }

            /** These tests need to be revised as well */

////            for (int i = 0, j = id; i < number_of_sub_matrices && j < total_number_of_sub_matrices; i++, j += size) {
////                REQUIRE(matrix.GetSubMatrix(i).GetNumberofTiles() == number_of_tiles_per_sub_matrix[j]);
////            }
//
//
//            /** TEST GET SUB-MATRICES*/
//            auto sub_matrices = matrix.GetSubMatrices();
//            size_t offset = id * 2;
//            auto *host_mem = new T[tile_rows * tile_cols];
//            for (int i = 0; i < number_of_sub_matrices; i++) {
//                size_t number_of_tiles_in_sub_matrix = sub_matrices[i]->GetNumberofTiles();
//                for (int j = 0; j < number_of_tiles_in_sub_matrix; j++) {
//                    hcorepp::memory::Memcpy(host_mem,
//                                            sub_matrices[i]->GetTiles()[j]->GetTileSubMatrix(0),
//                                            tile_rows * tile_cols, context.GetMainContext(),
//                                            hcorepp::memory::MemoryTransfer::DEVICE_TO_HOST);
//                    context.SyncMainContext();
//                    for (int k = 0; k < tile_cols * tile_rows; k++) {
//                        REQUIRE(std::abs(host_mem[k] - tiles_data_expected[j + offset][k]) <= std::abs(eps));
//                    }
//                }
//                offset += size * 2;
//            }
//
//            /** TEST GET SUBMATRIX **/
//            offset = id * 2;
//
//            for (int i = 0; i < number_of_sub_matrices; i++) {
//                auto sub_matrix = &matrix.GetSubMatrix(i);
//                size_t number_of_tiles_in_sub_matrix = sub_matrix->GetNumberofTiles();
//                for (int j = 0; j < number_of_tiles_in_sub_matrix; j++) {
//                    hcorepp::memory::Memcpy(host_mem, sub_matrix->GetTiles()[j]->GetTileSubMatrix(0),
//                                            tile_rows * tile_cols, context.GetMainContext(),
//                                            hcorepp::memory::MemoryTransfer::DEVICE_TO_HOST);
//                    context.SyncMainContext();
//                    for (int k = 0; k < tile_cols * tile_rows; k++) {
//                        REQUIRE(std::abs(host_mem[k] - tiles_data_expected[j + offset][k]) <= std::abs(eps));
//                    }
//                }
//                offset += size * 2;
//            }
////
////            /** TEST GET TILE POINTER **/
//            offset = id * 2;
//            for (int i = 0, idx = id;
//                 i < number_of_sub_matrices, idx < total_number_of_sub_matrices; i++, idx += size) {
//                for (int j = 0; j < number_of_tiles_per_sub_matrix[idx]; j++) {
//                    size_t row = (offset + j) / 2;
//                    size_t col = (offset + j) % 2;
//
//                    auto tile = matrix.GetTilePointer(row, col);
//                    hcorepp::memory::Memcpy(host_mem, tile->GetTileSubMatrix(0), tile_rows * tile_cols,
//                                            context.GetMainContext(),
//                                            hcorepp::memory::MemoryTransfer::DEVICE_TO_HOST);
//                    context.SyncMainContext();
//                    for (int k = 0; k < tile_cols * tile_rows; k++) {
//                        REQUIRE(std::abs(host_mem[k] - tiles_data_expected[j + offset][k]) <= std::abs(eps));
//                    }
//                }
//                offset += size * 2;
//            }
//            delete[] host_mem;
//
//        } else {
//            REQUIRE(matrix.GetNumOfSubMatrices() == 0);
//        }
//    }
}


TEMPLATE_TEST_CASE("MatrixTest", "[Matrix]", double, float) {
    TEST_MATRIX<TestType>();
}

