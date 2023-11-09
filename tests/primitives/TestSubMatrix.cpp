#include <catch2/catch_all.hpp>
#include <hicmapp/primitives/submatrix.hpp>
#include <hcorepp/kernels/memory.hpp>

using namespace hicmapp::common;
using namespace hicmapp::primitives;

template<typename T>
void TEST_SUBMATRIX() {
    SECTION("Test CM, 1 sub_matrix matrix") {
        /** assuming a matrix composed of
         * 1 submatrix and 2 tiles:
         *
            0  6
            1  7
            2  8
            -----
            3  9
            4  10
            5  11
         *
        **/
        hicmapp::runtime::HicmaContext context;
        size_t global_matrix_cols = 2;
        size_t global_matrix_rows = 6;
        size_t sub_matrix_cols = 2;
        size_t sub_matrix_rows = 6;
        size_t tile_cols = 2;
        size_t tile_rows = 3;
        float eps = 1e-6;
        size_t tile_global_start_index_in_rows = 0;
        size_t tile_global_start_index_in_cols = 0;
        size_t owner_id = 0;
        T data_array[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        T tiles_data[] = {0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11};

        StorageLayout storage_layout = hicmapp::common::StorageLayout::HicmaCM;

        auto sub_matrix = new SubMatrix<T>(data_array, tile_rows, tile_cols, global_matrix_rows, global_matrix_cols,
                                           sub_matrix_rows, sub_matrix_cols, tile_global_start_index_in_rows,
                                           tile_global_start_index_in_cols, storage_layout, owner_id, context);

        /** TEST GET NUMBER OF TILES **/
        size_t number_of_tiles = sub_matrix->GetNumberofTiles();
        REQUIRE(number_of_tiles == 2);

        /** TEST GET NUMBER OF TILES IN ROW **/
        size_t number_of_tiles_in_row = sub_matrix->GetNumOfTilesinRows();
        REQUIRE(number_of_tiles_in_row == 2);

        /** TEST GET NUMBER OF TILES IN COL **/
        size_t number_of_tiles_in_col = sub_matrix->GetNumOfTilesinCols();
        REQUIRE(number_of_tiles_in_col == 1);

        /** TEST GET NUMBER OF TILES ROWS **/
        REQUIRE(sub_matrix->GetTileRows() == tile_rows);

        /** TEST GET NUMBER OF TILES COLS **/
        REQUIRE(sub_matrix->GetTileCols() == tile_cols);

        /** TEST GET TILE **/
        size_t offset = 0;
        auto tiles = sub_matrix->GetTiles();
        auto* host_mem = new T[tile_cols * tile_rows];
        for (int i = 0; i < number_of_tiles; i++) {
            hcorepp::memory::Memcpy(host_mem, tiles[i]->GetTileSubMatrix(0), tile_rows * tile_cols, context.GetMainContext(), hcorepp::memory::MemoryTransfer::DEVICE_TO_HOST);
            context.SyncMainContext();
            for (int k = 0; k < tile_rows * tile_cols; k++) {
                REQUIRE(std::abs(host_mem[k]  - tiles_data[k + offset]) <= std::abs(eps));
            }
            offset += tile_rows * tile_cols;
        }

        /** TEST GET TILE POINTER **/
        offset = 0;
        for (int i = 0; i < number_of_tiles_in_row; i++) {
            for (int j = 0; j < number_of_tiles_in_col; j++) {
                auto tile = sub_matrix->GetTilePointer(i, j);
                hcorepp::memory::Memcpy(host_mem, tile->GetTileSubMatrix(0), tile_rows * tile_cols, context.GetMainContext(), hcorepp::memory::MemoryTransfer::DEVICE_TO_HOST);
                context.SyncMainContext();
                for (int k = 0; k < tile_rows * tile_cols; k++) {
                    REQUIRE(std::abs(host_mem[k] - tiles_data[k + offset]) <= std::abs(eps));
                }
                offset += tile_rows * tile_cols;
            }
        }

        /** TEST CONTAINS TILE **/
        REQUIRE(sub_matrix->ContainsTile(0, 0) == true);
        REQUIRE(sub_matrix->ContainsTile(0, 1) == false);
        REQUIRE(sub_matrix->ContainsTile(1, 0) == true);
        REQUIRE(sub_matrix->ContainsTile(1, 1) == false);
        REQUIRE(sub_matrix->ContainsTile(0, 2) == false);

        /** TEST GetSubMatrixOwnerId **/
        REQUIRE(sub_matrix->GetSubMatrixOwnerId() == 0);

        /** TEST IS VALID **/
        REQUIRE(sub_matrix->IsValid() == true);
        delete [] host_mem;
        delete sub_matrix;
    }

    SECTION("Test CM, 1 sub_matrix matrix, null-pointer data") {

        /** assuming a matrix composed of
         * 1 submatrix and 4 tiles:
         *
         *
         *  n  n | n  n
         *  n  n | n  n
         *  -----|------
         *  n  n | n  n
         *  n  n | n  n
         *
        **/
        hicmapp::runtime::HicmaContext context;
        size_t global_matrix_cols = 4;
        size_t global_matrix_rows = 4;
        size_t sub_matrix_cols = 4;
        size_t sub_matrix_rows = 4;
        size_t tile_cols = 2;
        size_t tile_rows = 2;
        size_t tile_global_start_index_in_rows = 0;
        size_t tile_global_start_index_in_cols = 0;
        size_t owner_id = 0;
        T *data_array = nullptr;

        StorageLayout storage_layout = hicmapp::common::StorageLayout::HicmaCM;

        auto sub_matrix = new SubMatrix<T>(data_array, tile_rows, tile_cols, global_matrix_rows, global_matrix_cols,
                                           sub_matrix_rows, sub_matrix_cols, tile_global_start_index_in_rows,
                                           tile_global_start_index_in_cols, storage_layout, owner_id, context);

        /** TEST CONTAINS TILE **/
        REQUIRE(sub_matrix->ContainsTile(0, 0) == true);
        REQUIRE(sub_matrix->ContainsTile(0, 1) == true);
        REQUIRE(sub_matrix->ContainsTile(1, 0) == true);
        REQUIRE(sub_matrix->ContainsTile(1, 1) == true);
        REQUIRE(sub_matrix->ContainsTile(0, 2) == false);

        /** TEST GetSubMatrixOwnerId **/
        REQUIRE(sub_matrix->GetSubMatrixOwnerId() == 0);

        /** TEST IS VALID **/
        REQUIRE(sub_matrix->IsValid() == true);
        delete sub_matrix;
    }

    SECTION("Test RM, 1 sub_matrix matrix") {
        /** assuming a matrix composed of
         * 1 submatrix and 2 tiles:
         *
         0   1
         2   3
         4   5
        -------
         6   7
         8   9
         10 11
         *
        **/
        hicmapp::runtime::HicmaContext context;
        size_t global_matrix_cols = 2;
        size_t global_matrix_rows = 6;
        size_t sub_matrix_cols = 2;
        size_t sub_matrix_rows = 6;
        float eps = 1e-6;
        size_t tile_cols = 2;
        size_t tile_rows = 3;
        size_t tile_global_start_index_in_rows = 0;
        size_t tile_global_start_index_in_cols = 0;
        size_t owner_id = 0;
        T data_array[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        T tiles_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

        StorageLayout storage_layout = hicmapp::common::StorageLayout::HicmaRM;

        auto sub_matrix = new SubMatrix<T>(data_array, tile_rows, tile_cols, global_matrix_rows, global_matrix_cols,
                                           sub_matrix_rows, sub_matrix_cols, tile_global_start_index_in_rows,
                                           tile_global_start_index_in_cols, storage_layout, owner_id, context);

        /** TEST GET NUMBER OF TILES **/
        size_t number_of_tiles = sub_matrix->GetNumberofTiles();
        REQUIRE(number_of_tiles == 2);

        /** TEST GET NUMBER OF TILES IN ROW **/
        size_t number_of_tiles_in_row = sub_matrix->GetNumOfTilesinRows();
        REQUIRE(number_of_tiles_in_row == 2);

        /** TEST GET NUMBER OF TILES IN COL **/
        size_t number_of_tiles_in_col = sub_matrix->GetNumOfTilesinCols();
        REQUIRE(number_of_tiles_in_col == 1);

        /** TEST GET NUMBER OF TILES ROWS **/
        REQUIRE(sub_matrix->GetTileRows() == tile_rows);

        /** TEST GET NUMBER OF TILES COLS **/
        REQUIRE(sub_matrix->GetTileCols() == tile_cols);

        /** TEST GET TILE **/
        size_t offset = 0;
        auto tiles = sub_matrix->GetTiles();
        auto* host_mem = new T[tile_cols * tile_rows];
        for (int i = 0; i < number_of_tiles; i++) {
            hcorepp::memory::Memcpy(host_mem, tiles[i]->GetTileSubMatrix(0), tile_rows * tile_cols, context.GetMainContext(), hcorepp::memory::MemoryTransfer::DEVICE_TO_HOST);
            context.SyncMainContext();
            for (int k = 0; k < tile_rows * tile_cols; k++) {
                REQUIRE(std::abs(host_mem[k]  - tiles_data[k + offset]) <= std::abs(eps));
            }
            offset += tile_rows * tile_cols;
        }

        /** TEST GET TILE POINTER **/
        offset = 0;
        for (int i = 0; i < number_of_tiles_in_row; i++) {
            for (int j = 0; j < number_of_tiles_in_col; j++) {
                auto tile = sub_matrix->GetTilePointer(i, j);
                hcorepp::memory::Memcpy(host_mem, tile->GetTileSubMatrix(0), tile_rows * tile_cols, context.GetMainContext(), hcorepp::memory::MemoryTransfer::DEVICE_TO_HOST);
                context.SyncMainContext();
                for (int k = 0; k < tile_rows * tile_cols; k++) {
                    REQUIRE(std::abs(host_mem[k]  - tiles_data[k + offset]) <= std::abs(eps));
                }
                offset += tile_rows * tile_cols;
            }
        }

        /** TEST CONTAINS TILE **/
        REQUIRE(sub_matrix->ContainsTile(0, 0) == true);
        REQUIRE(sub_matrix->ContainsTile(0, 1) == false);
        REQUIRE(sub_matrix->ContainsTile(1, 0) == true);
        REQUIRE(sub_matrix->ContainsTile(1, 1) == false);
        REQUIRE(sub_matrix->ContainsTile(0, 2) == false);

        /** TEST GetSubMatrixOwnerId **/
        REQUIRE(sub_matrix->GetSubMatrixOwnerId() == 0);

        /** TEST IS VALID **/
        REQUIRE(sub_matrix->IsValid() == true);
        delete [] host_mem;
        delete sub_matrix;
    }

    SECTION("Test RM, 1 sub_matrix matrix, non-divisible") {
        /** assuming a matrix composed of
         * 1 submatrix and 2 tiles:
         *
        0	1    | 2	3    |	4
        5	6    | 7	8    |	9
        ----------------------------
        10	11   |  12	13   |	14
        15	16   |	17	18   |	19
        ----------------------------
        20	21   |	22	23   |	24
         *
        **/
        hicmapp::runtime::HicmaContext context;
        size_t global_matrix_cols = 5;
        size_t global_matrix_rows = 5;
        size_t sub_matrix_cols = 5;
        size_t sub_matrix_rows = 5;
        size_t tile_cols = 2;
        size_t tile_rows = 2;
        float eps = 1e-6;
        size_t tile_global_start_index_in_rows = 0;
        size_t tile_global_start_index_in_cols = 0;
        size_t owner_id = 0;
        T data_array[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        T tiles_data[] = {0, 1, 5, 6, 2, 3, 7, 8, 4, 9, 10, 11, 15, 16, 12, 13, 17, 18, 14, 19, 20, 21, 22, 23, 24};

        StorageLayout storage_layout = hicmapp::common::StorageLayout::HicmaRM;

        auto sub_matrix = new SubMatrix<T>(data_array, tile_rows, tile_cols, global_matrix_rows, global_matrix_cols,
                                           sub_matrix_rows, sub_matrix_cols, tile_global_start_index_in_rows,
                                           tile_global_start_index_in_cols, storage_layout, owner_id, context);

        /** TEST GET NUMBER OF TILES **/
        size_t number_of_tiles = sub_matrix->GetNumberofTiles();
        REQUIRE(number_of_tiles == 9);

        /** TEST GET NUMBER OF TILES IN ROW **/
        size_t number_of_tiles_in_row = sub_matrix->GetNumOfTilesinRows();
        REQUIRE(number_of_tiles_in_row == 3);

        /** TEST GET NUMBER OF TILES IN COL **/
        size_t number_of_tiles_in_col = sub_matrix->GetNumOfTilesinCols();
        REQUIRE(number_of_tiles_in_col == 3);

        /** TEST GET NUMBER OF TILES ROWS **/
        REQUIRE(sub_matrix->GetTileRows() == tile_rows);

        /** TEST GET NUMBER OF TILES COLS **/
        REQUIRE(sub_matrix->GetTileCols() == tile_cols);

        /** TEST GET TILE **/
        size_t offset = 0;
        auto tiles = sub_matrix->GetTiles();
        for (int i = 0; i < number_of_tiles; i++) {
            size_t tile_i_rows = tiles[i]->GetNumOfRows();
            size_t tile_i_cols = tiles[i]->GetNumOfCols();
            auto* host_mem = new T[tile_i_cols * tile_i_rows];
            hcorepp::memory::Memcpy(host_mem, tiles[i]->GetTileSubMatrix(0), tile_i_rows * tile_i_cols, context.GetMainContext(), hcorepp::memory::MemoryTransfer::DEVICE_TO_HOST);
            context.SyncMainContext();
            for (int k = 0; k < tile_i_rows * tile_i_cols; k++) {
                REQUIRE(std::abs(host_mem[k]  - tiles_data[k + offset]) <= std::abs(eps));
            }
            delete [] host_mem;
            offset += tile_i_rows * tile_i_cols;
        }

        /** TEST GET TILE POINTER **/
        offset = 0;
        for (int i = 0; i < number_of_tiles_in_row; i++) {
            for (int j = 0; j < number_of_tiles_in_col; j++) {
                auto tile = sub_matrix->GetTilePointer(i, j);
                size_t tile_i_rows = tile->GetNumOfRows();
                size_t tile_i_cols = tile->GetNumOfCols();
                auto* host_mem = new T[tile_i_cols * tile_i_rows];
                hcorepp::memory::Memcpy(host_mem, tile->GetTileSubMatrix(0), tile_i_rows * tile_i_cols, context.GetMainContext(), hcorepp::memory::MemoryTransfer::DEVICE_TO_HOST);
                context.SyncMainContext();
                for (int k = 0; k < tile_i_rows * tile_i_cols; k++) {
                    REQUIRE(std::abs(host_mem[k]  - tiles_data[k + offset]) <= std::abs(eps));
                }
                delete [] host_mem;
                offset += tile_i_rows * tile_i_cols;
            }
        }

        REQUIRE(sub_matrix->ContainsTile(0, 0) == true);
        REQUIRE(sub_matrix->ContainsTile(0, 1) == true);
        REQUIRE(sub_matrix->ContainsTile(0, 2) == true);
        REQUIRE(sub_matrix->ContainsTile(0, 3) == false);
        REQUIRE(sub_matrix->ContainsTile(1, 0) == true);
        REQUIRE(sub_matrix->ContainsTile(1, 1) == true);
        REQUIRE(sub_matrix->ContainsTile(1, 2) == true);
        REQUIRE(sub_matrix->ContainsTile(1, 3) == false);
        REQUIRE(sub_matrix->ContainsTile(2, 0) == true);
        REQUIRE(sub_matrix->ContainsTile(2, 1) == true);
        REQUIRE(sub_matrix->ContainsTile(2, 2) == true);
        REQUIRE(sub_matrix->ContainsTile(2, 3) == false);
        REQUIRE(sub_matrix->ContainsTile(3, 0) == false);



        /** TEST GetSubMatrixOwnerId **/
        REQUIRE(sub_matrix->GetSubMatrixOwnerId() == 0);

        /** TEST IS VALID **/
        REQUIRE(sub_matrix->IsValid() == true);
        delete sub_matrix;}
}

TEMPLATE_TEST_CASE("SubMatrixTest", "[SubMatrix]", float, double) {
    TEST_SUBMATRIX<TestType>();
}