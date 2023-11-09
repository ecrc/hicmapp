
#include <iostream>
#include <hicmapp/primitives/matrix.hpp>
#include <hicmapp/primitives/decomposer/concrete/slowest_dimension_decomposer.hpp>
#include <hicmapp/primitives/decomposer/concrete/two_dimension_cyclic_decomposer.hpp>

const int global_elements_in_rows = 8;
const int global_elements_in_cols = 8;

using namespace hicmapp::primitives;

int main(int argc, char *argv[]) {

    // 2d array allocation
    auto data = new float[global_elements_in_rows][global_elements_in_cols];

    size_t index = 0;
    for (int i = 0; i < global_elements_in_rows; i++) {
        for (int j = 0; j < global_elements_in_cols; j++) {
            data[i][j] = index;
            index++;
        }
    }

    for (int i = 0; i < global_elements_in_rows; i++) {
        for (int j = 0; j < global_elements_in_cols; j++) {
            std::cout << " data [" << i << "][" << j << "] = " << data[i][j] << " \t";
        }
        std::cout << "\n";
    }

    size_t num_of_sub_matrices = 2;
    size_t tile_rows = 2;
    size_t tile_cols = 2;
//    SlowestDimDecomposer decomposer(num_of_sub_matrices, hicmapp::common::StorageLayout::HicmaRM);

#ifdef HICMAPP_USE_MPI
    int id;
    MPI_Init(&argc, &argv);
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    std::cout << " TOTAL num of processes = " << size << "\n";
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    std::cout << " Process Id : " << id << " \n";
#endif

    hicmapp::runtime::HicmaContext context;

    size_t num_of_processes_in_rows = 2;
    size_t num_of_processes_in_cols = 3;
    TwoDimCyclicDecomposer decomposer(num_of_processes_in_rows, num_of_processes_in_cols);

    Matrix<float> matrix((float *) data, global_elements_in_rows,
                         global_elements_in_cols, tile_rows, tile_cols,
                         hicmapp::common::StorageLayout::HicmaRM, decomposer, context);

    auto sub_matrices = matrix.GetSubMatrices();

    int sub_matrix_idx = 0;
#ifdef HICMAPP_USE_MPI
//    for (int i = 0; i < size; i++) {
        if (id == 5) {
            std::cout << " PRINTING SUbmatrices for Process :: " << id << "\n";
#endif

            for (auto sub_matrix: sub_matrices) {
                std::cout << " ============== SUBMATRIX  " << sub_matrix_idx << " ================ \n";
                auto num_of_tiles = sub_matrix->GetNumberofTiles();
                std::cout << " NUM of TIles in subMatrix : " << num_of_tiles << "\n";
                auto tile_idx = 0;
                auto tiles = sub_matrix->GetTiles();
                for (auto tile: tiles) {
                    auto tile_rows = tile->GetNumOfRows();
                    auto tile_cols = tile->GetNumOfCols();
                    auto tile_data = tile->GetTileSubMatrix(0);
                    std::cout << " Printing daata in tile : " << tile_idx << "\n";
                    for (int i = 0; i < tile_rows; i++) {
                        for (int j = 0; j < tile_cols; j++) {
                            std::cout << " data [" << i << "][" << j << "] = " << tile_data[i * tile_cols + j] << "\t";
                        }
                        std::cout << "\n";
                    }
                    tile_idx++;
                }
                sub_matrix_idx++;
            }
#ifdef HICMAPP_USE_MPI
        }
//    }
#endif
    delete[] data;

#ifdef HICMAPP_USE_MPI
    MPI_Finalize();
#endif

}