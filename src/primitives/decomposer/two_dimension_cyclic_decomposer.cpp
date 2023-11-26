#include <hicmapp/primitives/decomposer/concrete/two_dimension_cyclic_decomposer.hpp>
#include <hicmapp/common/definitions.h>
#include <iostream>

namespace hicmapp::primitives {

    TwoDimCyclicDecomposer::TwoDimCyclicDecomposer(size_t aNumOfProcessesInRows, size_t aNumOfProcessesInCols)
            : mNumOfProcessesInRows(aNumOfProcessesInRows),
              mNumOfProcessesInCols(aNumOfProcessesInCols) {
    }

    std::vector<MatrixSpecifications>
    TwoDimCyclicDecomposer::Decompose(size_t aGlobalMatrixTilesInRows, size_t aGlobalMatrixTilesInCols,
                                      bool aDiagonalMatrix) {

        std::vector<MatrixSpecifications> matrix_specifications;

        int process_id = 0;
        int num_of_processes = 1;
#ifdef HICMAPP_USE_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
        MPI_Comm_size(MPI_COMM_WORLD, &num_of_processes);
#endif

        if (num_of_processes > mNumOfProcessesInCols * mNumOfProcessesInRows) {
            return {};
        }

        if (num_of_processes == 1) {
            mNumOfProcessesInCols = 1;
            mNumOfProcessesInRows = 1;
            return {{aGlobalMatrixTilesInRows, aGlobalMatrixTilesInCols, 0, 0, mNumOfProcessesInRows,
                     mNumOfProcessesInCols, process_id}};
        }

        if (num_of_processes < mNumOfProcessesInCols * mNumOfProcessesInRows) {
            throw std::runtime_error(
                    " Num of processes passed is less than requested grid size \n ");
        }

        auto num_of_tiles_in_rows = 1;
        auto num_of_tiles_in_cols = 1;
        size_t owner_id = 0;
        size_t initial_owner_row;

        for (size_t tile_row_idx = 0; tile_row_idx < aGlobalMatrixTilesInRows; tile_row_idx++) {
            if (tile_row_idx % mNumOfProcessesInRows == 0) {
                initial_owner_row = 0;
            } else {
                initial_owner_row = mNumOfProcessesInCols;
            }
            for (size_t tile_col_idx = 0; tile_col_idx < aGlobalMatrixTilesInCols; tile_col_idx++) {
                if (tile_col_idx % mNumOfProcessesInCols == 0) {
                    owner_id = initial_owner_row % num_of_processes;
                }
                matrix_specifications.emplace_back(num_of_tiles_in_rows, num_of_tiles_in_cols,
                                                   tile_row_idx, tile_col_idx, mNumOfProcessesInRows,
                                                   mNumOfProcessesInCols, owner_id);
                owner_id = (owner_id + 1) % num_of_processes;
            }
        }
        return matrix_specifications;
    }
}
