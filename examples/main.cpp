#include <iostream>
#include "hicmapp/primitives/matrix.hpp"
#include "hicmapp/primitives/decomposer/concrete/slowest_dimension_decomposer.hpp"
#include "hicmapp/primitives/decomposer/concrete/two_dimension_cyclic_decomposer.hpp"

const int global_elements_in_rows = 4;
const int global_elements_in_cols = 4;

using namespace hicmapp::primitives;

int main(int argc, char *argv[]) {

    hcorepp::kernels::RunContext context;

    // 2d array allocation
    auto data = new float[global_elements_in_rows * global_elements_in_cols];

    for (int j = 0; j < global_elements_in_cols; j++) {
        for (int i = 0; i < global_elements_in_rows; i++) {
            int idx = j * global_elements_in_rows + i;
            data[idx] = idx;
        }
    }

    for (int j = 0; j < global_elements_in_cols; j++) {
        for (int i = 0; i < global_elements_in_rows; i++) {
            int idx = j * global_elements_in_rows + i;
            std::cout << " Input [" << i << "][" << j << "] = " << data[idx] << " \t";
        }
        std::cout << "\n";
    }


    int rank = 1;
    CompressionParameters parameters = {1e-3};
    auto *compressed_tile = new CompressedTile<float>(global_elements_in_rows, global_elements_in_cols, (float *) data,
                                                      global_elements_in_rows, parameters, blas::Layout::ColMajor,
                                                      context);

    auto COutput = new float[global_elements_in_rows * global_elements_in_cols];

    auto *new_tile = new CompressedTile<float>(global_elements_in_rows, global_elements_in_cols, nullptr,
                                   global_elements_in_rows, parameters, blas::Layout::ColMajor, context);
    new_tile = compressed_tile;

    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, global_elements_in_rows,
               global_elements_in_cols,
               new_tile->GetTileRank(), 1.0f,
               new_tile->GetUMatrix(),
               new_tile->GetNumOfRows(),
               new_tile->GetVMatrix(),
               new_tile->GetTileRank(), 0.0f,
               COutput, global_elements_in_rows);


    for (int j = 0; j < global_elements_in_cols; j++) {
        for (int i = 0; i < global_elements_in_rows; i++) {
            int idx = j * global_elements_in_rows + i;
            std::cout << " Output [" << i << "][" << j << "] = " << std::round(COutput[idx]) << " \t";
        }
        std::cout << "\n";
    }


    delete compressed_tile;
}