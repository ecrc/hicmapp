#include <iostream>
#include <hicmapp/api/Hicmapp.hpp>
#include <hicmapp/runtime/interface/RunTimeFactory.hpp>
#include <hicmapp/primitives/matrix.hpp>
#include <hicmapp/primitives/decomposer/concrete/slowest_dimension_decomposer.hpp>
#include <hicmapp/primitives/decomposer/concrete/two_dimension_cyclic_decomposer.hpp>
#include "hicmapp/problem-manager/StarshManager.hpp"
#include <hicmapp/utils/MatrixHelpers.hpp>
#include <lapacke.h>
#include <cstring>


using namespace hicmapp::primitives;

int main(int argc, char *argv[]) {
    int global_elements_in_rows = 1000;
    int global_elements_in_cols = 1000;

    size_t num_of_sub_matrices = 1;
    size_t tile_rows = 250;
    int max_rank = 250;
    size_t tile_cols = max_rank * 2;
    size_t block_size = tile_rows * tile_cols;
    int diagonal = 4;
    global_elements_in_cols = diagonal * max_rank * 2;
    auto sub_matrix_rows = global_elements_in_rows;
    auto sub_matrix_cols = global_elements_in_cols;

    size_t num_of_processes_in_rows = 1;
    size_t num_of_processes_in_cols = 1;
    double accuracy;
    hicmapp::runtime::HicmaCommunicator communicator;
#ifdef HICMAPP_USE_MPI
    MPI_Comm comm = MPI_COMM_WORLD;
    int id;
    MPI_Init(&argc, &argv);
    int size;
    MPI_Comm_size(comm, &size);
    std::cout << " TOTAL num of processes = " << size << "\n";
    MPI_Comm_rank(comm, &id);
    std::cout << " Process Id : " << id << " \n";
    communicator.SetMPICommunicator(MPI_COMM_WORLD);
#endif

    hicmapp::runtime::HicmaContext context(communicator);

    TwoDimCyclicDecomposer decomposer(num_of_processes_in_rows, num_of_processes_in_cols);

    Matrix<double> matrixDense(nullptr, global_elements_in_rows,
                               global_elements_in_rows, tile_rows, tile_rows,
                               hicmapp::common::StorageLayout::HicmaCM, decomposer, context);

    Matrix<double> matrixAUV(nullptr, global_elements_in_rows,
                             global_elements_in_cols, tile_rows, tile_cols,
                             hicmapp::common::StorageLayout::HicmaCM, decomposer, context, max_rank);

    double *ArkArray = new double[diagonal * diagonal];

    memset(ArkArray, 0, diagonal * diagonal * sizeof(double));
    LAPACKE_dlaset(LAPACK_COL_MAJOR, 'L', diagonal, diagonal, 0.0, tile_rows,
                   ArkArray, diagonal);
    Matrix<double> matrixARK(ArkArray, diagonal,
                             diagonal, 1, 1,
                             hicmapp::common::StorageLayout::HicmaCM, decomposer, context);
    ProblemManager problem_manager(hicmapp::common::ProblemType::PROBLEM_TYPE_SS);
    problem_manager.SetProblemProperty(hicmapp::primitives::ProblemProperty::HICMA_PROB_PROPERTY_N,
                                       global_elements_in_rows);
    problem_manager.SetProblemProperty(hicmapp::primitives::ProblemProperty::HICMA_PROB_PROPERTY_NDIM,
                                       2);
    problem_manager.SetProblemProperty(hicmapp::primitives::ProblemProperty::HICMA_PROB_PROPERTY_BETA,
                                       0.1);
    problem_manager.SetProblemProperty(hicmapp::primitives::ProblemProperty::HICMA_PROB_PROPERTY_NU,
                                       0.5);
    problem_manager.SetProblemProperty(hicmapp::primitives::ProblemProperty::HICMA_PROB_PROPERTY_NOISE,
                                       1.e-2);
    problem_manager.SetProblemProperty(hicmapp::primitives::ProblemProperty::HICMA_PROB_PROPERTY_WAVE_K,
                                       20);
    problem_manager.SetProblemProperty(hicmapp::primitives::ProblemProperty::HICMA_PROB_PROPERTY_DIAG,
                                       global_elements_in_rows);
    problem_manager.SetProblemProperty(hicmapp::primitives::ProblemProperty::HICMA_PROB_PROPERTY_SYM,
                                       'S');
    problem_manager.SetProblemProperty(hicmapp::primitives::ProblemProperty::HICMA_PROB_PROPERTY_BLOCK_SIZE,
                                       (int)tile_rows);

    hicmapp::operations::StarsHManager::SetStarsHFormat(problem_manager);

    hicmapp::api::Hicmapp<double>::Init();

    hicmapp::api::Hicmapp<double>::GenerateDenseMatrix(hicmapp::common::Uplo::HicmaLower, matrixDense, false);

    accuracy = 1e-7;
    hicmapp::api::Hicmapp<double>::GenerateCompressedMatrix(hicmapp::common::Uplo::HicmaLower, matrixAUV,
                                                            accuracy, false);

    hicmapp::api::Hicmapp<double>::UncompressMatrix(hicmapp::common::Uplo::HicmaLower, matrixAUV, matrixARK,
                                                    matrixDense);

    double *array = new double[diagonal * diagonal];

    hicmapp::utils::MatrixHelpers<double>::MatrixToArray(matrixARK, array);

    hicmapp::utils::MatrixHelpers<double>::PrintArray(array, diagonal, diagonal,
                                                      hicmapp::common::StorageLayout::HicmaCM);

    double *adense = new double[global_elements_in_rows * global_elements_in_rows];

    hicmapp::utils::MatrixHelpers<double>::MatrixToArray(matrixDense, adense);

    hicmapp::utils::MatrixHelpers<double>::PrintArray(adense, global_elements_in_rows, global_elements_in_rows,
                                                      hicmapp::common::StorageLayout::HicmaCM);

    hicmapp::api::Hicmapp<double>::Finalize();

    delete[]array;
    delete[]ArkArray;
    delete[]adense;

#ifdef HICMAPP_USE_MPI
    MPI_Finalize();
#endif

}