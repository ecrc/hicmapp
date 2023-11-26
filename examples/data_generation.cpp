#include <iostream>
#include <hicmapp/api/Hicmapp.hpp>
#include <hicmapp/runtime/interface/RunTimeFactory.hpp>
#include <hicmapp/primitives/matrix.hpp>
#include <hicmapp/primitives/decomposer/concrete/slowest_dimension_decomposer.hpp>
#include <hicmapp/primitives/decomposer/concrete/two_dimension_cyclic_decomposer.hpp>
#include "hicmapp/problem-manager/StarshManager.hpp"
#include <hicmapp/utils/MatrixHelpers.hpp>
#include <hicmapp/primitives/ProblemManager.hpp>


const int global_elements_in_rows = 10;
const int global_elements_in_cols = 10;

using namespace hicmapp::primitives;

int main(int argc, char *argv[]) {

    size_t tile_rows = 2;
    size_t tile_cols = 2;
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
    communicator.SetMPICommunicator(comm);
#endif


    hicmapp::runtime::HicmaContext context(communicator);
    size_t num_of_processes_in_rows = 1;
    size_t num_of_processes_in_cols = 1;
    TwoDimCyclicDecomposer decomposer(num_of_processes_in_rows, num_of_processes_in_cols);

    Matrix<double> matrix(nullptr, global_elements_in_rows,
                          global_elements_in_cols, tile_rows, tile_cols,
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

    hicmapp::api::Hicmapp<double>::GenerateDenseMatrix(hicmapp::common::Uplo::HicmaLower, matrix, false);

    double *array = new double[global_elements_in_rows * global_elements_in_cols];

    hicmapp::utils::MatrixHelpers<double>::MatrixToArray(matrix, array);

    hicmapp::utils::MatrixHelpers<double>::PrintArray(array, global_elements_in_rows, global_elements_in_rows,
                                                      hicmapp::common::StorageLayout::HicmaCM);

    hicmapp::api::Hicmapp<double>::Finalize();
    delete[]array;

#ifdef HICMAPP_USE_MPI
    MPI_Finalize();
#endif

}