#include <iostream>
#include <hicmapp/api/Hicmapp.hpp>
#include <hicmapp/runtime/interface/RunTimeFactory.hpp>
#include <hicmapp/primitives/matrix.hpp>
#include <hicmapp/primitives/decomposer/concrete/slowest_dimension_decomposer.hpp>
#include <hicmapp/primitives/decomposer/concrete/two_dimension_cyclic_decomposer.hpp>
#include "hicmapp/problem-manager/StarshManager.hpp"
#include <hicmapp/primitives/ProblemManager.hpp>
#include <cblas.h>
#include "hicmapp/utils/MatrixHelpers.hpp"

using namespace hicmapp::primitives;

int main(int argc, char *argv[]) {
    double accuracy = 1e-9;
    double fixed_rank_decay = 2;

    int global_elements_in_rows = 4;
    int global_elements_in_cols = 4;
    size_t tile_rows = 4;
    size_t tile_cols = 4;

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
    /** Empty matrices generation.. */
    size_t num_of_processes_in_rows = 1;
    size_t num_of_processes_in_cols = 1;
    TwoDimCyclicDecomposer decomposer(num_of_processes_in_rows, num_of_processes_in_cols);

    Matrix<double> matrixA(nullptr, global_elements_in_rows,
                           global_elements_in_cols, tile_rows, tile_cols,
                           hicmapp::common::StorageLayout::HicmaCM, decomposer, context);
    Matrix<double> matrixB(nullptr, global_elements_in_rows,
                           global_elements_in_cols, tile_rows, tile_cols,
                           hicmapp::common::StorageLayout::HicmaCM, decomposer, context);
    Matrix<double> matrixC(nullptr, global_elements_in_rows,
                           global_elements_in_cols, tile_rows, tile_cols,
                           hicmapp::common::StorageLayout::HicmaCM, decomposer, context);

    int max_rank = 4;

    Matrix<double> matrixAUV(nullptr, global_elements_in_rows,
                             global_elements_in_cols, tile_rows, tile_cols,
                             hicmapp::common::StorageLayout::HicmaCM, decomposer, context, max_rank);

    Matrix<double> matrixBUV(nullptr, global_elements_in_rows,
                             global_elements_in_cols, tile_rows, tile_cols,
                             hicmapp::common::StorageLayout::HicmaCM, decomposer, context, max_rank);
    Matrix<double> matrixCUV(nullptr, global_elements_in_rows,
                             global_elements_in_cols, tile_rows, tile_cols,
                             hicmapp::common::StorageLayout::HicmaCM, decomposer, context, max_rank);

    int rk_tile_num_of_rows = 1;
    int rk_tile_num_of_cols = 1;
    int rk_num_of_rows = matrixAUV.GetNumOfGlobalTilesInRows();//4;
    int rk_num_of_cols = matrixAUV.GetNumOfGlobalTilesInCols();//4;

    Matrix<double> matrixArk(nullptr, rk_num_of_rows, rk_num_of_cols,
                             rk_tile_num_of_rows, rk_tile_num_of_cols,
                             hicmapp::common::StorageLayout::HicmaCM, decomposer, context);
    Matrix<double> matrixBrk(nullptr, rk_num_of_rows, rk_num_of_cols,
                             rk_tile_num_of_rows, rk_tile_num_of_cols,
                             hicmapp::common::StorageLayout::HicmaCM, decomposer, context);
    Matrix<double> matrixCrk(nullptr, rk_num_of_rows, rk_num_of_cols,
                             rk_tile_num_of_rows, rk_tile_num_of_cols,
                             hicmapp::common::StorageLayout::HicmaCM, decomposer, context);


    /** Generate Random StarsH problem */
    ProblemManager problem_manager(hicmapp::common::ProblemType::PROBLEM_TYPE_RND);
    problem_manager.SetProblemProperty(hicmapp::primitives::HICMA_PROB_PROPERTY_NDIM, 2);
    problem_manager.SetProblemProperty(hicmapp::primitives::HICMA_PROB_PROPERTY_NOISE, 0.0);
    problem_manager.SetProblemProperty(hicmapp::primitives::HICMA_PROB_PROPERTY_DECAY, fixed_rank_decay);
    problem_manager.SetProblemProperty(hicmapp::primitives::HICMA_PROB_PROPERTY_N, global_elements_in_rows);
    problem_manager.SetProblemProperty(hicmapp::primitives::HICMA_PROB_PROPERTY_BLOCK_SIZE, (int)tile_rows);
    problem_manager.SetProblemProperty(hicmapp::primitives::HICMA_PROB_PROPERTY_MT, rk_num_of_rows);
    problem_manager.SetProblemProperty(hicmapp::primitives::HICMA_PROB_PROPERTY_NT, rk_num_of_rows);
    problem_manager.SetProblemProperty(hicmapp::primitives::HICMA_PROB_PROPERTY_SYM, 'S');

    hicmapp::operations::StarsHManager::SetStarsHFormat(problem_manager);
    hicmapp::api::Hicmapp<double>::Init();

    /** Generate Dense Matrices*/
    hicmapp::api::Hicmapp<double>::GenerateDenseMatrix(hicmapp::common::Uplo::HicmaUpperLower, matrixA, false);
    hicmapp::api::Hicmapp<double>::GenerateDenseMatrix(hicmapp::common::Uplo::HicmaUpperLower, matrixB, false);
    hicmapp::api::Hicmapp<double>::GenerateDenseMatrix(hicmapp::common::Uplo::HicmaUpperLower, matrixC, false);


    /** Generate Compressed Matrices*/
    hicmapp::api::Hicmapp<double>::GenerateCompressedMatrix(hicmapp::common::Uplo::HicmaUpperLower, matrixAUV,
                                                            accuracy, false);


    hicmapp::api::Hicmapp<double>::GenerateCompressedMatrix(hicmapp::common::Uplo::HicmaUpperLower, matrixBUV,
                                                            accuracy, false);
    hicmapp::api::Hicmapp<double>::GenerateCompressedMatrix(hicmapp::common::Uplo::HicmaUpperLower, matrixCUV,
                                                            accuracy, false);

    {
        std::cout << "========================= PRINTING AUV INPUT ============================== \n";

        auto *AUVOutput = new double[global_elements_in_rows * global_elements_in_rows];
        double *tile_u_data = matrixAUV.GetSubMatrix(0).GetTilePointer(0, 0)->GetTileSubMatrix(0);
        auto rows_u = matrixAUV.GetSubMatrix(0).GetTilePointer(0, 0)->GetNumOfRows();
        auto cols_u = matrixAUV.GetSubMatrix(0).GetTilePointer(0, 0)->GetTileRank();
        double *tile_v_data = matrixAUV.GetSubMatrix(0).GetTilePointer(0, 0)->GetTileSubMatrix(1);
        auto rows_v = matrixAUV.GetSubMatrix(0).GetTilePointer(0, 0)->GetTileRank();
        auto cols_v = matrixAUV.GetSubMatrix(0).GetTilePointer(0, 0)->GetNumOfCols();

        cblas_dgemm(
                CblasColMajor,
                CblasNoTrans, CblasNoTrans,
                (int) rows_u, (int) cols_v, (int) cols_u,
                1, (double *) tile_u_data, (int) rows_u,
                (double *) tile_v_data, (int) rows_v,
                0, (double *) AUVOutput, (int) rows_u);

        hicmapp::utils::MatrixHelpers<double>::PrintArray(AUVOutput, global_elements_in_rows, global_elements_in_rows,
                                                          hicmapp::common::StorageLayout::HicmaCM);
        delete[]AUVOutput;
    }
    {
        std::cout << "========================= PRINTING BUV INPUT ============================== \n";
        auto *BUVOutput = new double[global_elements_in_rows * global_elements_in_rows];

        double *tile_u_data = matrixBUV.GetSubMatrix(0).GetTilePointer(0, 0)->GetTileSubMatrix(0);
        auto rows_u = matrixBUV.GetSubMatrix(0).GetTilePointer(0, 0)->GetNumOfRows();
        auto cols_u = matrixBUV.GetSubMatrix(0).GetTilePointer(0, 0)->GetTileRank();
        double *tile_v_data = matrixBUV.GetSubMatrix(0).GetTilePointer(0, 0)->GetTileSubMatrix(1);
        auto rows_v = matrixBUV.GetSubMatrix(0).GetTilePointer(0, 0)->GetTileRank();
        auto cols_v = matrixBUV.GetSubMatrix(0).GetTilePointer(0, 0)->GetNumOfCols();

        cblas_dgemm(
                CblasColMajor,
                CblasNoTrans, CblasNoTrans,
                (int) rows_u, (int) cols_v, (int) rows_v,
                1, (double *) tile_u_data, (int) rows_u,
                (double *) tile_v_data, (int) rows_v,
                0, (double *) BUVOutput, (int) rows_u);

        hicmapp::utils::MatrixHelpers<double>::PrintArray(BUVOutput, global_elements_in_rows, global_elements_in_rows,
                                                          hicmapp::common::StorageLayout::HicmaCM);

        delete[]BUVOutput;
    }

    double alpha = 1;
    double beta = 0;
    CompressionParameters aSVDArguments = {accuracy};
    {

        hicmapp::api::Hicmapp<double>::Gemm(matrixAUV, blas::Op::NoTrans, matrixBUV, blas::Op::NoTrans,
                                            matrixCUV, alpha, beta, context, false, aSVDArguments);

        double *tile_u_data = matrixCUV.GetSubMatrix(0).GetTilePointer(0, 0)->GetTileSubMatrix(0);
        auto rows_u = matrixCUV.GetSubMatrix(0).GetTilePointer(0, 0)->GetNumOfRows();
        auto cols_u = matrixCUV.GetSubMatrix(0).GetTilePointer(0, 0)->GetTileRank();
        double *tile_v_data = matrixCUV.GetSubMatrix(0).GetTilePointer(0, 0)->GetTileSubMatrix(1);
        auto rows_v = matrixCUV.GetSubMatrix(0).GetTilePointer(0, 0)->GetTileRank();
        auto cols_v = matrixCUV.GetSubMatrix(0).GetTilePointer(0, 0)->GetNumOfCols();

        auto *cOutput = new double[global_elements_in_rows * global_elements_in_rows];
        cblas_dgemm(
                CblasColMajor,
                CblasNoTrans, CblasNoTrans,
                (int) rows_u, (int) cols_v, (int) rows_v,
                alpha, (double *) tile_u_data, (int) rows_u,
                (double *) tile_v_data, (int) rows_v,
                beta, (double *) cOutput, (int) rows_u);


        std::cout << "========================= PRINTING CUV OUTPUT ============================== \n";
        hicmapp::utils::MatrixHelpers<double>::PrintArray(cOutput, global_elements_in_rows, global_elements_in_rows,
                                                          hicmapp::common::StorageLayout::HicmaCM);

        delete[]cOutput;
    }

    hicmapp::api::Hicmapp<double>::Finalize();


#ifdef HICMAPP_USE_MPI
    MPI_Finalize();
#endif

}