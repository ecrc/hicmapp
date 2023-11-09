#include "hicmapp/problem-manager/StarshManager.hpp"
#include <hicmapp/api/Hicmapp.hpp>
#include <fstream>
#include <lapacke_utils.h>
#include <mkl_cblas.h>
#include "hcorepp/api/HCore.hpp"
#include "hcorepp/kernels/kernels.hpp"
#include "hcorepp/helpers/Timer.hpp"

int main(int argc, char *argv[]) {
    // single tile dimensions.
    int tile_size = 512;
    // Target accuracy.
    double accuracy = 1e-6;
    // Assuming square matrix, default tile matrix is 2 x 2 tiles.
    int matrix_tiles = 2;
    // The number of threads to run with.
    int thread_number = 1;
    // Parse optional arguments from command line.
    if (argc > 1) {
        matrix_tiles = atoi(argv[1]);
        if (argc > 2) {
            accuracy = atof(argv[2]);
            if (argc > 3) {
                tile_size = atoi(argv[3]);
                if (argc > 4) {
                    thread_number = atoi(argv[4]);
                }
            }
        }
    }
    std::cout << "Running with Accuracy = " << accuracy;
    std::cout << ", Tile Size = " << tile_size;
    std::cout << ", Tiles Per Row = " << matrix_tiles
              << " and Total Matrix = " << tile_size * matrix_tiles << std::endl;
    int starsh_decay = 2;
    size_t global_elements_in_rows = matrix_tiles * tile_size;
    int size = 1;
    int id = 0;
    hicmapp::runtime::HicmaContext context;

#ifdef HICMAPP_USE_MPI
    int required = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &required);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    std::cout << " Process Id : " << id << " \n";
#endif

    hicmapp::api::Hicmapp<double>::Init(thread_number, 0, -1);

    /** Generate Random StarsH problem */
    ProblemManager problem_manager(hicmapp::common::ProblemType::PROBLEM_TYPE_SS);
    problem_manager.SetProblemProperty(hicmapp::primitives::HICMA_PROB_PROPERTY_NDIM, 2);
    problem_manager.SetProblemProperty(hicmapp::primitives::HICMA_PROB_PROPERTY_BETA, 0.1);
    problem_manager.SetProblemProperty(hicmapp::primitives::HICMA_PROB_PROPERTY_NU, 0.5);
    problem_manager.SetProblemProperty(hicmapp::primitives::HICMA_PROB_PROPERTY_NOISE, 1e-2);
    problem_manager.SetProblemProperty(hicmapp::primitives::HICMA_PROB_PROPERTY_SYM, 'S');
    problem_manager.SetProblemProperty(hicmapp::primitives::HICMA_PROB_PROPERTY_DECAY, starsh_decay);
    problem_manager.SetProblemProperty(hicmapp::primitives::HICMA_PROB_PROPERTY_N, tile_size * matrix_tiles);
    problem_manager.SetProblemProperty(hicmapp::primitives::HICMA_PROB_PROPERTY_BLOCK_SIZE, tile_size);
    problem_manager.SetProblemProperty(hicmapp::primitives::HICMA_PROB_PROPERTY_MT, matrix_tiles);
    problem_manager.SetProblemProperty(hicmapp::primitives::HICMA_PROB_PROPERTY_NT, matrix_tiles);

    hicmapp::operations::StarsHManager::SetStarsHFormat(problem_manager);

    int max_rank = tile_size / MAX_RANK_RATIO;

    CompressionParameters svd_parameters(accuracy);

    int a_mt = matrix_tiles;
    int a_nt = matrix_tiles;
    int b_mt = a_nt;
    int b_nt = matrix_tiles;
    int row_tile_size = tile_size;
    int column_tile_size = tile_size;

    hcorepp::helpers::Timer timer;

    Matrix<double> matrix_Diag(nullptr, a_mt * row_tile_size,
                               column_tile_size, row_tile_size,
                               column_tile_size,
                               hicmapp::common::StorageLayout::HicmaCM, context,
                               0, true);
    Matrix<double> matrix_dense(nullptr, a_mt * row_tile_size,
                                a_nt * column_tile_size, row_tile_size,
                                column_tile_size, hicmapp::common::StorageLayout::HicmaCM,
                                context);
    Matrix<double> matrix_AUV(nullptr, a_mt * row_tile_size,
                              b_nt * column_tile_size, row_tile_size,
                              column_tile_size, hicmapp::common::StorageLayout::HicmaCM,
                              context, svd_parameters);

    Matrix<double> matrix_RK(nullptr, b_mt, b_nt, 1, 1, hicmapp::common::StorageLayout::HicmaCM, context);

    timer.StartSnapshot();
    hicmapp::api::Hicmapp<double>::GenerateDiagonalTiles(hicmapp::common::Uplo::HicmaLower, matrix_AUV, matrix_RK,
                                                         matrix_Diag, 0, max_rank, accuracy, 0, matrix_dense,
                                                         context);
    timer.Snapshot("Generation");

    auto Adense = matrix_dense.ToRawMatrix(context);

    auto *swork = (double *) calloc(2 * global_elements_in_rows, sizeof(double));

    auto raw_matrix = matrix_dense.ToRawMatrix(context);
    Matrix<double> matrix_dense2(raw_matrix.GetData(), a_mt * row_tile_size, a_nt * column_tile_size, row_tile_size,
                                 column_tile_size, hicmapp::common::StorageLayout::HicmaCM, context);

    int fixed_rank = 0;
    timer.StartSnapshot();
    hicmapp::api::Hicmapp<double>::Cholesky(hicmapp::common::Uplo::HicmaLower, matrix_AUV, matrix_Diag, matrix_RK,
                                            fixed_rank, max_rank, accuracy, context);

    timer.Snapshot("Cholesky");

    {
        /// checking accuracy...
        auto Adense2 = matrix_dense.ToRawMatrix(context);
        for (size_t j = 0; j < Adense2.GetM(); j++) {
            for (size_t i = 0; i < j; i++) {
                Adense2.GetData()[j * Adense2.GetM() + i] = 0.0;
            }
        }
        auto normA = Adense2.Normmest(swork);
        hicmapp::api::Hicmapp<double>::UncompressMatrix(hicmapp::common::Uplo::HicmaLower, matrix_AUV, matrix_RK,
                                                        matrix_dense2);
        hicmapp::api::Hicmapp<double>::DiagVecToMat(matrix_Diag, matrix_dense2, context);
        auto AhicmaT = matrix_dense2.ToRawMatrix(context);
        auto Ahicma = matrix_dense2.ToRawMatrix(context);

        double normAhicma = 0;
        {
            size_t i, j;
            for (j = 0; j < Ahicma.GetM(); j++) {
                for (i = 0; i < j; i++) {
                    Ahicma.GetData()[j * Ahicma.GetM() + i] = 0.0;
                }
            }

            hcorepp::helpers::RawMatrix<double> orgAhicma_raw_matrix = Ahicma.Clone();
            normAhicma = orgAhicma_raw_matrix.Normmest(swork);
        }

        {
            size_t i, j;
            for (j = 0; j < Adense.GetM(); j++) {
                for (i = 0; i < j; i++) {
                    Adense.GetData()[j * Adense.GetM() + i] = 0.0;
                }
            }
        }

        LAPACKE_dge_trans(LAPACK_COL_MAJOR, (lapack_int) Ahicma.GetM(), (lapack_int) Ahicma.GetM(),
                          (const double *) Ahicma.GetData(), (lapack_int) Ahicma.GetM(),
                          (double *) AhicmaT.GetData(), (lapack_int) AhicmaT.GetM());

        blas::trmm(blas::Layout::ColMajor, blas::Side::Left, blas::Uplo::Lower, blas::Op::NoTrans, blas::Diag::NonUnit,
                   AhicmaT.GetM(), AhicmaT.GetN(), 1.0, Ahicma.GetData(), Ahicma.GetM(), AhicmaT.GetData(),
                   Ahicma.GetM());

        {
            size_t i, j;
            for (j = 0; j < AhicmaT.GetM(); j++) {
                for (i = 0; i < j; i++) {
                    AhicmaT.GetData()[j * AhicmaT.GetM() + i] = 0.0;
                }
            }
        }

        size_t nelm = AhicmaT.GetM() * AhicmaT.GetN();

        cblas_daxpy(nelm, -1.0, AhicmaT.GetData(), 1, Adense.GetData(), 1);

        auto normDenseAppDiff = Adense.Normmest(swork);
        double accuracyDenseAppDiff = normDenseAppDiff / normA;
        printf("\n\nnormA:%.2e normDenseAppdiff:%.2e Accuracy: %.2e\n", normA, normDenseAppDiff, accuracyDenseAppDiff);
        std::cout << "Generation Time " << timer.GetSnapshot("Generation") << " ms" << std::endl;
        std::cout << "Cholesky Time " << timer.GetSnapshot("Cholesky") << " ms" << std::endl;

    }

    hicmapp::api::Hicmapp<double>::Finalize();

#ifdef HICMAPP_USE_MPI
    MPI_Finalize();
#endif
    return 0;

}