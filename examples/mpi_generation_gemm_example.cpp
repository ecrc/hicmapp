#include <hicmapp/api/Hicmapp.hpp>
#include <hcorepp/helpers/Timer.hpp>
#include <hcorepp/helpers/generators/concrete/LatmsGenerator.hpp>
#include <hcorepp/helpers/TileMatrix.hpp>
#include <hcorepp/kernels/memory.hpp>
#include "hcorepp/kernels/kernels.hpp"
#include "hicmapp/primitives/ProblemManager.hpp"
#include "hicmapp/problem-manager/StarshManager.hpp"
#include <hcorepp/helpers/LapackWrappers.hpp>
#include <hicmapp/primitives/decomposer/concrete/two_dimension_cyclic_decomposer.hpp>
#include <fstream>
#ifdef HICMAPP_USE_MPI
#include <mpi.h>
#endif

using namespace std::chrono;
using namespace hcorepp::operators;
using namespace hcorepp::helpers;
using namespace hcorepp::kernels;

template<typename T>
void GetTileData(T *aDataPtr, const Tile<T> *aTile, hicmapp::runtime::HicmaContext &aContext) {
    if (aTile->isDense()) {
        auto m = aTile->GetNumOfRows();
        auto n = aTile->GetNumOfCols();
        auto *data = aTile->GetTileSubMatrix(0);
        hcorepp::memory::Memcpy<T>(aDataPtr, data, m * n,
                                   aContext.GetMainContext(),
                                   hcorepp::memory::MemoryTransfer::DEVICE_TO_HOST);
        return;
    }

    auto *comp_tile = static_cast<const CompressedTile<T> *>(aTile);
    auto m = aTile->GetNumOfRows();
    auto n = aTile->GetNumOfCols();
    auto rank = aTile->GetTileRank();
    size_t num_elements = rank * m;
    T *cu = new T[num_elements];
    hcorepp::memory::Memcpy<T>(cu, comp_tile->GetUMatrix(), num_elements,
                               aContext.GetMainContext(),
                               hcorepp::memory::MemoryTransfer::DEVICE_TO_HOST);
    num_elements = n * rank;
    T *cv = new T[num_elements];
    hcorepp::memory::Memcpy<T>(cv, comp_tile->GetVMatrix(), num_elements,
                               aContext.GetMainContext(),
                               hcorepp::memory::MemoryTransfer::DEVICE_TO_HOST);
    aContext.SyncMainContext();

    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               m, n, rank, 1.0, cu,
               comp_tile->GetULeadingDim(), cv,
               comp_tile->GetVLeadingDim(), 0.0, aDataPtr, m);
    delete[] cu;
    delete[] cv;
}

double *ToDense(CompressedTile<double> *aTile, hicmapp::runtime::HicmaContext &aContext) {
    auto tile_size = aTile->GetNumOfRows() * aTile->GetTileRank() + aTile->GetTileRank() * aTile->GetNumOfCols();
    auto *dense_ptr = new double[aTile->GetNumOfRows() * aTile->GetNumOfCols()];
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               aTile->GetNumOfRows(), aTile->GetNumOfCols(), aTile->GetTileRank(), 1.0, aTile->GetUMatrix(),
               aTile->GetULeadingDim(), aTile->GetVMatrix(),
               aTile->GetVLeadingDim(), 0.0, dense_ptr, aTile->GetNumOfRows());

    return dense_ptr;

}

template<typename T>
size_t GetSubMatrixData(T **aRet, SubMatrix<T> *aSubMatrix, hicmapp::runtime::HicmaContext &aContext) {
    auto tiles_st_idx_row = aSubMatrix->GetTilesGlobalStIdxInRows();
    auto tiles_st_idx_col = aSubMatrix->GetTilesGlobalStIdxInCols();
    auto tiles_row = aSubMatrix->GetNumOfTilesinRows();
    auto tiles_col = aSubMatrix->GetNumOfTilesinCols();


    size_t data_offset = 0;
    size_t submatrix_size = 0;
    std::vector<Tile<T> *> &tiles = aSubMatrix->GetTiles();
    for (auto *submatrix_tile: tiles) {
        submatrix_size += (submatrix_tile->GetNumOfRows() *
                           submatrix_tile->GetNumOfCols());
    }

    auto *submatrix_data = new T[submatrix_size];
    memset(submatrix_data, 0, submatrix_size * sizeof(T));
    for (auto *submatrix_tile: tiles) {
        auto tile_size = submatrix_tile->GetNumOfRows() *
                         submatrix_tile->GetNumOfCols();
        GetTileData(&submatrix_data[data_offset], submatrix_tile, aContext);
        data_offset += tile_size;
        aContext.SyncAll();
    }

    *aRet = submatrix_data;
    return submatrix_size;
}


int main(int argc, char *argv[]) {
    // single tile dimensions.
    int tile_size = 512;
    // parameters needed for matrix multiplication driver to operate correctly.
    double alpha = 1;
    double beta = 1;
    blas::Op trans_a = blas::Op::NoTrans;
    blas::Op trans_b = blas::Op::NoTrans;
    // parameters for matrix generation.
    int64_t mode = 0;
    blas::real_type<double> cond = 1;
    // Target accuracy.
    double accuracy = 1e-6;
    // Assuming square matrix, default tile matrix is 2 x 2 tiles.
    int matrix_tiles = 2;
    // Parse optional arguments from command line.
    if (argc > 1) {
        matrix_tiles = atoi(argv[1]);
        if (argc > 2) {
            accuracy = atof(argv[2]);
            if (argc > 3) {
                tile_size = atoi(argv[3]);
            }
        }
    }

    ProblemManager problem_manager(hicmapp::common::ProblemType::PROBLEM_TYPE_SS);
    problem_manager.SetProblemProperty(hicmapp::primitives::ProblemProperty::HICMA_PROB_PROPERTY_N,
                                       tile_size * matrix_tiles);
    problem_manager.SetProblemProperty(hicmapp::primitives::ProblemProperty::HICMA_PROB_PROPERTY_NDIM,
                                       2);
    problem_manager.SetProblemProperty(hicmapp::primitives::ProblemProperty::HICMA_PROB_PROPERTY_BETA,
                                       0.1);
    problem_manager.SetProblemProperty(hicmapp::primitives::ProblemProperty::HICMA_PROB_PROPERTY_NU,
                                       0.5);
    problem_manager.SetProblemProperty(hicmapp::primitives::ProblemProperty::HICMA_PROB_PROPERTY_NOISE,
                                       1.e-4);
    problem_manager.SetProblemProperty(hicmapp::primitives::ProblemProperty::HICMA_PROB_PROPERTY_SYM,
                                       'S');
    problem_manager.SetProblemProperty(hicmapp::primitives::ProblemProperty::HICMA_PROB_PROPERTY_BLOCK_SIZE,
                                       (int) tile_size);

    hicmapp::operations::StarsHManager::SetStarsHFormat(problem_manager);
    hicmapp::runtime::HicmaContext context;

    int a_mt = matrix_tiles;
    int a_nt = matrix_tiles;
    int b_mt = a_nt;
    int b_nt = matrix_tiles;
    int c_mt = a_mt;
    int c_nt = b_nt;
    int row_tile_size = tile_size;
    int column_tile_size = tile_size;
    size_t ref_flops;
    size_t dense_flops = 0;

    int size = 1;
    int id = 0;

#ifdef HICMAPP_USE_MPI
    int required = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &required);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    std::cout << " Process Id : " << id << " \n";
#endif

    hicmapp::api::Hicmapp<double>::Init(size, 0, -1);

    // Create full matrices with automatic generation.
    hcorepp::helpers::Timer timer;
    Matrix<double> gen_a(nullptr, a_mt * row_tile_size, a_nt * column_tile_size, row_tile_size,
                         column_tile_size, hicmapp::common::StorageLayout::HicmaCM, context);
    Matrix<double> gen_b(nullptr, a_mt * row_tile_size, a_nt * column_tile_size, row_tile_size,
                         column_tile_size, hicmapp::common::StorageLayout::HicmaCM, context);
    Matrix<double> gen_c(nullptr, a_mt * row_tile_size, a_nt * column_tile_size, row_tile_size,
                         column_tile_size, hicmapp::common::StorageLayout::HicmaCM, context);

    timer.StartSnapshot();
    hicmapp::api::Hicmapp<double>::GenerateDenseMatrix(hicmapp::common::Uplo::HicmaUpperLower, gen_a, false);
    hicmapp::api::Hicmapp<double>::GenerateDenseMatrix(hicmapp::common::Uplo::HicmaUpperLower, gen_b, false);

    auto full_a = gen_a.ToRawMatrix(context);
    auto full_b = gen_b.ToRawMatrix(context);
    auto full_c = gen_c.ToRawMatrix(context);

    auto initial_c = full_c.Clone();
    timer.Snapshot("generation");
    // Solve reference solution
    {
        auto a_device = hcorepp::memory::AllocateArray<double>(full_a.GetM() * full_a.GetN(),
                                                               context.GetMainContext());
        auto b_device = hcorepp::memory::AllocateArray<double>(full_b.GetM() * full_b.GetN(),
                                                               context.GetMainContext());
        auto c_device = hcorepp::memory::AllocateArray<double>(full_c.GetM() * full_c.GetN(),
                                                               context.GetMainContext());
        hcorepp::memory::Memcpy<double>(a_device, full_a.GetData(),
                                        full_a.GetM() * full_a.GetN(), context.GetMainContext(),
                                        hcorepp::memory::MemoryTransfer::HOST_TO_DEVICE);
        hcorepp::memory::Memcpy<double>(b_device, full_b.GetData(), full_b.GetM() * full_b.GetN(),
                                        context.GetMainContext(),
                                        hcorepp::memory::MemoryTransfer::HOST_TO_DEVICE);
        hcorepp::memory::Memcpy<double>(c_device, full_c.GetData(), full_c.GetM() * full_c.GetN(),
                                        context.GetMainContext(),
                                        hcorepp::memory::MemoryTransfer::HOST_TO_DEVICE);
        context.SyncMainContext();
        timer.StartSnapshot();
        hcorepp::kernels::HCoreKernels<double>::Gemm(blas::Layout::ColMajor, trans_a, trans_b, full_c.GetM(),
                                                     full_c.GetN(), full_a.GetN(), alpha, a_device,
                                                     full_a.GetM(), b_device,
                                                     full_b.GetM(), beta, c_device, full_c.GetM(),
                                                     context.GetMainContext());
        context.SyncMainContext();
        timer.Snapshot("ref_gemm");
        ref_flops = 2 * full_c.GetM() * full_c.GetN() * full_a.GetN();
        hcorepp::memory::Memcpy<double>(full_c.GetData(), c_device, full_c.GetM() * full_c.GetN(),
                                        context.GetMainContext(),
                                        hcorepp::memory::MemoryTransfer::DEVICE_TO_HOST);
        context.SyncMainContext();
        hcorepp::memory::DestroyArray(a_device, context.GetMainContext());
        hcorepp::memory::DestroyArray(b_device, context.GetMainContext());
        hcorepp::memory::DestroyArray(c_device, context.GetMainContext());
    }
    // Get memory footprint in KB
    size_t ref_memory_footprint = (full_a.GetMemoryFootprint() + full_b.GetMemoryFootprint()
                                   + full_c.GetMemoryFootprint()) / 1024;
    // Norm for error calculations
    blas::real_type<double> a_norm = full_a.Norm();
    blas::real_type<double> b_norm = full_b.Norm();
    blas::real_type<double> c_init_norm = initial_c.Norm();

    size_t dense_memory_footprint = 0;
    double dense_error = 0;
    double dense_error_normalized = 0;
//        auto decomposer = TwoDimCyclicDecomposer(size, 1);
    auto decomposer = SlowestDimDecomposer(size, hicmapp::common::StorageLayout::HicmaCM);

    // Dense Warmup
    {
        Matrix<double> c_dense(nullptr, a_mt * row_tile_size, a_nt * column_tile_size, row_tile_size,
                               column_tile_size, hicmapp::common::StorageLayout::HicmaCM, decomposer, context);
        context.SyncMainContext();

        hicmapp::api::Hicmapp<double>::Gemm(gen_a, blas::Op::NoTrans, gen_b, blas::Op::NoTrans, c_dense,
                                            alpha, beta, context, false);
    }
    // Dense Flow
    {
        timer.StartSnapshot();
        // Create dense tile matrix
        Matrix<double> c_dense(nullptr, c_mt * row_tile_size, c_nt * column_tile_size, row_tile_size,
                               column_tile_size, hicmapp::common::StorageLayout::HicmaCM, decomposer, context);
        context.SyncAll();
        timer.Snapshot("dense_creation");
        // Do matrix multiplication.
        timer.StartSnapshot();
        dense_flops = hicmapp::api::Hicmapp<double>::Gemm(gen_a, blas::Op::NoTrans, gen_b, blas::Op::NoTrans,
                                                          c_dense, alpha, beta, context, true);
        context.SyncAll();
        timer.Snapshot("dense_gemm");

        // Retrieve results back from tile format for verification.
        timer.StartSnapshot();
        auto full_dense_c = c_dense.ToRawMatrix(context);

        //            if(id == 0) {
//                fs = std::ofstream("DenseRawMatrix.txt");
//                full_dense_c.Print(fs);
//                fs.close();
//                fs = std::ofstream("ReferenceMatrix.txt");
//                full_c.Print(fs);
//                fs.close();
//            }


        context.SyncAll();

        full_dense_c.ReferenceDifference(full_c);


        dense_error = full_dense_c.Norm();

        dense_error_normalized = dense_error / ((a_norm + b_norm + c_init_norm) *

                                                std::numeric_limits<double>::epsilon() *

                                                std::min(initial_c.GetN(), initial_c.GetM()));

        timer.Snapshot("dense_error_calc");

// Error checking.

        if (dense_error_normalized >= 10 && id == 0) {

            std::cout << "Example didn't pass, dense HCore++ error > 10 " << std::endl;

        }

// Get memory footprint in KB

        dense_memory_footprint = (gen_a.GetMemoryFootprint() + gen_b.GetMemoryFootprint()

                                  + c_dense.GetMemoryFootprint()) / 1024;
        // Get memory footprint in KB
        dense_memory_footprint = (gen_a.GetMemoryFootprint() + gen_b.GetMemoryFootprint()
                                  + c_dense.GetMemoryFootprint()) / 1024;
    }

    // Compressed flow
    CompressionParameters svd_parameters(accuracy);
    std::vector<std::vector<int64_t>> comp_ranks;

    Matrix<double> a_comp(nullptr, a_mt * row_tile_size,
                          a_nt * column_tile_size, row_tile_size,
                          column_tile_size, hicmapp::common::StorageLayout::HicmaCM, decomposer, context,
                          svd_parameters);
    Matrix<double> b_comp(nullptr, b_mt * row_tile_size,
                          b_nt * column_tile_size, row_tile_size,
                          column_tile_size, hicmapp::common::StorageLayout::HicmaCM, decomposer, context,
                          svd_parameters);

    hicmapp::api::Hicmapp<double>::GenerateCompressedMatrix(hicmapp::common::Uplo::HicmaUpperLower, a_comp,
                                                            svd_parameters, false);
    hicmapp::api::Hicmapp<double>::GenerateCompressedMatrix(hicmapp::common::Uplo::HicmaUpperLower, b_comp,
                                                            svd_parameters, false);
    // Compressed Warmup
    {

        Matrix<double> c_comp(nullptr, c_mt * row_tile_size,
                              c_nt * column_tile_size, row_tile_size,
                              column_tile_size, hicmapp::common::StorageLayout::HicmaCM, decomposer, context,
                              svd_parameters);
        context.SyncMainContext();
        hicmapp::api::Hicmapp<double>::Gemm(a_comp, blas::Op::NoTrans, b_comp, blas::Op::NoTrans, c_comp,
                                            alpha,
                                            beta, context, true, svd_parameters);
    }
    //Reset all compression timers
    timer.ResetSnapshot("comp_creation");
    timer.ResetSnapshot("comp_gemm");
    timer.ResetSnapshot("comp_error_calc");
    timer.StartSnapshot();
    {
        Matrix<double> c_comp(nullptr, a_mt * row_tile_size,
                              b_nt * column_tile_size, row_tile_size,
                              column_tile_size, hicmapp::common::StorageLayout::HicmaCM, decomposer,
                              context,
                              svd_parameters);
        context.SyncMainContext();

        timer.Snapshot("comp_creation");
        // Do matrix multiplication.
        timer.StartSnapshot();
        auto comp_flops = hicmapp::api::Hicmapp<double>::Gemm(a_comp, blas::Op::NoTrans, b_comp,
                                                              blas::Op::NoTrans, c_comp, alpha, beta,
                                                              context,
                                                              true, svd_parameters);

        context.SyncMainContext();
        timer.Snapshot("comp_gemm");

//                    std::ofstream fs = std::ofstream("CompressedHicmaMatrix.txt." + std::to_string(id));
//                    c_comp.Print(fs);
//                    fs.close();

        // Retrieve results back from tile format for verification.
        timer.StartSnapshot();
        auto full_approximate_c = c_comp.ToRawMatrix(context);
//                    if(id == 0 && idx == 2) {
//                        fs = std::ofstream("CompressedRawMatrix.txt");
//                        full_approximate_c.Print(fs);
//                        fs.close();
//                        fs = std::ofstream("ReferenceMatrix.txt");
//                        full_c.Print(fs);
//                        fs.close();
//                    }
//                 Retrieve results back from tile format for verification.

        timer.StartSnapshot();
        // Calculate compressed tile matrix reference error
        full_approximate_c.ReferenceDifference(full_c);
        double comp_error = full_approximate_c.Norm();
        double comp_error_normalized = comp_error / ((a_norm + b_norm + c_init_norm) * accuracy *
                                                     std::min(initial_c.GetN(), initial_c.GetM()));
        timer.Snapshot("comp_error_calc");
        // Error checking.
        if (comp_error_normalized >= 10 && id == 0) {
            std::cout << "Example didn't pass, compressed HCore++ error > 10 " << std::endl;
        }
        // Get memory footprint in KB
        size_t compressed_memory_footprint = (a_comp.GetMemoryFootprint() + b_comp.GetMemoryFootprint()
                                              + c_comp.GetMemoryFootprint()) / 1024;
        if (id == 0) {
            printf("tile_count, tile_size, matrix_size, type, error, error_normalized, memory(KB), creation(ms), gemm_time(ms), flops\n");
            printf("%d, %d, %d, ref, 0, 0, %zu, %f, %f, %zu\n",
                   matrix_tiles, tile_size, matrix_tiles * tile_size,
                   ref_memory_footprint, timer.GetSnapshot("generation"),
                   timer.GetSnapshot("ref_gemm"), ref_flops);
            printf("%d, %d, %d, dense, %e, %e, %zu, %f, %f, %zu\n",
                   matrix_tiles, tile_size, matrix_tiles * tile_size, dense_error,
                   dense_error_normalized,
                   dense_memory_footprint, timer.GetSnapshot("dense_creation"),
                   timer.GetSnapshot("dense_gemm"), dense_flops);
            printf("%d, %d, %d, %2.1e, %e, %e, %zu, %f, %f, %zu\n",
                   matrix_tiles, tile_size, matrix_tiles * tile_size, accuracy, comp_error,
                   comp_error_normalized,
                   compressed_memory_footprint, timer.GetSnapshot("comp_creation"),
                   timer.GetSnapshot("comp_gemm"), comp_flops);
        }
        context.SyncMainContext();
    }
    hicmapp::api::Hicmapp<double>::Finalize();
#ifdef HICMAPP_USE_MPI
    MPI_Finalize();
#endif
    return 0;
}