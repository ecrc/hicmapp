#include <hicmapp/api/Hicmapp.hpp>
#include <hcorepp/helpers/Timer.hpp>
#include <hcorepp/helpers/generators/concrete/LatmsGenerator.hpp>
#include <hcorepp/helpers/generators/concrete/TileLatmsGenerator.hpp>
#include <hcorepp/helpers/TileMatrix.hpp>
#include <hcorepp/kernels/memory.hpp>
#include <fstream>
#include "hcorepp/kernels/kernels.hpp"


#define STREAMS 16


using namespace std::chrono;
using namespace hcorepp::operators;
using namespace hcorepp::helpers;
using namespace hcorepp::kernels;


int main(int argc, char *argv[]) {
    // single tile dimensions.
    int tile_size = 512;
    double fixed_rank_decay = 2;
    // parameters needed for matrix multiplication driver to operate correctly.
    double alpha = 1;
    double beta = 1;
    blas::Op trans_a = blas::Op::NoTrans;
    blas::Op trans_b = blas::Op::NoTrans;
    // parameters for matrix generation.
    int64_t mode = 0;
    blas::real_type<double> cond = 1;
    // Target accuracy.
    std::vector<double> accuracy_list = {1e-1, 1e-4, 1e-6};
    // Assuming square matrix, default tile matrix is 2 x 2 tiles.
    int matrix_tiles = 2;
    int per_tile_generation = 0;
    int num_of_threads = -1;
    int size = 1;
    int id = 0;
#ifdef HICMAPP_USE_MPI
    int required = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &required);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    std::cout << " TOTAL num of processes = " << size << "\n";
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    std::cout << " Process Id : " << id << " \n";
#endif
    // Parse optional arguments from command line.
    if (argc > 1) {
        matrix_tiles = atoi(argv[1]);
        if (argc > 2) {
            accuracy_list.clear();
            std::string acc_str = argv[2];
            std::stringstream ss(acc_str);
            for (double i; ss >> i;) {
                accuracy_list.push_back(i);
                if (ss.peek() == ',')
                    ss.ignore();
            }
            if (argc > 3) {
                tile_size = atoi(argv[3]);
                if (argc > 4) {
                    per_tile_generation = atoi(argv[4]);
                    if (argc > 5) {
                        num_of_threads = atoi(argv[5]);
                    }
                }
            }
        }
    }
    // Check for verbosity
    bool print_header = true;
    {
        const char *val = std::getenv("HICMAPP_VERBOSE");
        if (val != nullptr) { // invalid to assign nullptr to std::string
            std::string value = val;
            if (value == "ON") {
                print_header = true;
            }
        }
    }

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
    size_t dense_flops;


    int64_t iseed[4] = {0, 0, 0, 1};

    hicmapp::api::Hicmapp<double>::Init(size, 0, num_of_threads);
    // Create full matrices with automatic generation.
    hcorepp::helpers::Timer timer;
    generators::Generator<double> *generator;
    if (per_tile_generation > 0) {
        generator = new generators::TileLatmsGenerator<double>(iseed, mode, cond, tile_size);
    } else {
        generator = new generators::LatmsGenerator<double>(iseed, mode, cond);
    }
    RawMatrix<double> full_a(a_mt * row_tile_size, a_nt * column_tile_size, *generator);
    RawMatrix<double> full_b(b_mt * row_tile_size, b_nt * column_tile_size, *generator);
    RawMatrix<double> full_c(c_mt * row_tile_size, c_nt * column_tile_size);

    delete generator;
    auto initial_c = full_c.Clone();
    timer.Snapshot("generation");
    {
        auto *warm_a = hcorepp::memory::AllocateArray<double>(full_a.GetM() * full_a.GetN(),
                                                              context.GetMainContext());
        auto *warm_b = hcorepp::memory::AllocateArray<double>(full_b.GetM() * full_b.GetN(),
                                                              context.GetMainContext());
        auto *warm_c = hcorepp::memory::AllocateArray<double>(full_c.GetM() * full_c.GetN(),
                                                              context.GetMainContext());
        hcorepp::memory::Memcpy<double>(warm_a, full_a.GetData(),
                                        full_a.GetM() * full_a.GetN(), context.GetMainContext(),
                                        hcorepp::memory::MemoryTransfer::HOST_TO_DEVICE);
        hcorepp::memory::Memcpy<double>(warm_b, full_b.GetData(), full_b.GetM() * full_b.GetN(),
                                        context.GetMainContext(),
                                        hcorepp::memory::MemoryTransfer::HOST_TO_DEVICE);
        hcorepp::memory::Memcpy<double>(warm_c, full_c.GetData(), full_c.GetM() * full_c.GetN(),
                                        context.GetMainContext(),
                                        hcorepp::memory::MemoryTransfer::HOST_TO_DEVICE);
        context.SyncMainContext();
        hcorepp::kernels::HCoreKernels<double>::Gemm(blas::Layout::ColMajor, trans_a, trans_b, full_c.GetM(),
                                                     full_c.GetN(), full_a.GetN(), alpha, warm_a,
                                                     full_a.GetM(), warm_b,
                                                     full_b.GetM(), beta, warm_c, full_c.GetM(),
                                                     context.GetMainContext());
        hcorepp::memory::DestroyArray(warm_a, context.GetMainContext());
        hcorepp::memory::DestroyArray(warm_b, context.GetMainContext());
        hcorepp::memory::DestroyArray(warm_c, context.GetMainContext());
    }
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

    size_t dense_memory_footprint;
    double dense_error;
    double dense_error_normalized;
    // Dense Warmup
    {
        Matrix<double> a_dense(full_a.GetData(), a_mt * row_tile_size, a_nt * column_tile_size, row_tile_size,
                               column_tile_size, hicmapp::common::StorageLayout::HicmaCM, context);
        Matrix<double> b_dense(full_b.GetData(), a_mt * row_tile_size, a_nt * column_tile_size, row_tile_size,
                               column_tile_size, hicmapp::common::StorageLayout::HicmaCM, context);
        Matrix<double> c_dense(initial_c.GetData(), a_mt * row_tile_size, a_nt * column_tile_size, row_tile_size,
                               column_tile_size, hicmapp::common::StorageLayout::HicmaCM, context);
        context.SyncMainContext();
        hicmapp::api::Hicmapp<double>::Gemm(a_dense, blas::Op::NoTrans, b_dense, blas::Op::NoTrans, c_dense, alpha,
                                            beta, context, true);
    }
    // Dense Flow
    {
        timer.StartSnapshot();
        // Create dense tile matrix
        Matrix<double> a_dense(full_a.GetData(), a_mt * row_tile_size, a_nt * column_tile_size, row_tile_size,
                               column_tile_size, hicmapp::common::StorageLayout::HicmaCM, context);
        Matrix<double> b_dense(full_b.GetData(), a_mt * row_tile_size, a_nt * column_tile_size, row_tile_size,
                               column_tile_size, hicmapp::common::StorageLayout::HicmaCM, context);
        Matrix<double> c_dense(initial_c.GetData(), a_mt * row_tile_size, a_nt * column_tile_size, row_tile_size,
                               column_tile_size, hicmapp::common::StorageLayout::HicmaCM, context);
        context.SyncAll();
        timer.Snapshot("dense_creation");
        // Do matrix multiplication.
        timer.StartSnapshot();
        dense_flops = hicmapp::api::Hicmapp<double>::Gemm(a_dense, blas::Op::NoTrans, b_dense, blas::Op::NoTrans,
                                                          c_dense, alpha, beta, context, true);
        context.SyncAll();
        timer.Snapshot("dense_gemm");
        // Retrieve results back from tile format for verification.
        timer.StartSnapshot();
        auto full_dense_c = c_dense.ToRawMatrix(context);
        context.SyncAll();
        full_dense_c.ReferenceDifference(full_c);

        dense_error = full_dense_c.Norm();

        dense_error_normalized = dense_error / ((a_norm + b_norm + c_init_norm) *
                                                std::numeric_limits<double>::epsilon() *
                                                std::min(initial_c.GetN(), initial_c.GetM()));
        timer.Snapshot("dense_error_calc");
        // Error checking.
        if (dense_error_normalized >= 10) {
            std::cout << "Example didn't pass, dense HCore++ error > 10 " << std::endl;
        }
        // Get memory footprint in KB
        dense_memory_footprint = (a_dense.GetMemoryFootprint() + b_dense.GetMemoryFootprint()
                                  + c_dense.GetMemoryFootprint()) / 1024;
    }
    // Compressed flow
    bool first_print = true;

    for (auto &accuracy: accuracy_list) {
        CompressionParameters svd_parameters(accuracy);
        std::vector<std::vector<size_t>> comp_ranks;
        // Compressed Warmup
        {
            Matrix<double> a_comp(full_a.GetData(), a_mt * row_tile_size,
                                  a_nt * column_tile_size, row_tile_size,
                                  column_tile_size, hicmapp::common::StorageLayout::HicmaCM, context, svd_parameters);
            Matrix<double> b_comp(full_b.GetData(), b_mt * row_tile_size,
                                  b_nt * column_tile_size, row_tile_size,
                                  column_tile_size, hicmapp::common::StorageLayout::HicmaCM, context, svd_parameters);
            Matrix<double> c_comp(initial_c.GetData(), a_mt * row_tile_size,
                                  b_nt * column_tile_size, row_tile_size,
                                  column_tile_size, hicmapp::common::StorageLayout::HicmaCM, context, svd_parameters);
            context.SyncMainContext();
            hicmapp::api::Hicmapp<double>::Gemm(a_comp, blas::Op::NoTrans, b_comp, blas::Op::NoTrans, c_comp, alpha, beta , context, true, svd_parameters);

            auto cc_nt = c_comp.GetNumOfGlobalTilesInCols();
            auto cc_mt = c_comp.GetNumOfGlobalTilesInRows();
            comp_ranks.resize(cc_mt);
            for (int j = 0; j < cc_mt; j++) {
                comp_ranks[j].resize(cc_nt);
            }
            for (int i = 0; i < cc_nt; i++) {
                for (int j = 0; j < cc_mt; j++) {
                    auto* c_tile = c_comp.GetSubMatrices()[0]->GetTiles()[j * cc_mt + i];
                    auto rank = c_tile->GetTileRank();
                    comp_ranks[j][i] = rank;
                }
            }
        }
        //Reset all compression timers
        timer.ResetSnapshot("comp_creation");
        timer.ResetSnapshot("comp_gemm");
        timer.ResetSnapshot("comp_error_calc");
        timer.StartSnapshot();
        // Create compressed tiles matrix
        Matrix<double> a_comp(full_a.GetData(), a_mt * row_tile_size,
                              a_nt * column_tile_size, row_tile_size,
                              column_tile_size, hicmapp::common::StorageLayout::HicmaCM, context, svd_parameters);
        Matrix<double> b_comp(full_b.GetData(), b_mt * row_tile_size,
                              b_nt * column_tile_size, row_tile_size,
                              column_tile_size, hicmapp::common::StorageLayout::HicmaCM, context, svd_parameters);
        {
            Matrix<double> c_comp(initial_c.GetData(), a_mt * row_tile_size,
                                  b_nt * column_tile_size, row_tile_size,
                                  column_tile_size, hicmapp::common::StorageLayout::HicmaCM, context, svd_parameters);
            context.SyncMainContext();
            timer.Snapshot("comp_creation");
            // Do matrix multiplication.
            timer.StartSnapshot();
            auto comp_flops = hicmapp::api::Hicmapp<double>::Gemm(a_comp, blas::Op::NoTrans, b_comp, blas::Op::NoTrans, c_comp, alpha, beta , context, true, svd_parameters);
            context.SyncMainContext();
            timer.Snapshot("comp_gemm");
            // Retrieve results back from tile format for verification.
            timer.StartSnapshot();
            auto full_approximate_c = c_comp.ToRawMatrix(context);
            // Calculate compressed tile matrix reference error
            full_approximate_c.ReferenceDifference(full_c);
            double comp_error = full_approximate_c.Norm();
            double comp_error_normalized = comp_error / ((a_norm + b_norm + c_init_norm) * accuracy *
                                                         std::min(initial_c.GetN(), initial_c.GetM()));
            timer.Snapshot("comp_error_calc");
            // Error checking.
            if (comp_error_normalized >= 10) {
                std::cout << "Example didn't pass, compressed HCore++ error > 10 " << std::endl;
            }
            // Get memory footprint in KB
            size_t compressed_memory_footprint = (a_comp.GetMemoryFootprint() + b_comp.GetMemoryFootprint()
                                                  + c_comp.GetMemoryFootprint()) / 1024;
            // Print results
            if (first_print) {
                if (print_header) {
                    printf("tile_count, tile_size, matrix_size, type, error, error_normalized, memory(KB), creation(ms), gemm_time(ms), flops\n");
                    print_header = false;
                }
                printf("%d, %d, %d, ref, 0, 0, %zu, %f, %f, %zu\n",
                       matrix_tiles, tile_size, matrix_tiles * tile_size,
                       ref_memory_footprint, timer.GetSnapshot("generation"),
                       timer.GetSnapshot("ref_gemm"), ref_flops);
                printf("%d, %d, %d, dense, %e, %e, %zu, %f, %f, %zu\n",
                       matrix_tiles, tile_size, matrix_tiles * tile_size, dense_error, dense_error_normalized,
                       dense_memory_footprint, timer.GetSnapshot("dense_creation"),
                       timer.GetSnapshot("dense_gemm"), dense_flops);
                first_print = false;
            }
            printf("%d, %d, %d, %2.1e, %e, %e, %zu, %f, %f, %zu\n",
                   matrix_tiles, tile_size, matrix_tiles * tile_size, accuracy, comp_error, comp_error_normalized,
                   compressed_memory_footprint, timer.GetSnapshot("comp_creation"),
                   timer.GetSnapshot("comp_gemm"), comp_flops);
        }

        timer.ResetSnapshot("comp_creation");
        timer.ResetSnapshot("comp_gemm");
        timer.ResetSnapshot("comp_error_calc");
        {
            Matrix<double> c_comp(initial_c.GetData(), a_mt * row_tile_size,
                                  b_nt * column_tile_size, row_tile_size,
                                  column_tile_size, hicmapp::common::StorageLayout::HicmaCM, context, svd_parameters);
            context.SyncMainContext();
            timer.Snapshot("comp_creation");
            // Do matrix multiplication.
            timer.StartSnapshot();
            auto comp_flops = hicmapp::api::Hicmapp<double>::Gemm(a_comp, blas::Op::NoTrans, b_comp, blas::Op::NoTrans, c_comp, alpha, beta , context, true, svd_parameters, comp_ranks);
            timer.Snapshot("comp_gemm");
            // Retrieve results back from tile format for verification.
            timer.StartSnapshot();
            auto full_approximate_c = c_comp.ToRawMatrix(context);
            // Calculate compressed tile matrix reference error
            full_approximate_c.ReferenceDifference(full_c);
            double comp_error = full_approximate_c.Norm();
            double comp_error_normalized = comp_error / ((a_norm + b_norm + c_init_norm) * accuracy *
                                                         std::min(initial_c.GetN(), initial_c.GetM()));
            timer.Snapshot("comp_error_calc");
            // Error checking.
            if (comp_error_normalized >= 10) {
                std::cout << "Example didn't pass, compressed HCore++ error > 10 " << std::endl;
            }
            // Get memory footprint in KB
            size_t compressed_memory_footprint = (a_comp.GetMemoryFootprint() + b_comp.GetMemoryFootprint()
                                                  + c_comp.GetMemoryFootprint()) / 1024;
            printf("%d, %d, %d, %2.1e-fixed-rank, %e, %e, %zu, %f, %f, %zu\n",
                   matrix_tiles, tile_size, matrix_tiles * tile_size, accuracy, comp_error, comp_error_normalized,
                   compressed_memory_footprint, timer.GetSnapshot("comp_creation"),
                   timer.GetSnapshot("comp_gemm"), comp_flops);
        }
    }

    hicmapp::api::Hicmapp<double>::Finalize();

#ifdef HICMAPP_USE_MPI
    MPI_Finalize();
#endif
    return 0;
}