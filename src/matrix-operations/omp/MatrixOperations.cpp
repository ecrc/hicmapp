#include <hicmapp/primitives/matrix.hpp>
#include <hicmapp/matrix-operations/interface/MatrixOperations.hpp>
#include <hicmapp/runtime/interface/RunTimeSingleton.hpp>
#include "hicmapp/tile-operations/TileOperations.hpp"
#include <hicmapp/utils/MatrixHelpers.hpp>
#include <fstream>
#include <hcorepp/kernels/memory.hpp>

#ifdef USE_OMP
#include <omp.h>
#include "hcorepp/kernels/kernels.hpp"

#ifdef BLAS_HAVE_MKL
#include <mkl.h>
#endif
#endif

namespace hicmapp::operations {

    template<typename T>
    int MatrixOperations<T>::GenerateCompressedMatrix(common::Uplo auplo, Matrix<T> &aMatrix,
                                                      const CompressionParameters &aSVDArguments,
                                                      bool aAsync) {

        int process_id = 0;
        int processes = 1;
#ifdef HICMAPP_USE_MPI
        MPI_Comm_rank(aMatrix.GetContext().GetCommunicator().GetMPICommunicatior(), &process_id);
        MPI_Comm_size(aMatrix.GetContext().GetCommunicator().GetMPICommunicatior(), &processes);
#endif
        size_t num_of_global_tiles_in_rows = aMatrix.GetNumOfGlobalTilesInRows();
        size_t num_of_global_tiles_in_cols = aMatrix.GetNumOfGlobalTilesInCols();

        if (!aMatrix.IsMatrixValid()) {
            throw std::runtime_error("Matrix is invalid");
        }

        if (aMatrix.GetGlobalNumOfRowsInMatrix() == 0 || aMatrix.GetGlobalNumOfColsInMatrix() == 0) {
            return 0;
        }

        auto RunTime_instance = hicmapp::runtime::RunTimeSingleton<T>::GetRunTimeInstance();
        RunTime_instance->RegisterHandles(aMatrix);

        if (aMatrix.GetStorageLayout() == common::StorageLayout::HicmaCM) {
            for (size_t col_idx = 0; col_idx < num_of_global_tiles_in_cols; col_idx++) {
                for (size_t row_idx = 0; row_idx < num_of_global_tiles_in_rows; row_idx++) {
                    if ((auplo == common::Uplo::HicmaLower && row_idx < col_idx) ||
                        (auplo == common::Uplo::HicmaUpper && row_idx > col_idx) ||
                        !aMatrix.ContainsTile(row_idx, col_idx)) {
                        continue;
                    }

                    RunTime_instance->GenerateCompressedMatrix(aMatrix, row_idx, col_idx, aSVDArguments);
                }
            }
        } else if (aMatrix.GetStorageLayout() == common::StorageLayout::HicmaRM) {
            for (size_t row_idx = 0; row_idx < num_of_global_tiles_in_rows; row_idx++) {
                for (size_t col_idx = 0; col_idx < num_of_global_tiles_in_cols; col_idx++) {
                    if ((auplo == common::Uplo::HicmaLower && row_idx < col_idx) ||
                        (auplo == common::Uplo::HicmaUpper && row_idx > col_idx) ||
                        !aMatrix.ContainsTile(row_idx, col_idx)) {
                        continue;
                    }

                    RunTime_instance->GenerateCompressedMatrix(aMatrix, row_idx, col_idx, aSVDArguments);

                }
            }
        }

        RunTime_instance->Flush(aMatrix);
        if (!aAsync) {
            RunTime_instance->Sync();
        }
        //unregister data handles
        RunTime_instance->UnRegisterHandles(aMatrix);

        return 0;
    }

    template<typename T>
    int MatrixOperations<T>::GenerateDenseMatrix(common::Uplo auplo, Matrix<T> &aMatrix, bool aASync) {
        int process_id = 0;
        int processes = 1;
#ifdef HICMAPP_USE_MPI
        MPI_Comm_rank(aMatrix.GetContext().GetCommunicator().GetMPICommunicatior(), &process_id);
        MPI_Comm_size(aMatrix.GetContext().GetCommunicator().GetMPICommunicatior(), &processes);
#endif
        size_t num_of_global_tiles_in_rows = aMatrix.GetNumOfGlobalTilesInRows();
        size_t num_of_global_tiles_in_cols = aMatrix.GetNumOfGlobalTilesInCols();

        if (!aMatrix.IsMatrixValid()) {
            throw std::runtime_error("Matrix is invalid");
        }

        if (aMatrix.GetGlobalNumOfRowsInMatrix() == 0 || aMatrix.GetGlobalNumOfColsInMatrix() == 0) {
            return 0;
        }

        auto RunTime_instance = hicmapp::runtime::RunTimeSingleton<T>::GetRunTimeInstance();
        RunTime_instance->RegisterHandles(aMatrix);

        if (aMatrix.GetStorageLayout() == common::StorageLayout::HicmaCM) {
            for (size_t col_idx = 0; col_idx < num_of_global_tiles_in_cols; col_idx++) {
                for (size_t row_idx = 0; row_idx < num_of_global_tiles_in_rows; row_idx++) {
                    if ((auplo == common::Uplo::HicmaLower && row_idx < col_idx) ||
                        (auplo == common::Uplo::HicmaUpper && row_idx > col_idx) ||
                        !aMatrix.ContainsTile(row_idx, col_idx)) {
                        continue;
                    }

                    RunTime_instance->GenerateDenseMatrix(aMatrix, row_idx, col_idx);
                }
            }
        } else if (aMatrix.GetStorageLayout() == common::StorageLayout::HicmaRM) {
            for (size_t row_idx = 0; row_idx < num_of_global_tiles_in_rows; row_idx++) {
                for (size_t col_idx = 0; col_idx < num_of_global_tiles_in_cols; col_idx++) {
                    if ((auplo == common::Uplo::HicmaLower && row_idx < col_idx) ||
                        (auplo == common::Uplo::HicmaUpper && row_idx > col_idx) ||
                        !aMatrix.ContainsTile(row_idx, col_idx)) {
                        continue;
                    }

                    RunTime_instance->GenerateDenseMatrix(aMatrix, row_idx, col_idx);

                }
            }
        }


        RunTime_instance->Flush(aMatrix);
        if (!aASync) {
            RunTime_instance->Sync();
        }
        //unregister data handles
        RunTime_instance->UnRegisterHandles(aMatrix);

        return 0;
    }

    template<typename T>
    int MatrixOperations<T>::UncompressMatrix(common::Uplo auplo, Matrix<T> &aMatrixUV, Matrix<T> &aMatrixRK,
                                              Matrix<T> &aMatrixD) {

        if (!aMatrixUV.IsMatrixValid()) {
            return -1;
        }

        if (!aMatrixRK.IsMatrixValid()) {
            return -1;
        }

        if (!aMatrixD.IsMatrixValid()) {
            return -1;
        }

        int process_id = 0;
        int processes = 1;
#ifdef HICMAPP_USE_MPI
        MPI_Comm_rank(aMatrixUV.GetContext().GetCommunicator().GetMPICommunicatior(), &process_id);
        MPI_Comm_size(aMatrixUV.GetContext().GetCommunicator().GetMPICommunicatior(), &processes);
#endif

        size_t num_of_global_tiles_in_rows = aMatrixD.GetNumOfGlobalTilesInRows();
        size_t num_of_global_tiles_in_cols = aMatrixD.GetNumOfGlobalTilesInCols();
        auto RunTime_instance = hicmapp::runtime::RunTimeSingleton<T>::GetRunTimeInstance();

        RunTime_instance->RegisterHandles(aMatrixUV);
        RunTime_instance->RegisterHandles(aMatrixD);
        RunTime_instance->RegisterHandles(aMatrixRK);

        for (size_t col_idx = 0; col_idx < num_of_global_tiles_in_cols; col_idx++) {
            for (size_t row_idx = 0; row_idx < num_of_global_tiles_in_rows; row_idx++) {

                if (auplo == common::Uplo::HicmaLower && row_idx <= col_idx) {
                    continue;
                } else if (auplo == common::Uplo::HicmaUpper && row_idx >= col_idx) {
                    continue;
                }
                if (!aMatrixUV.ContainsTile(row_idx, col_idx)
                    && !aMatrixRK.ContainsTile(row_idx, col_idx)
                    && !aMatrixD.ContainsTile(row_idx, col_idx)) {
                    continue;
                }

                RunTime_instance->Uncompress(aMatrixUV, aMatrixD, aMatrixRK, row_idx, col_idx);
            }
        }

        RunTime_instance->Flush(aMatrixUV);
        RunTime_instance->Flush(aMatrixD);
        RunTime_instance->Flush(aMatrixRK);
        RunTime_instance->Sync();
        RunTime_instance->UnRegisterHandles(aMatrixUV);
        RunTime_instance->UnRegisterHandles(aMatrixD);
        RunTime_instance->UnRegisterHandles(aMatrixRK);

        return 0;
    }

    template<typename T>
    size_t MatrixOperations<T>::Gemm(Matrix<T> &aMatrixA, const blas::Op &aAOp, Matrix<T> &aMatrixB,
                                     const blas::Op &aBOp, Matrix<T> &aMatrixC, T &aAlpha, T &aBeta,
                                     runtime::HicmaContext &aContext, const CompressionParameters &aSVDArguments,
                                     const std::vector<std::vector<size_t>> &aRanks, bool aAllocatePool) {
        size_t num_of_global_tiles_in_rows_c = aMatrixC.GetNumOfGlobalTilesInRows();
        size_t num_of_global_tiles_in_cols_c = aMatrixC.GetNumOfGlobalTilesInCols();
        size_t num_of_global_tiles_in_rows_a = aMatrixA.GetNumOfGlobalTilesInRows();
        size_t num_of_global_tiles_in_cols_a = aMatrixA.GetNumOfGlobalTilesInCols();
        hcorepp::dataunits::MemoryHandler<T> &memoryHandler = hcorepp::dataunits::MemoryHandler<T>::GetInstance();

        std::vector<size_t> sizes;

        auto RunTime_instance = hicmapp::runtime::RunTimeSingleton<T>::GetRunTimeInstance();

#ifdef USE_OMP
#ifdef BLAS_HAVE_MKL
        size_t thread_number = mkl_get_max_threads();
        mkl_set_num_threads(std::ceil(thread_number / (num_of_global_tiles_in_rows_c * num_of_global_tiles_in_cols_c)));
#endif
#ifdef BLAS_HAVE_CUBLAS
        omp_set_num_threads(num_of_global_tiles_in_rows_c * num_of_global_tiles_in_cols_c);
#endif
#endif
        size_t flops = 0;

        if (!aMatrixA.IsMatrixValid()) {
            throw std::runtime_error("Matrix A invalid");
        }
        if (!aMatrixB.IsMatrixValid()) {
            throw std::runtime_error("Matrix B invalid");
        }
        if (!aMatrixC.IsMatrixValid()) {
            throw std::runtime_error("Matrix C invalid");
        }

        if (aMatrixC.GetGlobalNumOfRowsInMatrix() == 0 || aMatrixC.GetGlobalNumOfColsInMatrix() == 0 ||
            ((aAlpha == 0.0 || aMatrixA.GetGlobalNumOfColsInMatrix() == 0) && aBeta == 1.0)) {
            return flops;
        }

#ifdef USE_OMP
#pragma omp parallel default(none) shared(num_of_threads)
        {
            num_of_threads = omp_get_num_threads();
        }

        hcorepp::helpers::DebuggingTimer::SetTimersCount(num_of_threads);
#ifdef HICMAPP_USE_TIMER
        for (size_t i = 0; i < num_of_threads; i++) {
            hcorepp::helpers::DebuggingTimer *timer = hcorepp::helpers::DebuggingTimer::GetDebuggingTimer(i);
            if (timer != nullptr) {
                timer->ResetAllSnapshots();
            }
        }
#endif
#else
        hcorepp::helpers::DebuggingTimer *timer = hcorepp::helpers::DebuggingTimer::GetDebuggingTimer();
#ifdef HICMAPP_USE_TIMER
        if(timer != nullptr) {
            timer->ResetAllSnapshots();
        }
#endif
        if (aAllocatePool) {
            sizes = CalculateGemmPoolSize(aMatrixA, aAOp, aMatrixB, aBOp, aMatrixC,
                                          aAlpha,
                                          aBeta, aContext, aSVDArguments, aRanks);
            for (size_t i = 0; i < sizes.size(); i++) {
                if (sizes[i] > 0) {
                    timer->StartSnapshot("HicmaPP::MatrixOperations::AllocatingPool");
                    memoryHandler.Initialize(sizes[i], i);
                    timer->Snapshot("HicmaPP::MatrixOperations::AllocatingPool");
                }
            }
        }

#endif

        /**
         * A = m * k
         * B = k * n
         * C = m * n
         */
        size_t idx = 0;
        size_t streams = aContext.GetNumOfContexts();
        RunTime_instance->RegisterHandles(aMatrixA);
        RunTime_instance->RegisterHandles(aMatrixB);
        RunTime_instance->RegisterHandles(aMatrixC);
        if (aMatrixC.GetStorageLayout() == common::StorageLayout::HicmaCM) {
#ifdef USE_OMP
#pragma omp parallel for collapse(2) default(none) shared(aMatrixA, aMatrixB, aMatrixC, aRanks, \
        num_of_global_tiles_in_cols_c, num_of_global_tiles_in_rows_c, num_of_global_tiles_in_rows_a, num_of_global_tiles_in_cols_a, aAlpha, aAOp, aBOp, aBeta, aSVDArguments, aContext, RunTime_instance) private(idx) reduction(+:flops)
#endif
            for (size_t col_idx_c = 0; col_idx_c < num_of_global_tiles_in_cols_c; col_idx_c++) {
                for (size_t row_idx_c = 0; row_idx_c < num_of_global_tiles_in_rows_c; row_idx_c++) {
                    CompressionParameters parameters = aSVDArguments;
                    if (!aRanks.empty()) {
                        parameters = CompressionParameters(aSVDArguments.GetAccuracy(), false, true,
                                                           false, aRanks[row_idx_c][col_idx_c]);
                    }

                    if (aAOp == blas::Op::NoTrans) {
                        for (size_t col_idx_a = 0; col_idx_a < num_of_global_tiles_in_cols_a; col_idx_a++) {
                            if (!aMatrixA.ContainsTile(row_idx_c, col_idx_a) &&
                                !aMatrixB.ContainsTile(col_idx_a, col_idx_c) &&
                                !aMatrixC.ContainsTile(row_idx_c, col_idx_c)) {
                                continue;
                            }
#ifdef USE_OMP
                            hcorepp::kernels::RunContext context = aContext.GetActiveContext().ForkChildContext();
                            hcorepp::dataunits::MemoryHandler<T> memoryHandler(context);
#else
                            if (memoryHandler.IsInitialized(idx % streams)) {
                                memoryHandler.Reset(idx % streams);
                            }
                            const hcorepp::kernels::RunContext &context = aContext.GetContext(idx % streams);
#endif

                            flops += RunTime_instance->Gemm(aAlpha, aMatrixA, row_idx_c, col_idx_a, aAOp,
                                                            aMatrixB,
                                                            col_idx_a, col_idx_c,
                                                            aBOp, aBeta, aMatrixC, row_idx_c, col_idx_c, context,
                                                            parameters, memoryHandler.GetMemoryUnit(idx % streams),
                                                            false);
                        }
                        idx++;
                    } else {
                        for (size_t row_idx_a = 0; row_idx_a < num_of_global_tiles_in_rows_a; row_idx_a++) {
                            if (!aMatrixA.ContainsTile(row_idx_c, row_idx_a) &&
                                !aMatrixB.ContainsTile(row_idx_a, col_idx_c) &&
                                !aMatrixC.ContainsTile(row_idx_c, col_idx_c)) {
                                continue;
                            }
#ifdef USE_OMP
                            hcorepp::kernels::RunContext context = aContext.GetActiveContext().ForkChildContext();
                            hcorepp::dataunits::MemoryHandler<T> memoryHandler(context);

#else
                            if (memoryHandler.IsInitialized(idx % streams)) {
                                memoryHandler.Reset(idx % streams);
                            }
                            const hcorepp::kernels::RunContext &context = aContext.GetContext(idx % streams);
#endif
                            flops += RunTime_instance->Gemm(aAlpha, aMatrixA, row_idx_c, row_idx_a, aAOp,
                                                            aMatrixB,
                                                            row_idx_a, col_idx_c,
                                                            aBOp, aBeta, aMatrixC, row_idx_c, col_idx_c, context,
                                                            parameters, memoryHandler.GetMemoryUnit(idx % streams),
                                                            false);
                        }
                        idx++;
                    }
                    RunTime_instance->Flush(aMatrixC, row_idx_c, col_idx_c);
                }
            }
        } else if (aMatrixC.GetStorageLayout() == common::StorageLayout::HicmaRM) {
#ifdef USE_OMP
#pragma omp parallel for collapse(2) default(none) shared(aMatrixA, aMatrixB, aMatrixC, aRanks, \
        num_of_global_tiles_in_cols_c, num_of_global_tiles_in_rows_c, num_of_global_tiles_in_rows_a, num_of_global_tiles_in_cols_a, aAlpha, aAOp, aBOp, aBeta, aSVDArguments, aContext, RunTime_instance, handlers) private(idx, streams) reduction(+:flops)
#endif
            for (size_t row_idx_c = 0; row_idx_c < num_of_global_tiles_in_rows_c; row_idx_c++) {
                for (size_t col_idx_c = 0; col_idx_c < num_of_global_tiles_in_cols_c; col_idx_c++) {
                    CompressionParameters parameters = aSVDArguments;
                    if (!aRanks.empty()) {
                        parameters = CompressionParameters(aSVDArguments.GetAccuracy(), false, true,
                                                           false, aRanks[row_idx_c][col_idx_c]);
                    }
                    if (aAOp == blas::Op::NoTrans) {
                        for (size_t col_idx_a = 0; col_idx_a < num_of_global_tiles_in_cols_a; col_idx_a++) {
                            if (!aMatrixA.ContainsTile(row_idx_c, col_idx_a) &&
                                !aMatrixB.ContainsTile(col_idx_a, col_idx_c) &&
                                !aMatrixC.ContainsTile(row_idx_c, col_idx_c)) {
                                continue;
                            }
#ifdef USE_OMP
                            hcorepp::kernels::RunContext context = aContext.GetActiveContext().ForkChildContext();
                            hcorepp::dataunits::MemoryHandler<T> memoryHandler(context);
#else
                            if (memoryHandler.IsInitialized(idx % streams)) {
                                memoryHandler.Reset(idx % streams);
                            }

                            const hcorepp::kernels::RunContext &context = aContext.GetContext(idx % streams);
#endif
                            flops += RunTime_instance->Gemm(aAlpha, aMatrixA, row_idx_c, col_idx_a, aAOp,
                                                            aMatrixB,
                                                            col_idx_a, col_idx_c,
                                                            aBOp, aBeta, aMatrixC, row_idx_c, col_idx_c, context,
                                                            parameters, memoryHandler.GetMemoryUnit(idx % streams));
                        }
                        idx++;
                    } else {
                        for (size_t row_idx_a = 0; row_idx_a < num_of_global_tiles_in_rows_a; row_idx_a++) {
                            if (!aMatrixA.ContainsTile(row_idx_c, row_idx_a) &&
                                !aMatrixB.ContainsTile(row_idx_a, col_idx_c) &&
                                !aMatrixC.ContainsTile(row_idx_c, col_idx_c)) {
                                continue;
                            }
#ifdef USE_OMP
                            hcorepp::kernels::RunContext context = aContext.GetActiveContext().ForkChildContext();
                            hcorepp::dataunits::MemoryHandler<T> memoryHandler(context);
#else
                            if (memoryHandler.IsInitialized(idx % streams)) {
                                memoryHandler.Reset(idx % streams);
                            }
                            const hcorepp::kernels::RunContext &context = aContext.GetContext(idx % streams);
#endif
                            flops += RunTime_instance->Gemm(aAlpha, aMatrixA, row_idx_c, row_idx_a, aAOp,
                                                            aMatrixB,
                                                            row_idx_a, col_idx_c,
                                                            aBOp, aBeta, aMatrixC, row_idx_c, col_idx_c, context,
                                                            parameters, memoryHandler.GetMemoryUnit(idx % streams),
                                                            false);
                        }
                        idx++;
                    }
                    RunTime_instance->Flush(aMatrixC, row_idx_c, col_idx_c);
                }
            }
        }

        RunTime_instance->Flush(aMatrixA);
        RunTime_instance->Flush(aMatrixB);
        RunTime_instance->Sync();
        RunTime_instance->UnRegisterHandles(aMatrixA);
        RunTime_instance->UnRegisterHandles(aMatrixB);
        RunTime_instance->UnRegisterHandles(aMatrixC);

#ifdef USE_OMP
        aContext.GetActiveContext().Sync();
#ifdef HICMAPP_USE_TIMER
        std::vector<std::string> snapshot_names = hcorepp::helpers::DebuggingTimer::GetDebuggingTimer()->GetSnapshotsNames();
        std::stringstream ss;
        std::string fixed_or_variable = aRanks.empty()? "_0" : "_fixed";
        std::string dense_or_compressed;
        if(aMatrixA.GetSubMatrix(0).GetTiles()[0]->GetNumberOfMatrices() > 1) {
            dense_or_compressed = "comp_gemm";
            ss << aSVDArguments.GetAccuracy();
        }
        else {
            dense_or_compressed = "dense_gemm";
            ss << "0";
        }
        std::ofstream time_file(dense_or_compressed + "_" + std::to_string(aMatrixC.GetGlobalNumOfColsInMatrix()) + "_" + ss.str() + fixed_or_variable +
                                + ".time");
        for (auto name: snapshot_names) {
            time_file << name << ",";
            for (size_t i = 0; i < num_of_threads; i++) {
                hcorepp::helpers::DebuggingTimer *timer = hcorepp::helpers::DebuggingTimer::GetDebuggingTimer(i);
                timer->PrintSnapshot(name, time_file);
            }
            time_file << "\n";
        }
#endif
#ifdef BLAS_HAVE_MKL
    mkl_set_num_threads(thread_number);
#endif
#else
        for (size_t i = 0; i < streams; i++) {
            aContext.SyncContext(i);
            timer->StartSnapshot("Hicmapp::MatrixOperations::DestroyingPool");
            memoryHandler.FreeMemoryUnit(i);
            timer->Snapshot("Hicmapp::MatrixOperations::DestroyingPool");
        }
#ifdef HICMAPP_USE_TIMER
        std::stringstream ss;
        std::string fixed_or_variable = aRanks.empty()? "_0" : "_fixed";
        std::string dense_or_compressed;
        if(aMatrixA.GetSubMatrix(0).GetTiles()[0]->isCompressed()) {
            dense_or_compressed = "comp_gemm";
            ss << aSVDArguments.GetAccuracy();
        }
        else {
            dense_or_compressed = "dense_gemm";
            ss << "0";
        }
        std::ofstream time_file(dense_or_compressed + "_" + std::to_string(aMatrixC.GetGlobalNumOfColsInMatrix()) + "_" + ss.str() + fixed_or_variable +
                                + ".time");

        timer->PrintAllSnapshots(time_file);
#endif
#endif
        return flops;
    }

    template<typename T>
    size_t
    MatrixOperations<T>::Cholesky(common::Uplo aUpperLower, Matrix<T> &aMatrixAUV, Matrix<T> &aMatrixADiagonal,
                                  Matrix<T> &aMatrixARK, int aRank, int aMaxRank, double aAccuracy,
                                  runtime::HicmaContext &aContext) {
        size_t flops = 0;

        if (!aMatrixAUV.IsMatrixValid()) {
            throw std::runtime_error("Matrix UV invalid");
        }
        if (!aMatrixARK.IsMatrixValid()) {
            throw std::runtime_error("Matrix RK invalid");
        }
        if (!aMatrixADiagonal.IsMatrixValid()) {
            throw std::runtime_error("Matrix Diagonal invalid");
        }
        if (aMatrixADiagonal.GetNumOfRowsInTile() != aMatrixADiagonal.GetNumOfColsInTile()) {
            throw std::runtime_error("Matrix Diagonal invalid, only squared matrices are supported..");
        }
        if (aUpperLower != common::Uplo::HicmaLower && aUpperLower != common::Uplo::HicmaUpper) {
            throw std::runtime_error("Illegal value for Upper lower used during cholesky..");
        }

        auto RunTime_instance = hicmapp::runtime::RunTimeSingleton<T>::GetRunTimeInstance();

        hcorepp::dataunits::MemoryHandler<T> &memoryHandler = hcorepp::dataunits::MemoryHandler<T>::GetInstance();

        RunTime_instance->RegisterHandles(aMatrixAUV);
        RunTime_instance->RegisterHandles(aMatrixADiagonal);
        RunTime_instance->RegisterHandles(aMatrixARK);

        /// Only hicma lower is supported..
        for (size_t k = 0; k < aMatrixADiagonal.GetNumOfGlobalTilesInRows(); k++) {
            T alpha = 1;
            auto row_idx = k;
            auto col_idx = 0;

            if (aMatrixADiagonal.ContainsTile(row_idx, col_idx)) {
                RunTime_instance->Potrf(aMatrixADiagonal, row_idx, col_idx, blas::Uplo::Lower,
                                        aContext.GetContext(), memoryHandler.GetMemoryUnit());
            }

            for (size_t m = k + 1; m < aMatrixADiagonal.GetNumOfGlobalTilesInRows(); m++) {
                if (aMatrixADiagonal.ContainsTile(row_idx, col_idx) ||
                    aMatrixAUV.ContainsTile(m, k)) {
                    RunTime_instance->Trsm(blas::Side::Left, blas::Uplo::Lower, blas::Op::NoTrans,
                                           blas::Diag::NonUnit, alpha, aMatrixADiagonal, row_idx, col_idx,
                                           aMatrixAUV, m, k, aContext.GetContext(),
                                           memoryHandler.GetMemoryUnit());
                }
            }
            RunTime_instance->Flush(aMatrixAUV, k, k);


            for (size_t n = k + 1; n < aMatrixADiagonal.GetNumOfGlobalTilesInRows(); n++) {
                alpha = -1;
                T beta = 1;

                if (aMatrixAUV.ContainsTile(n, k) || aMatrixADiagonal.ContainsTile(n, 0)) {
                    RunTime_instance->Syrk(aMatrixAUV, n, k, blas::Op::NoTrans, aMatrixADiagonal, n, 0,
                                           blas::Uplo::Lower, alpha, beta, aContext.GetContext(),
                                           memoryHandler.GetMemoryUnit());
                }

                for (size_t m = n + 1; m < aMatrixADiagonal.GetNumOfGlobalTilesInRows(); m++) {
                    if (aMatrixAUV.ContainsTile(m, k) || aMatrixAUV.ContainsTile(n, k)
                        || aMatrixAUV.ContainsTile(m, n)) {
                        auto parameters = CompressionParameters(aAccuracy);
                        RunTime_instance->Gemm(alpha, aMatrixAUV, m, k, blas::Op::NoTrans, aMatrixAUV,
                                               n, k, blas::Op::Trans, beta, aMatrixAUV, m, n,
                                               aContext.GetContext(),
                                               parameters, memoryHandler.GetMemoryUnit(), true);
                    }
                }
                RunTime_instance->Flush(aMatrixAUV, n, k);
            }
        }

        RunTime_instance->Flush(aMatrixAUV);
        RunTime_instance->Flush(aMatrixADiagonal);
        RunTime_instance->Flush(aMatrixARK);
        RunTime_instance->Sync();
        RunTime_instance->UnRegisterHandles(aMatrixAUV);
        RunTime_instance->UnRegisterHandles(aMatrixADiagonal);
        RunTime_instance->UnRegisterHandles(aMatrixARK);

        return flops;
    }

    template<typename T>
    size_t
    MatrixOperations<T>::GenerateDiagonalTiles(common::Uplo auplo, Matrix<T> &aMatrixUV, Matrix<T> &aMatrixRK,
                                               Matrix<T> &aMatrixDiag, unsigned long long int seed, int maxrank,
                                               double tol, int compress_diag, Matrix<T> &aMatrixDense,
                                               runtime::HicmaContext &aContext) {
        size_t flops = 0;

        if (!aMatrixUV.IsMatrixValid()) {
            throw std::runtime_error("Matrix UV invalid");
        }
        if (!aMatrixRK.IsMatrixValid()) {
            throw std::runtime_error("Matrix RK invalid");
        }
        if (!aMatrixDiag.IsMatrixValid()) {
            throw std::runtime_error("Matrix Diagonal invalid");
        }
        if (!aMatrixDense.IsMatrixValid()) {
            throw std::runtime_error("Matrix Dense invalid");
        }

        auto RunTime_instance = hicmapp::runtime::RunTimeSingleton<T>::GetRunTimeInstance();

        hcorepp::dataunits::MemoryHandler<T> &memoryHandler = hcorepp::dataunits::MemoryHandler<T>::GetInstance();

        RunTime_instance->RegisterHandles(aMatrixUV);
        RunTime_instance->RegisterHandles(aMatrixDiag);
        RunTime_instance->RegisterHandles(aMatrixRK);
        RunTime_instance->RegisterHandles(aMatrixDense);

        int num_of_global_tiles_in_rows = aMatrixUV.GetNumOfGlobalTilesInRows();
        int num_of_global_tiles_in_cols = aMatrixUV.GetNumOfGlobalTilesInCols();
        for (int m = 0; m < num_of_global_tiles_in_rows; m++) {

            for (int n = 0; n < num_of_global_tiles_in_cols; n++) {

                if (!aMatrixUV.ContainsTile(m, n) &&
                    !aMatrixDiag.ContainsTile(m, 0) &&
                    !aMatrixRK.ContainsTile(m, n) &&
                    !aMatrixDense.ContainsTile(m, n)) {
                    continue;
                }

                if (auplo == common::Uplo::HicmaLower && m < n) {
                    continue;
                } else if (auplo == common::Uplo::HicmaUpper && m > n) {
                    continue;
                }

                int call_diag = 0;
                int AD_icol;
                if (m == n) {
                    call_diag = 1;
                    AD_icol = 0;
                } else {
                    call_diag = 0;
                    AD_icol = n;
                }

                if (call_diag) {
                    RunTime_instance->GenerateDiagonalTile(aMatrixUV, aMatrixDiag, m, AD_icol, aMatrixRK, m, n,
                                                           seed,
                                                           maxrank, tol, compress_diag, aMatrixDense,
                                                           aContext.GetContext(), call_diag);
                } else {
                    RunTime_instance->GenerateDiagonalTile(aMatrixUV, aMatrixDiag, m, AD_icol, aMatrixRK, m, n,
                                                           seed,
                                                           maxrank, tol, compress_diag, aMatrixDense,
                                                           aContext.GetContext(), call_diag);
                }
            }
        }


        RunTime_instance->Flush(aMatrixUV);
        RunTime_instance->Flush(aMatrixDiag);
        RunTime_instance->Flush(aMatrixRK);
        RunTime_instance->Flush(aMatrixDense);

        RunTime_instance->Sync();

        //unregister data handles
        RunTime_instance->UnRegisterHandles(aMatrixUV);
        RunTime_instance->UnRegisterHandles(aMatrixDiag);
        RunTime_instance->UnRegisterHandles(aMatrixRK);
        RunTime_instance->UnRegisterHandles(aMatrixDense);

    }

    template<typename T>
    size_t
    MatrixOperations<T>::DiagVecToMat(Matrix<T> &aMatrixDiag, Matrix<T> &aMatrixDense,
                                      runtime::HicmaContext &aContext) {

        auto RunTime_instance = hicmapp::runtime::RunTimeSingleton<T>::GetRunTimeInstance();

        hcorepp::dataunits::MemoryHandler<T> &memoryHandler = hcorepp::dataunits::MemoryHandler<T>::GetInstance();

        RunTime_instance->RegisterHandles(aMatrixDense);
        RunTime_instance->RegisterHandles(aMatrixDiag);

        int num_of_global_tiles_in_rows = aMatrixDiag.GetNumOfGlobalTilesInRows();

        for (int m = 0; m < num_of_global_tiles_in_rows; m++) {
            if (aMatrixDiag.ContainsTile(m, 0)
                || aMatrixDense.ContainsTile(m, m)) {
                RunTime_instance->LaCpy(aMatrixDiag, m, 0, aMatrixDense, m, m, aContext.GetContext());
            }
        }
        RunTime_instance->Flush(aMatrixDense);
        RunTime_instance->Flush(aMatrixDiag);
        RunTime_instance->Sync();
        RunTime_instance->UnRegisterHandles(aMatrixDense);
        RunTime_instance->UnRegisterHandles(aMatrixDiag);
    }


    template<typename T>
    std::vector<size_t>
    MatrixOperations<T>::CalculateGemmPoolSize(Matrix<T> &aMatrixA, const blas::Op &aAOp, Matrix<T> &aMatrixB,
                                               const blas::Op &aBOp, Matrix<T> &aMatrixC, T &aAlpha, T &aBeta,
                                               runtime::HicmaContext &aContext,
                                               const CompressionParameters &aSVDArguments,
                                               const std::vector<std::vector<size_t>> &aRanks) {
        size_t num_of_global_tiles_in_rows_c = aMatrixC.GetNumOfGlobalTilesInRows();
        size_t num_of_global_tiles_in_cols_c = aMatrixC.GetNumOfGlobalTilesInCols();
        size_t num_of_global_tiles_in_rows_a = aMatrixA.GetNumOfGlobalTilesInRows();
        size_t num_of_global_tiles_in_cols_a = aMatrixA.GetNumOfGlobalTilesInCols();

        if (!aMatrixA.IsMatrixValid()) {
            throw std::runtime_error("Matrix A invalid");
        }
        if (!aMatrixB.IsMatrixValid()) {
            throw std::runtime_error("Matrix B invalid");
        }
        if (!aMatrixC.IsMatrixValid()) {
            throw std::runtime_error("Matrix C invalid");
        }

        if (aMatrixC.GetGlobalNumOfRowsInMatrix() == 0 || aMatrixC.GetGlobalNumOfColsInMatrix() == 0 ||
            ((aAlpha == 0.0 || aMatrixA.GetGlobalNumOfColsInMatrix() == 0) && aBeta == 1.0)) {
            return {0};
        }

        size_t streams = aContext.GetNumOfContexts();
        std::vector<size_t> pool_sizes(streams);
        size_t idx = 0;

        CompressionParameters parameters = aSVDArguments;
        if (aMatrixC.GetStorageLayout() == common::StorageLayout::HicmaCM) {
            for (size_t col_idx_c = 0; col_idx_c < num_of_global_tiles_in_cols_c; col_idx_c++) {
                for (size_t row_idx_c = 0; row_idx_c < num_of_global_tiles_in_rows_c; row_idx_c++) {
                    if (!aRanks.empty()) {
                        parameters = CompressionParameters(aSVDArguments.GetAccuracy(), false, true,
                                                           false, aRanks[row_idx_c][col_idx_c]);
                    }
                    if (!aMatrixC.ContainsTile(row_idx_c, col_idx_c)) {
                        continue;
                    }
                    Tile<T> *tile_c = aMatrixC.GetTilePointer(row_idx_c, col_idx_c);
                    if (aAOp == blas::Op::NoTrans) {
                        for (size_t col_idx_a = 0; col_idx_a < num_of_global_tiles_in_cols_a; col_idx_a++) {
                            if (!aMatrixA.ContainsTile(row_idx_c, col_idx_a) ||
                                !aMatrixB.ContainsTile(col_idx_a, col_idx_c)) {
                                continue;
                            }
                            Tile<T> *tile_a = aMatrixA.GetTilePointer(row_idx_c, col_idx_a);
                            Tile<T> *tile_b = aMatrixB.GetTilePointer(col_idx_a, col_idx_c);
                            // HCORE GEMM CALL...
                            pool_sizes[idx % streams] = std::max(pool_sizes[idx % streams],
                                                                 hcorepp::api::HCore<T>::CalculateMemoryPoolSize(
                                                                         *tile_a, *tile_b,
                                                                         *tile_c, parameters,
                                                                         aContext.GetContext(idx % streams)));
                        }
                        idx++;
                    } else {
                        for (size_t row_idx_a = 0; row_idx_a < num_of_global_tiles_in_rows_a; row_idx_a++) {
                            Tile<T> *tile_a = aMatrixA.GetTilePointer(row_idx_c, row_idx_a);
                            Tile<T> *tile_b = aMatrixB.GetTilePointer(row_idx_a, col_idx_c);
                            // HCORE GEMM CALL...
                            pool_sizes[idx % streams] = std::max(pool_sizes[idx % streams],
                                                                 hcorepp::api::HCore<T>::CalculateMemoryPoolSize(
                                                                         *tile_a, *tile_b,
                                                                         *tile_c, parameters,
                                                                         aContext.GetContext(idx % streams)));
                        }
                        idx++;
                    }
                }
            }
        } else if (aMatrixC.GetStorageLayout() == common::StorageLayout::HicmaRM) {
            for (size_t row_idx_c = 0; row_idx_c < num_of_global_tiles_in_rows_c; row_idx_c++) {
                for (size_t col_idx_c = 0; col_idx_c < num_of_global_tiles_in_cols_c; col_idx_c++) {
                    if (!aRanks.empty()) {
                        parameters = CompressionParameters(aSVDArguments.GetAccuracy(), false, true,
                                                           false, aRanks[row_idx_c][col_idx_c]);
                    }
                    Tile<T> *tile_c = aMatrixC.GetTilePointer(row_idx_c, col_idx_c);
                    if (aAOp == blas::Op::NoTrans) {
                        for (size_t col_idx_a = 0; col_idx_a < num_of_global_tiles_in_cols_a; col_idx_a++) {
                            Tile<T> *tile_a = aMatrixA.GetTilePointer(row_idx_c, col_idx_a);
                            Tile<T> *tile_b = aMatrixB.GetTilePointer(col_idx_a, col_idx_c);
                            // HCORE GEMM CALL...
                            pool_sizes[idx % streams] = std::max(pool_sizes[idx % streams],
                                                                 hcorepp::api::HCore<T>::CalculateMemoryPoolSize(
                                                                         *tile_a, *tile_b,
                                                                         *tile_c, parameters,
                                                                         aContext.GetContext(idx % streams)));
                        }
                        idx++;
                    } else {
                        for (size_t row_idx_a = 0; row_idx_a < num_of_global_tiles_in_rows_a; row_idx_a++) {
                            Tile<T> *tile_a = aMatrixA.GetTilePointer(row_idx_c, row_idx_a);
                            Tile<T> *tile_b = aMatrixB.GetTilePointer(row_idx_a, col_idx_c);
                            // HCORE GEMM CALL...
                            pool_sizes[idx % streams] = std::max(pool_sizes[idx % streams],
                                                                 hcorepp::api::HCore<T>::CalculateMemoryPoolSize(
                                                                         *tile_a, *tile_b,
                                                                         *tile_c, parameters,
                                                                         aContext.GetContext(idx % streams)));
                        }
                        idx++;
                    }
                }
            }
        }

        return pool_sizes;

    }


    HICMAPP_INSTANTIATE_CLASS(MatrixOperations)

}