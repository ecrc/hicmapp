#include <hicmapp/runtime/concrete/starpu/starpu.hpp>
#include <hcorepp/operators/interface/TilePacker.hpp>
#include <hcorepp/operators/interface/Tile.hpp>
#include <hcorepp/operators/concrete/Compressed.hpp>
#include <hicmapp/tile-operations/TileOperations.hpp>
#include <starpu_data.h>
#include <hicmapp/runtime/concrete/starpu/factory.hpp>
#include <hicmapp/runtime/concrete/starpu/hicma_starpu.hpp>
#include <hicmapp/runtime/interface/StarpuCodelet.hpp>
#include <hicmapp/runtime/concrete/starpu/starpu-codelets/gemm-codelet.hpp>
#include <hicmapp/runtime/concrete/starpu/starpu-codelets/syrk-codelet.hpp>
#include <hicmapp/runtime/concrete/starpu/starpu-codelets/potrf-codelet.hpp>
#include <hicmapp/runtime/concrete/starpu/starpu-codelets/trsm-codelet.hpp>
#include <hicmapp/runtime/concrete/starpu/starpu-codelets/generate-dgytlr-diag-codelet.hpp>
#include <hicmapp/runtime/concrete/starpu/starpu-codelets/generate-dgytlr-codelet.hpp>
#include <hicmapp/runtime/concrete/starpu/starpu-codelets/uncompress-codelet.hpp>
#include <hicmapp/runtime/concrete/starpu/starpu-codelets/generate_codelet.hpp>
#include <hicmapp/runtime/concrete/starpu/starpu-codelets/generate_compressed_data_codelet.hpp>
#include <hicmapp/runtime/concrete/starpu/starpu-codelets/lacpy-codelet.hpp>


/**
 *  Malloc/Free of the data
 */
#ifdef STARPU_MALLOC_SIMULATION_FOLDED
#define FOLDED STARPU_MALLOC_SIMULATION_FOLDED
#else
#define FOLDED 0
#endif

namespace hicmapp::runtime {
    static size_t tag_sep = 16;

    template<typename T>
    StarPu<T>::StarPu(hicmapp::runtime::HicmaHardware &aHardware) {
        if (!starpu_is_initialized()) {
            mConf = (starpu_conf_t *) malloc(sizeof(starpu_conf_t));
            starpu_conf_init(mConf);
            mConf->ncpus = aHardware.mCPUs;
            mConf->ncuda = aHardware.mGPUs;
            mConf->nopencl = 0;
            if (mConf->ncuda > 0) {
                mConf->sched_policy_name = "dmdas";
            } else {
                /**
                 * Set scheduling to "ws"/"lws" if no cuda devices used because it
                 * behaves better on homogneneous architectures. If the user wants
                 * to use another scheduling strategy, he can set STARPU_SCHED
                 * env. var. to whatever he wants
                 */
#if (STARPU_MAJOR_VERSION > 1) || ((STARPU_MAJOR_VERSION == 1) && (STARPU_MINOR_VERSION >= 2))
                mConf->sched_policy_name = "lws";
#else
                mConf->sched_policy_name = "ws";
#endif
            }

            auto ncpus = mConf->ncpus;
            auto ncuda = mConf->ncuda;
            auto nthreads_per_worker = aHardware.mThreadsPerWorker;

            if (ncpus + ncuda >= 64) {
                ncpus = 64 - ncuda;
            }

            if (ncpus != -1 && nthreads_per_worker != -1) {
                int worker = 0;

                for (worker = 0; worker < ncpus; worker++) {
                    mConf->workers_bindid[worker] = (worker + 1) * nthreads_per_worker - 1;
                }

                for (worker = 0; worker < ncpus; worker++) {
                    mConf->workers_bindid[worker + ncuda] = worker * nthreads_per_worker;
                }

                mConf->use_explicit_workers_bindid = 1;
            }


#ifdef BLAS_HAVE_MKL
            auto envMKL = std::getenv("MKL_NUM_THREADS");
            int nmkl = 0;
            if (envMKL != nullptr) {
                std::string envStr(envMKL);
                nmkl = std::stoi(envMKL);
            }

#endif

#ifdef HAVE_STARPU_MALLOC_ON_NODE_SET_DEFAULT_FLAGS
            starpu_malloc_on_node_set_default_flags(STARPU_MAIN_RAM, STARPU_MALLOC_PINNED | STARPU_MALLOC_COUNT
#ifdef STARPU_MALLOC_SIMULATION_FOLDED
            | STARPU_MALLOC_SIMULATION_FOLDED
#endif
            );
#endif

#ifdef HICMAPP_USE_MPI
            int flag = 0;
#if !defined(CHAMELEON_SIMULATION)
            MPI_Initialized(&flag);
#endif
            int info = starpu_mpi_init_conf(nullptr, nullptr, 0, MPI_COMM_WORLD, mConf);
            if (info) {
                throw std::runtime_error("StarPu Initialization Failed");
            }
#else
            int info = starpu_init(mConf);

            if (info) {
                throw std::runtime_error("StarPu Initialization Failed");
            }

#endif
#ifdef USE_CUDA
            starpu_cublas_init();
#endif
            std::cout << "StarPu Initialized with " << aHardware.mCPUs << " CPU(s) and " << aHardware.mGPUs << " GPU(s)"
                      << std::endl;
        }

    }

    template<typename T>
    TileHandlesMap &StarPu<T>::GetMatrixHandles(size_t aMatrixId) {
        // RunTimeHandle is not found..
        if (mRunTimeHandles.find(aMatrixId) == mRunTimeHandles.end()) {
            mRunTimeHandles[aMatrixId] = TileHandlesMap{};
        }
        return mRunTimeHandles[aMatrixId];
    }

    template<typename T>
    int StarPu<T>::GenerateDenseMatrix(Matrix<T> &aMatrix, size_t aTileIdxInRows, size_t aTileIdxInCols) {

        auto codelet = (new GenerateCodelet<T>())->GetCodelet();

        void (*callback)(void *) = nullptr;
        auto tile_handles = StarPu<T>::GetTileHandles(aMatrix, aTileIdxInRows, aTileIdxInCols);

        starpu_insert_task(
                codelet,
                STARPU_VALUE, &aTileIdxInRows, sizeof(size_t),
                STARPU_VALUE, &aTileIdxInCols, sizeof(size_t),
                STARPU_PRIORITY, 0,
                STARPU_CALLBACK, callback,
                #if defined(CHAMELEON_CODELETS_HAVE_NAME)
                        STARPU_NAME, "dhagdm",
                #endif
                        STARPU_RW, tile_handles[0],
                STARPU_W, tile_handles[1],
                0);

        return 0;
    }

    template<typename T>
    void
    StarPu<T>::RegisterTileHandles(Matrix<T> &A, size_t aM, size_t aN) {

        common::StorageLayout storage_layout = A.GetStorageLayout();
        size_t handle_index = 0;
        if (storage_layout == common::StorageLayout::HicmaRM) {
            handle_index = aM * A.GetNumOfGlobalTilesInCols() + aN;
        } else if (storage_layout == common::StorageLayout::HicmaCM) {
            handle_index = aN * A.GetNumOfGlobalTilesInRows() + aM;
        }

        TileHandlesMap &handles_map = this->GetMatrixHandles(A.GetMatrixId());

        auto &tile_handles = GetTileHandles(handles_map, handle_index);

        int myrank = 0;
#ifdef HICMAPP_USE_MPI
        MPI_Comm_rank(A.GetContext().GetCommunicator().GetMPICommunicatior(), &myrank);
#endif

        int owner = A.GetTileOwnerId(aM, aN);

        int tag_offset = A.GetNumOfGlobalTilesInRows() * A.GetNumOfGlobalTilesInCols();

        if (myrank == owner) {
            auto tile = A.GetTilePointer(aM, aN);
            auto metadata_data = hcorepp::operators::TilePacker<T>::UnPackTile(*tile,
                                                                               hcorepp::kernels::ContextManager::GetInstance().GetContext());

            auto metadata = metadata_data.first;
            auto tile_data = metadata_data.second;
            if (this->mTileMetadata.count(A.GetMatrixId()) == 0) {
                this->mTileMetadata[A.GetMatrixId()] = {};
            }
            this->mTileMetadata[A.GetMatrixId()].emplace_back(aM, aN, metadata);

            auto &metadata_handle = tile_handles[0];
            int home_node = STARPU_MAIN_RAM;

            starpu_variable_data_register(&metadata_handle, home_node, (uintptr_t) (void *) metadata,
                                          sizeof(TileMetadata));

#ifdef HICMAPP_USE_MPI
            auto tag = (A.GetMatrixId() << tag_sep) | (handle_index);
            starpu_data_set_rank(metadata_handle, owner);
            starpu_data_set_tag(metadata_handle, tag);
#endif

            auto &tile_handle = tile_handles[1];

            auto &dh = tile->GetDataHolder().get();

            uint32_t leading_dim = dh.GetLeadingDim();
            uint32_t rows = dh.GetNumOfRows();
            uint32_t cols = dh.GetNumOfCols();

            starpu_matrix_data_register(&tile_handle, home_node, (uintptr_t) (void *) tile->GetTileSubMatrix(0),
                                        leading_dim, rows, cols, sizeof(T));
#ifdef HICMAPP_USE_MPI
            tag = (A.GetMatrixId() << tag_sep) | (handle_index + tag_offset);
            starpu_data_set_rank(tile_handle, owner);
            starpu_data_set_tag(tile_handle, tag);
#endif

        } else {

            auto &metadata_handle = tile_handles[0];
            int home_node = -1;
            auto metadata = A.GetTileMetadata(aM, aN);
            auto tile_rows = metadata->mNumOfRows;
            auto tile_cols = metadata->mNumOfCols;
            auto tile_layout = (blas::Layout) A.GetStorageLayout();
            auto tile_leading_dim = (tile_layout == blas::Layout::ColMajor) ? tile_rows : tile_cols;
            auto tile_type = A.GetMatrixTileType();
            if (tile_type == COMPRESSED) {
                size_t max_rank = std::max(std::min(tile_rows, tile_cols) / MAX_RANK_RATIO, 1UL);
                auto num_elements = tile_rows * max_rank + max_rank * tile_cols;
                tile_rows = tile_leading_dim = num_elements;
                tile_cols = 1;
            }


            starpu_variable_data_register(&metadata_handle, home_node, (uintptr_t) (void *) nullptr,
                                          sizeof(TileMetadata));

#ifdef HICMAPP_USE_MPI
            auto tag = (A.GetMatrixId() << tag_sep) | (handle_index);
            starpu_data_set_rank(metadata_handle, owner);
            starpu_data_set_tag(metadata_handle, tag);
#endif

            auto &tile_handle = tile_handles[1];

            /* Revise This */
            starpu_matrix_data_register(&tile_handle, home_node, (uintptr_t) nullptr, tile_leading_dim,
                                        tile_rows, tile_cols, sizeof(T));

#ifdef HICMAPP_USE_MPI
            tag = (A.GetMatrixId() << tag_sep) | (handle_index + tag_offset);
            starpu_data_set_rank(tile_handle, owner);
            starpu_data_set_tag(tile_handle, tag);
#endif
        }

    }


    template<typename T>
    TileHandles &StarPu<T>::GetTileHandles(TileHandlesMap &aHandlesMap, size_t aHandleIdx) {
        if (aHandlesMap.count(aHandleIdx) == 0) {
            aHandlesMap[aHandleIdx] = {nullptr, nullptr};
        }

        return aHandlesMap[aHandleIdx];
    }

    template<typename T>
    TileHandles &
    StarPu<T>::GetTileHandles(Matrix<T> &A, size_t aM, size_t aN) {

        common::StorageLayout storage_layout = A.GetStorageLayout();
        size_t handle_index = 0;
        if (storage_layout == common::StorageLayout::HicmaRM) {
            handle_index = aM * A.GetNumOfGlobalTilesInCols() + aN;
        } else if (storage_layout == common::StorageLayout::HicmaCM) {
            handle_index = aN * A.GetNumOfGlobalTilesInRows() + aM;
        }

        TileHandlesMap &handle_map = this->GetMatrixHandles(A.GetMatrixId());

        if (handle_map.count(handle_index) == 0) {
            handle_map[handle_index] = {nullptr, nullptr};
        }

        return handle_map[handle_index];
    }

    template<typename T>
    int StarPu<T>::Sync() {
#ifdef HICMAPP_USE_MPI
        starpu_mpi_wait_for_all(MPI_COMM_WORLD);
        starpu_mpi_cache_flush_all_data(MPI_COMM_WORLD);
        starpu_mpi_barrier(MPI_COMM_WORLD);
#else
        starpu_task_wait_for_all();
#endif
        return 0;
    }

    template<typename T>
    StarPu<T>::~StarPu() {
        if (mConf != nullptr) {
            free(mConf);
        }
#if defined(HICMAPP_USE_MPI)
        starpu_mpi_shutdown();
#endif

#ifdef USE_CUDA
        starpu_cublas_shutdown();
#else
        starpu_shutdown();
#endif
    }

    template<typename T>
    void StarPu<T>::Flush(const Matrix<T> &aMatrix, const size_t aRowIdx, const size_t aColIdx) {
        size_t handle_index = 0;
        if (aMatrix.GetStorageLayout() == common::StorageLayout::HicmaRM) {
            handle_index = aRowIdx * aMatrix.GetNumOfGlobalTilesInCols() + aColIdx;
        } else if (aMatrix.GetStorageLayout() == common::StorageLayout::HicmaCM) {
            handle_index = aColIdx * aMatrix.GetNumOfGlobalTilesInRows() + aRowIdx;
        }

        auto &handle_map = this->GetMatrixHandles(aMatrix.GetMatrixId());

        auto &tile_handles = GetTileHandles(handle_map, handle_index);
#ifdef HICMAPP_USE_MPI
        auto tag_1 = starpu_mpi_data_get_tag(tile_handles[1]);
//        starpu_mpi_r
//        starpu_tag_wait(tag_1);
#endif


        for (auto &handle: tile_handles) {
            if (handle == nullptr) {
                continue;
            }
#ifdef HICMAPP_USE_MPI
            starpu_mpi_cache_flush(MPI_COMM_WORLD, handle);
#endif
            if (aMatrix.ContainsTile(aRowIdx, aColIdx)) {
                starpu_data_acquire_cb(handle, STARPU_R, (void (*)(void *)) &starpu_data_release, handle);
            }
        }
    }

    template<typename T>
    void StarPu<T>::Flush(const Matrix<T> &apMatrix) {
        size_t num_of_global_tiles_in_rows = apMatrix.GetNumOfGlobalTilesInRows();
        size_t num_of_global_tiles_in_cols = apMatrix.GetNumOfGlobalTilesInCols();
        for (size_t i = 0; i < num_of_global_tiles_in_rows; i++) {
            for (size_t j = 0; j < num_of_global_tiles_in_cols; j++) {
                this->Flush(apMatrix, i, j);
            }
        }
    }

    template<typename T>
    void StarPu<T>::Finalize() {
#ifdef HICMAPP_USE_MPI
        starpu_mpi_cache_flush_all_data(MPI_COMM_WORLD);
#endif

        starpu_task_wait_for_all();
#ifdef HICMAPP_USE_MPI
        starpu_mpi_barrier(MPI_COMM_WORLD);
#endif

        for (auto &matrix: mRunTimeHandles) {
            for (auto &tile: matrix.second) {
                this->UnRegisterTileHandles(tile.second);
            }
        }

#ifdef HICMAPP_USE_MPI
        starpu_mpi_shutdown();
#else
        starpu_shutdown();
#endif

#ifdef USE_CUDA
        starpu_cublas_shutdown();
#endif
    }

    template<typename T>
    void StarPu<T>::UnRegisterHandles(Matrix<T> &A) {
        auto &handles_map = this->GetMatrixHandles(A.GetMatrixId());

        // Sync each metadata with the appropriate tile.
        if (mTileMetadata.count(A.GetMatrixId())) {
            for (auto &it: mTileMetadata[A.GetMatrixId()]) {
                auto metadata = std::get<2>(it);
                A.GetTilePointer(std::get<0>(it), std::get<1>(it))->UpdateMetadata(*metadata);
            }
        }

        for (auto &tile: handles_map) {
            UnRegisterTileHandles(tile.second);
        }
        mRunTimeHandles.erase(A.GetMatrixId());
    }

    template<typename T>
    void StarPu<T>::UnRegisterTileHandles(TileHandles &aHandles) {
        for (auto &handle: aHandles) {
            if (handle != nullptr) {
                starpu_data_unregister(handle);
                handle = nullptr;
            }
        }
    }

    template<typename T>
    void StarPu<T>::RegisterHandles(Matrix<T> &A) {
        size_t num_of_global_tiles_in_rows = A.GetNumOfGlobalTilesInRows();
        size_t num_of_global_tiles_in_cols = A.GetNumOfGlobalTilesInCols();

        if (A.GetStorageLayout() == hicmapp::common::StorageLayout::HicmaCM) {
            for (size_t col_idx = 0; col_idx < num_of_global_tiles_in_cols; col_idx++) {
                for (size_t row_idx = 0; row_idx < num_of_global_tiles_in_rows; row_idx++) {
                    this->RegisterTileHandles(A, row_idx, col_idx);
                }
            }
        } else if (A.GetStorageLayout() == hicmapp::common::StorageLayout::HicmaRM) {
            for (size_t row_idx = 0; row_idx < num_of_global_tiles_in_rows; row_idx++) {
                for (size_t col_idx = 0; col_idx < num_of_global_tiles_in_cols; col_idx++) {
                    this->RegisterTileHandles(A, row_idx, col_idx);
                }
            }
        }
    }

    template<typename T>
    int StarPu<T>::GenerateCompressedMatrix(Matrix<T> &apMatrix, size_t aTileIdxInRows, size_t aTileIdxInCols,
                                            const CompressionParameters &aSVDArguments) {

        auto *codelet = (new GenerateCompressedDataCodelet<T>())->GetCodelet();

        void (*callback)(void *) = nullptr;

        auto tile_handles = StarPu<T>::GetTileHandles(apMatrix, aTileIdxInRows, aTileIdxInCols);

        starpu_insert_task(
                codelet,
                STARPU_VALUE, &aTileIdxInRows, sizeof(size_t),
                STARPU_VALUE, &aTileIdxInCols, sizeof(size_t),
                STARPU_VALUE, &aSVDArguments, sizeof(CompressionParameters),
                STARPU_VALUE,
                STARPU_PRIORITY, 0,
                STARPU_CALLBACK, callback,
                #if defined(CHAMELEON_CODELETS_HAVE_NAME)
                        STARPU_NAME, "dhagcm",
                #endif
                        STARPU_RW, tile_handles[0],
                STARPU_W, tile_handles[1],
                0);


        return 0;
    }

    template<typename T>
    size_t
    StarPu<T>::Gemm(T aAlpha, Matrix<T> &aMatrixA, const size_t &aRowIdxA, const size_t &aColIdxA,
                    const blas::Op &aAOp, Matrix<T> &aMatrixB, const size_t &aRowIdxB, const size_t &aColIdxB,
                    const blas::Op &aBOp, T aBeta, Matrix<T> &aMatrixC, const size_t &aRowIdxC,
                    const size_t &aColIdxC, const hcorepp::kernels::RunContext &aContext,
                    const CompressionParameters &aSVDArguments, hcorepp::dataunits::MemoryUnit<T> &aMemoryUnit,
                    bool aCholesky) {
        auto codelet = (new GemmCodelet<T>())->GetCodelet();

        void (*callback)(void *) = nullptr;

        int tile_a_owner = aMatrixA.GetTileOwnerId(aRowIdxA, aColIdxA);
        int tile_b_owner = aMatrixB.GetTileOwnerId(aRowIdxB, aColIdxB);
        int tile_c_owner = aMatrixC.GetTileOwnerId(aRowIdxC, aColIdxC);

        int execution_rank = tile_c_owner;
        if (tile_a_owner == tile_b_owner) {
            execution_rank = tile_a_owner;
        }

        auto &tile_a_handles = StarPu<T>::GetTileHandles(aMatrixA, aRowIdxA, aColIdxA);
        auto &tile_b_handles = StarPu<T>::GetTileHandles(aMatrixB, aRowIdxB, aColIdxB);
        auto &tile_c_handles = StarPu<T>::GetTileHandles(aMatrixC, aRowIdxC, aColIdxC);


        auto tag_0 = starpu_mpi_data_get_tag(tile_c_handles[0]);
        auto tag_1 = starpu_mpi_data_get_tag(tile_c_handles[1]);

        starpu_insert_task(codelet,
                           STARPU_VALUE, &aAlpha, sizeof(T),
                           STARPU_VALUE, &aAOp, sizeof(blas::Op),
                           STARPU_VALUE, &aBOp, sizeof(blas::Op),
                           STARPU_VALUE, &aBeta, sizeof(T),
                           STARPU_VALUE, &aSVDArguments, sizeof(CompressionParameters),
                           STARPU_VALUE, &aCholesky, sizeof(bool),
                           STARPU_PRIORITY, 0,
                           STARPU_CALLBACK, callback,
                           #if defined(CHAMELEON_CODELETS_HAVE_NAME)
                                   STARPU_NAME, "dgemm",
                           #endif
                           STARPU_R, tile_a_handles[0],
                           STARPU_R, tile_a_handles[1],
                           STARPU_R, tile_b_handles[0],
                           STARPU_R, tile_b_handles[1],
                           STARPU_RW, tile_c_handles[0],
                           STARPU_RW, tile_c_handles[1],
                           STARPU_TAG, tag_1,
                           STARPU_EXECUTE_ON_NODE, execution_rank,
                           0);
        return 0;
    }

    template<typename T>
    size_t StarPu<T>::Syrk(Matrix<T> &aMatrixA, const size_t &aRowIdxA,
                           const size_t &aColIdxA, const blas::Op &aAOp, Matrix<T> &aMatrixDiag,
                           const size_t &aRowIdxC, const size_t &aColIdxC, const blas::Uplo aUplo, T aAlpha,
                           T aBeta, const hcorepp::kernels::RunContext &aContext,
                           hcorepp::dataunits::MemoryUnit<T> &aMemoryUnit) {
        size_t flops = 0;

        auto codelet = (new SyrkCodelet<T>())->GetCodelet();

        void (*callback)(void *) = nullptr;

        int execution_rank = aMatrixA.GetTileOwnerId(aRowIdxA, aColIdxA);

        auto &tile_a_handles = StarPu<T>::GetTileHandles(aMatrixA, aRowIdxA, aColIdxA);
        auto &tile_diag_handles = StarPu<T>::GetTileHandles(aMatrixDiag, aRowIdxC, aColIdxC);


        auto tag_0 = starpu_mpi_data_get_tag(tile_diag_handles[0]);
        auto tag_1 = starpu_mpi_data_get_tag(tile_diag_handles[1]);

        starpu_insert_task(codelet,
                           STARPU_VALUE, &aAlpha, sizeof(T),
                           STARPU_VALUE, &aAOp, sizeof(blas::Op),
                           STARPU_VALUE, &aUplo, sizeof(blas::Uplo),
                           STARPU_VALUE, &aBeta, sizeof(T),
                           STARPU_PRIORITY, 3,
                           STARPU_CALLBACK, callback,
                           #if defined(CHAMELEON_CODELETS_HAVE_NAME)
                                   STARPU_NAME, "syrk",
                           #endif
                                   STARPU_R, tile_a_handles[0],
                           STARPU_R, tile_a_handles[1],
                           STARPU_RW, tile_diag_handles[0],
                           STARPU_RW, tile_diag_handles[1],
                           STARPU_TAG, tag_1,
                           STARPU_EXECUTE_ON_NODE, execution_rank,
                           0);

        return flops;
    }

    template<typename T>
    size_t
    StarPu<T>::Potrf(Matrix<T> &aMatrixA, const size_t &aRowIdxA, const size_t &aColIdxA, const blas::Uplo aUplo,
                     const hcorepp::kernels::RunContext &aContext,
                     hcorepp::dataunits::MemoryUnit<T> &aMemoryUnit) {
        size_t flops = 0;

        auto codelet = (new PotrfCodelet<T>())->GetCodelet();

        void (*callback)(void *) = nullptr;

        int tile_a_owner = aMatrixA.GetTileOwnerId(aRowIdxA, aColIdxA);

        int execution_rank = tile_a_owner;

        auto tile_a_handles = StarPu<T>::GetTileHandles(aMatrixA, aRowIdxA, aColIdxA);

        auto tag_0 = starpu_mpi_data_get_tag(tile_a_handles[0]);
        auto tag_1 = starpu_mpi_data_get_tag(tile_a_handles[1]);

        starpu_insert_task(codelet,
                           STARPU_VALUE, &aUplo, sizeof(blas::Uplo),
                           STARPU_PRIORITY, 5,
                           STARPU_CALLBACK, callback,
                           #if defined(CHAMELEON_CODELETS_HAVE_NAME)
                                   STARPU_NAME, "potrf",
                           #endif
                           STARPU_RW, tile_a_handles[0],
                           STARPU_RW, tile_a_handles[1],
                           STARPU_EXECUTE_ON_NODE, execution_rank,
                           STARPU_TAG, tag_1,
                           0);
        return flops;
    }

    template<typename T>
    size_t StarPu<T>::Trsm(blas::Side aSide, blas::Uplo aUplo, blas::Op aTrans, blas::Diag aDiag, T aAlpha,
                           Matrix<T> &aMatrixADiagonal, const size_t &aRowIdxA, const size_t &aColIdxA,
                           Matrix<T> &aMatrixAUV, const size_t &aRowIdxB, const size_t &aColIdxB,
                           const hcorepp::kernels::RunContext &aContext,
                           hcorepp::dataunits::MemoryUnit<T> &aMemoryUnit) {
        size_t flops = 0;

        auto codelet = (new TrsmCodelet<T>())->GetCodelet();

        void (*callback)(void *) = nullptr;

        int tile_b_owner = aMatrixAUV.GetTileOwnerId(aRowIdxB, aColIdxB);

        int execution_rank = tile_b_owner;

        auto &tile_a_handles = StarPu<T>::GetTileHandles(aMatrixADiagonal, aRowIdxA, aColIdxA);
        auto &tile_b_handles = StarPu<T>::GetTileHandles(aMatrixAUV, aRowIdxB, aColIdxB);


        auto tag_0 = starpu_mpi_data_get_tag(tile_b_handles[0]);
        auto tag_1 = starpu_mpi_data_get_tag(tile_b_handles[1]);

        starpu_insert_task(codelet,
                           STARPU_VALUE, &aSide, sizeof(blas::Side),
                           STARPU_VALUE, &aUplo, sizeof(blas::Uplo),
                           STARPU_VALUE, &aTrans, sizeof(blas::Op),
                           STARPU_VALUE, &aDiag, sizeof(blas::Diag),
                           STARPU_VALUE, &aAlpha, sizeof(T),
                           STARPU_PRIORITY, 4,
                           STARPU_CALLBACK, callback,
                           #if defined(CHAMELEON_CODELETS_HAVE_NAME)
                                   STARPU_NAME, "trsm",
                           #endif
                                   STARPU_R, tile_a_handles[0],
                           STARPU_R, tile_a_handles[1],
                           STARPU_RW, tile_b_handles[0],
                           STARPU_RW, tile_b_handles[1],
                           STARPU_TAG, tag_1,
                           STARPU_EXECUTE_ON_NODE, execution_rank,
                           0);
        return flops;
    }

    template<typename T>
    size_t
    StarPu<T>::GenerateDiagonalTile(Matrix<T> &apMatrixUV, Matrix<T> &apMatrixDiag, const size_t &aRowIdxDiag,
                                    const size_t &aColIdxDiag, Matrix<T> &apMatrixRK, const size_t &aRowIdx,
                                    const size_t &aColIdx, unsigned long long int seed, size_t maxrank, double tol,
                                    size_t compress_diag, Matrix<T> &apMatrixDense,
                                    const hcorepp::kernels::RunContext &aContext, bool diagonal_tile) {
        size_t flops = 0;

        if (diagonal_tile) {
            auto codelet = (new GenerateDgytlrDiagonalCodelet<T>())->GetCodelet();

            void (*callback)(void *) = nullptr;

            auto tile_auv_handles = StarPu<T>::GetTileHandles(apMatrixUV, aRowIdx, aColIdx);
            auto tile_ark_handles = StarPu<T>::GetTileHandles(apMatrixRK, aRowIdx, aColIdx);
            auto tile_dense_handles = StarPu<T>::GetTileHandles(apMatrixDense, aRowIdx, aColIdx);
            auto tile_diag_handles = StarPu<T>::GetTileHandles(apMatrixDiag, aRowIdxDiag, aColIdxDiag);

            size_t rows = apMatrixDiag.GetNumOfRowsInTile();
            if (aRowIdxDiag == apMatrixDiag.GetNumOfGlobalTilesInRows() - 1) {
                rows = apMatrixDiag.GetGlobalNumOfRowsInMatrix() - aRowIdxDiag * apMatrixDiag.GetNumOfRowsInTile();
            }
            size_t cols = rows;

            size_t lda_diag = rows;// apMatrixDiag.GetTilePointer(aRowIdxDiag, 0).GetLeadingDim();
            size_t ld_uv = 0;


            auto tag_0 = starpu_mpi_data_get_tag(tile_auv_handles[0]);
            auto tag_1 = starpu_mpi_data_get_tag(tile_auv_handles[1]);

            starpu_insert_task(codelet,
                               STARPU_VALUE, &aRowIdx, sizeof(size_t),
                               STARPU_VALUE, &aColIdx, sizeof(size_t),
                               STARPU_VALUE, &seed, sizeof(unsigned long long int),
                               STARPU_VALUE, &maxrank, sizeof(size_t),
                               STARPU_VALUE, &tol, sizeof(double),
                               STARPU_VALUE, &compress_diag, sizeof(size_t),
                               STARPU_VALUE, &lda_diag, sizeof(size_t),
                               STARPU_VALUE, &ld_uv, sizeof(size_t),
                               STARPU_VALUE, &ld_uv, sizeof(size_t),
                               STARPU_VALUE, &rows, sizeof(size_t),
                               STARPU_VALUE, &cols, sizeof(size_t),
                               STARPU_PRIORITY, 0,
                               STARPU_CALLBACK, callback,
                               #if defined(CHAMELEON_CODELETS_HAVE_NAME)
                                       STARPU_NAME, "dgytlr_diag",
                               #endif
                                       STARPU_W, tile_auv_handles[0],
                               STARPU_W, tile_auv_handles[1],
                               STARPU_W, tile_ark_handles[0],
                               STARPU_W, tile_ark_handles[1],
                               STARPU_RW, tile_dense_handles[0],
                               STARPU_RW, tile_dense_handles[1],
                               STARPU_W, tile_diag_handles[0],
                               STARPU_W, tile_diag_handles[1],
                               STARPU_TAG, tag_1,
                               0);

        } else {

            auto codelet = (new GenerateDgytlrCodelet<T>())->GetCodelet();

            void (*callback)(void *) = nullptr;

            auto tile_auv_handles = StarPu<T>::GetTileHandles(apMatrixUV, aRowIdx, aColIdx);
            auto tile_ark_handles = StarPu<T>::GetTileHandles(apMatrixRK, aRowIdx, aColIdx);
            auto tile_dense_handles = StarPu<T>::GetTileHandles(apMatrixDense, aRowIdx, aColIdx);


            size_t rows = apMatrixDiag.GetNumOfRowsInTile();
            if (aRowIdxDiag == apMatrixDiag.GetNumOfGlobalTilesInRows() - 1) {
                rows = apMatrixDiag.GetGlobalNumOfRowsInMatrix() - aRowIdxDiag * apMatrixDiag.GetNumOfRowsInTile();
            }

            size_t cols = apMatrixUV.GetNumOfRowsInTile();
            if (aRowIdx == apMatrixUV.GetNumOfGlobalTilesInRows() - 1) {
                cols = apMatrixUV.GetGlobalNumOfRowsInMatrix() - aRowIdx * apMatrixUV.GetNumOfRowsInTile();
            }

            size_t lda_diag = rows;//apMatrixDiag.GetTilePointer(aRowIdxDiag, 0).GetLeadingDim();
            size_t ld_uv = cols;//apMatrixUV.GetTilePointer(aRowIdx, 0).GetLeadingDim();


            auto tag_0 = starpu_mpi_data_get_tag(tile_auv_handles[0]);
            auto tag_1 = starpu_mpi_data_get_tag(tile_auv_handles[1]);

            starpu_insert_task(codelet,
                               STARPU_VALUE, &aRowIdx, sizeof(size_t),
                               STARPU_VALUE, &aColIdx, sizeof(size_t),
                               STARPU_VALUE, &seed, sizeof(unsigned long long int),
                               STARPU_VALUE, &maxrank, sizeof(size_t),
                               STARPU_VALUE, &tol, sizeof(double),
                               STARPU_VALUE, &compress_diag, sizeof(size_t),
                               STARPU_VALUE, &lda_diag, sizeof(size_t),
                               STARPU_VALUE, &ld_uv, sizeof(size_t),
                               STARPU_VALUE, &ld_uv, sizeof(size_t),
                               STARPU_VALUE, &rows, sizeof(size_t),
                               STARPU_VALUE, &cols, sizeof(size_t),
                               STARPU_PRIORITY, 0,
                               STARPU_CALLBACK, callback,
                               #if defined(CHAMELEON_CODELETS_HAVE_NAME)
                                       STARPU_NAME, "dgytlr",
                               #endif
                                       STARPU_W, tile_auv_handles[0],
                               STARPU_W, tile_auv_handles[1],
                               STARPU_W, tile_ark_handles[0],
                               STARPU_W, tile_ark_handles[1],
                               STARPU_RW, tile_dense_handles[0],
                               STARPU_RW, tile_dense_handles[1],
                               STARPU_TAG, tag_1,
                               0);
        }

        return flops;
    }

    template<typename T>
    size_t StarPu<T>::LaCpy(Matrix<T> &aMatrixA, const size_t &aRowIdxA, const size_t &aColIdxA,
                            Matrix<T> &aMatrixB, const size_t &aRowIdxB, const size_t &aColIdxB,
                            const hcorepp::kernels::RunContext &aContext) {

        size_t flops = 0;

        auto codelet = (new LacpyCodelet<T>())->GetCodelet();

        void (*callback)(void *) = nullptr;

        int tile_destination_owner = aMatrixB.GetTileOwnerId(aRowIdxB, aColIdxB);

        int execution_rank = tile_destination_owner;

        auto &tile_src_handles = StarPu<T>::GetTileHandles(aMatrixA, aRowIdxA, aColIdxA);
        auto &tile_dest_handles = StarPu<T>::GetTileHandles(aMatrixB, aRowIdxB, aColIdxB);

        starpu_insert_task(codelet,
                           STARPU_PRIORITY, 0,
                           STARPU_CALLBACK, callback,
                           #if defined(CHAMELEON_CODELETS_HAVE_NAME)
                                   STARPU_NAME, "lacpy",
                           #endif
                           STARPU_R, tile_src_handles[0],
                           STARPU_R, tile_src_handles[1],
                           STARPU_RW, tile_dest_handles[0],
                           STARPU_RW, tile_dest_handles[1],
                           STARPU_EXECUTE_ON_NODE, execution_rank,
                           0);
        return flops;
    }

    template<typename T>
    size_t
    StarPu<T>::Uncompress(Matrix<T> &apMatrixUV, Matrix<T> &apMatrixDense, Matrix<T> &apMatrixRk, const size_t &aRowIdx,
                          const size_t &aColIdx) {
        size_t flops = 0;

        auto codelet = (new UncompressCodelet<T>())->GetCodelet();

        void (*callback)(void *) = nullptr;

        int tile_dense_owner = apMatrixDense.GetTileOwnerId(aRowIdx, aColIdx);

        int execution_rank = tile_dense_owner;

        auto tile_uv_handles = StarPu<T>::GetTileHandles(apMatrixUV, aRowIdx, aColIdx);
        auto tile_dense_handles = StarPu<T>::GetTileHandles(apMatrixDense, aRowIdx, aColIdx);
        auto tile_rk_handles = StarPu<T>::GetTileHandles(apMatrixRk, aRowIdx, aColIdx);
        hcorepp::common::BlasOperation atrans = hcorepp::common::BlasOperation::OP_NoTRANS;
        hcorepp::common::BlasOperation btrans = hcorepp::common::BlasOperation::OP_CONJG;
        size_t ncols = apMatrixUV.GetNumOfRowsInTile();

        starpu_insert_task(codelet,
                           STARPU_VALUE, &atrans, sizeof(hcorepp::common::BlasOperation),
                           STARPU_VALUE, &btrans, sizeof(hcorepp::common::BlasOperation),
                           STARPU_VALUE, &ncols, sizeof(size_t),
                           STARPU_PRIORITY, 0,
                           STARPU_CALLBACK, callback,
                           #if defined(CHAMELEON_CODELETS_HAVE_NAME)
                                   STARPU_NAME, "uncompress",
                           #endif
                           STARPU_R, tile_uv_handles[0],
                           STARPU_R, tile_uv_handles[1],
                           STARPU_RW, tile_dense_handles[0],
                           STARPU_RW, tile_dense_handles[1],
                           STARPU_R, tile_rk_handles[0],
                           STARPU_R, tile_rk_handles[1],
                           STARPU_EXECUTE_ON_NODE, execution_rank,
                           0);

        return flops;

    }

    HICMAPP_INSTANTIATE_CLASS(StarPu)
}