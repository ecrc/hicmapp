#include <hcorepp/api/HCore.hpp>
#include <cblas.h>

extern "C" {
#include "starsh.h"
}

#include "hicmapp/tile-operations/TileOperations.hpp"
#include "hicmapp/problem-manager/StarshManager.hpp"
#include <lapacke.h>
#include "hcorepp/kernels/kernels.hpp"

namespace hicmapp::operations {

    template<typename T>
    int TileOperations<T>::GenerateCompressedMatrix(hcorepp::operators::CompressedTile<T> &aCompressedTile,
                                                    size_t aTileRowIdx, size_t aTileColIdx,
                                                    const hcorepp::operators::CompressionParameters &aSVDArguments) {

        STARSH_blrf *starsh_format = hicmapp::operations::StarsHManager::GetStarsHFormat();
        auto &context = hcorepp::kernels::ContextManager::GetInstance().GetContext();
        STARSH_cluster *RC = starsh_format->row_cluster, *CC = RC;
        void *RD = RC->data, *CD = RD;

        int num_of_rows = aCompressedTile.GetNumOfRows();
        int num_of_cols = aCompressedTile.GetNumOfCols();
        int leading_dim = aCompressedTile.GetLeadingDim();
        T *array = hcorepp::memory::AllocateArray<T>(leading_dim * num_of_cols, context);

        starsh_format->problem->kernel(num_of_rows, num_of_cols,
                                       (STARSH_int *) (RC->pivot + RC->start[aTileRowIdx]),
                                       (STARSH_int *) (CC->pivot + CC->start[aTileColIdx]),
                                       (void *) RD, (void *) CD, (void *) array, leading_dim);

        size_t rk;
        size_t maxRank = std::max(std::min(num_of_rows, num_of_cols) / MAX_RANK_RATIO, 1);
        size_t min_m_n = std::min(num_of_rows, num_of_cols);
        auto sigma = hcorepp::memory::AllocateArray<blas::real_type<T>>(min_m_n, context);
        hcorepp::dataunits::DataHolder<T> u_dataholder(num_of_rows, min_m_n, num_of_rows, nullptr, context);
        auto u = u_dataholder.GetData();
        hcorepp::dataunits::DataHolder<T> vt_dataholder(min_m_n, num_of_cols, min_m_n, nullptr, context);
        auto vt = vt_dataholder.GetData();

        hcorepp::kernels::HCoreKernels<T>::SVD(hcorepp::common::Job::SomeVec, hcorepp::common::Job::SomeVec,
                                               num_of_rows, num_of_cols, array, num_of_rows, sigma, u, num_of_rows,
                                               vt,
                                               min_m_n, aSVDArguments.GetOperationType(), nullptr, 0, 0, context);
        rk = 0;
        if (aSVDArguments.GetFixedRank()) {
            /// truncate according to fixed_rk
            rk = aSVDArguments.GetFixedRank();
            if (aSVDArguments.GetFixedRank() > min_m_n) {
                rk = min_m_n;
            }
        } else { // truncate according to accuracy
            hcorepp::kernels::HCoreKernels<T>::CalculateNewRank(rk, aSVDArguments.GetTruncatedSvd(), sigma,
                                                                min_m_n,
                                                                aSVDArguments.GetAccuracy(), context);
        }

        // Ensure at least rank is 1.
        rk = std::max(rk, 1UL);

        if (rk > maxRank) {
            rk = maxRank;
        }

        // VT eats Sigma.
        hcorepp::kernels::HCoreKernels<T>::CalculateVTnew(rk, aSVDArguments.GetUngqr(),
                                                          num_of_cols, sigma, vt, min_m_n,
                                                          vt_dataholder.GetNumOfRows(),
                                                          context);
        // Prepare UV array.
        auto auv = hcorepp::memory::AllocateArray<T>((num_of_rows + num_of_cols) * rk, context);
        hcorepp::memory::Memcpy<T>(auv, u, (num_of_rows * rk), context,
                                   hcorepp::memory::MemoryTransfer::DEVICE_TO_DEVICE);
        hcorepp::kernels::HCoreKernels<T>::LaCpy(hcorepp::common::MatrixType::General, rk, num_of_cols, vt, min_m_n,
                                                 &auv[num_of_rows * rk], rk, context);
        hcorepp::memory::DestroyArray(sigma, context);


        hcorepp::memory::Memcpy(aCompressedTile.GetUMatrix(), auv, num_of_rows * rk, context);
        hcorepp::memory::Memcpy(aCompressedTile.GetVMatrix(), &auv[num_of_rows * rk],
                                rk * num_of_cols, context);
        hcorepp::memory::DestroyArray(auv, context);


        hcorepp::operators::TileMetadata metadata(num_of_rows, num_of_cols, rk, maxRank, leading_dim,
                                                  aCompressedTile.GetLayout(),
                                                  hcorepp::operators::TileType::COMPRESSED);
        aCompressedTile.UpdateMetadata(metadata);

        return 0;
    }

    template<typename T>
    int
    TileOperations<T>::GenerateDenseTile(hcorepp::operators::DenseTile<T> &aDenseTile, size_t aTileRowIdx,
                                         size_t aTileColIdx) {

        STARSH_blrf *starsh_format = hicmapp::operations::StarsHManager::GetStarsHFormat();
        STARSH_cluster *RC = starsh_format->row_cluster, *CC = RC;
        void *RD = RC->data, *CD = RD;

        int num_of_rows = aDenseTile.GetNumOfRows();
        int num_of_cols = aDenseTile.GetNumOfCols();
        int leading_dim = aDenseTile.GetLeadingDim();
        T *array = (T *) aDenseTile.GetTileSubMatrix(0);

        starsh_format->problem->kernel(num_of_rows, num_of_cols,
                                       (STARSH_int *) (RC->pivot + RC->start[aTileRowIdx]),
                                       (STARSH_int *) (CC->pivot + CC->start[aTileColIdx]),
                                       (void *) RD, (void *) CD, (void *) array, leading_dim);

        return 0;
    }

    template<typename T>
    int
    TileOperations<T>::UnCompressTile(size_t aNumOfRows, size_t aNumOfCols, double aAlpha, const T *apAU,
                                      const T *apArk,
                                      size_t aLeadingDimA, const T *apBV, size_t aLeadingDimB,
                                      double aBeta,
                                      T *apC, size_t aLeadingDimC) {
        cblas_dgemm(
                CblasColMajor,
                CblasNoTrans, CblasTrans,
                aNumOfRows, aNumOfCols, apArk[0],
                aAlpha, (double *) apAU, aLeadingDimA,
                (double *) apBV, aLeadingDimB,
                aBeta, (double *) apC, aLeadingDimC);

        return 0;
    }

    template<typename T>
    size_t
    TileOperations<T>::Gemm(T aAlpha, const hcorepp::operators::Tile<T> &aA, const blas::Op &aAOp,
                            const hcorepp::operators::Tile<T> &aB,
                            const blas::Op &aBOp, T aBeta, hcorepp::operators::Tile<T> &aC,
                            const hcorepp::kernels::RunContext &aContext,
                            hcorepp::dataunits::MemoryUnit<T> &aMemoryUnit,
                            const hcorepp::operators::CompressionParameters &aSVDArguments, bool aCholesky) {
        size_t flops = 0;
        size_t &flops_ref = flops;
        hcorepp::api::HCore<T>::Gemm(aAlpha, aA, aAOp, aB, aBOp, aBeta, aC, aContext, flops_ref, aMemoryUnit,
                                     aSVDArguments, aCholesky);
        return flops;
    }

    template<typename T>
    size_t TileOperations<T>::Syrk(T aAlpha, const hcorepp::operators::Tile<T> &aA, const blas::Op &aAOp,
                                   const blas::Uplo aUplo, T aBeta,
                                   hcorepp::operators::Tile<T> &aC, const hcorepp::kernels::RunContext &aContext,
                                   hcorepp::dataunits::MemoryUnit<T> &aMemoryUnit) {
        size_t flops = 0;
        size_t &flops_ref = flops;

        hcorepp::api::HCore<T>::Syrk(aAlpha, aA, aAOp, aUplo, aBeta, aC, aContext, flops_ref, aMemoryUnit);

        return flops;
    }

    template<typename T>
    size_t TileOperations<T>::Potrf(hcorepp::operators::Tile<T> &aA, const blas::Uplo aUplo,
                                    const hcorepp::kernels::RunContext &aContext,
                                    hcorepp::dataunits::MemoryUnit<T> &aMemoryUnit) {
        size_t flops = 0;
        size_t &flops_ref = flops;

        hcorepp::api::HCore<T>::Potrf(aA, aUplo, aContext, flops_ref, aMemoryUnit);

        return flops;
    }

    template<typename T>
    size_t TileOperations<T>::Trsm(blas::Side aSide, blas::Uplo aUplo, blas::Op aTrans, blas::Diag aDiag, T aAlpha,
                                   hcorepp::operators::Tile<T> &aA, hcorepp::operators::Tile<T> &aB,
                                   const hcorepp::kernels::RunContext &aContext,
                                   hcorepp::dataunits::MemoryUnit<T> &aMemoryUnit) {
        size_t flops = 0;
        size_t &flops_ref = flops;

        hcorepp::api::HCore<T>::Trsm(aSide, aUplo, aTrans, aDiag, aAlpha, aA, aB, aContext, flops_ref,
                                     aMemoryUnit);

        return flops;
    }

    template<typename T>
    size_t
    TileOperations<T>::GenerateDiagonalTile(hcorepp::operators::Tile<T> *aAuv, hcorepp::operators::Tile<T> *aArk,
                                            hcorepp::operators::Tile<T> *aADense,
                                            hcorepp::operators::Tile<T> *aADiagonal,
                                            int aTileRowIdx, int aTileColIdx, unsigned long long int seed,
                                            int maxrank, double tol, int compress_diag, int lda, int ldu,
                                            int ldv, int rows, int cols,
                                            const hcorepp::kernels::RunContext &aContext) {
        size_t flops = 0;

        auto &comp_tile = (hcorepp::operators::CompressedTile<T> &) (*aAuv);

        auto *AU = comp_tile.GetUMatrix();
        auto *AV = comp_tile.GetVMatrix();
        T *ADiagonal = nullptr;
        bool A_diagonal_allocated = false;
        if (aADiagonal != nullptr) {
            ADiagonal = aADiagonal->GetTileSubMatrix(0);
        }

        auto *ARK = aArk->GetTileSubMatrix(0);
        auto *Dense = aADense->GetTileSubMatrix(0);

        int rank = 0;
        int oversample = 10;
        double *work;
        int *iwork;
        STARSH_blrf *blrf = hicmapp::operations::StarsHManager::GetStarsHFormat();
        STARSH_cluster *RC = blrf->row_cluster, *CC = RC;
        void *RD = RC->data, *CD = RD;
        T *saveAD;

        if ((aTileRowIdx != aTileColIdx) || compress_diag == 1) {
            saveAD = ADiagonal;
            if (ADiagonal == nullptr) {
                ADiagonal = (T *) malloc(sizeof(T) * lda * cols);
                A_diagonal_allocated = true;
            }
            assert(rows == lda);
        }

        blrf->problem->kernel(rows, cols, RC->pivot + RC->start[aTileRowIdx], CC->pivot + CC->start[aTileColIdx],
                              RD, CD, ADiagonal, lda);

        {
            char chall = 'A';
            dlacpy_(&chall, &rows, &cols, (double *) ADiagonal, &lda, (double *) Dense, &lda
#ifdef LAPACK_FORTRAN_STRLEN_END
                    , 0
#endif
            );
        }
        int mn = rows;
        int mn2 = maxrank + oversample;
        if (mn2 > mn)
            mn2 = mn;

        size_t lwork = cols, lwork_sdd = (4 * mn2 + 7) * mn2;
        if (lwork_sdd > lwork)
            lwork = lwork_sdd;
        lwork += (size_t)
                         mn2 * (2 * cols + rows + mn2 + 1);
        size_t liwork = 8 * mn2;

        iwork = (int *) malloc(sizeof(*iwork) * liwork);

        work = (double *) malloc(sizeof(*work) * lwork);

        if (aTileRowIdx != aTileColIdx ||
            compress_diag == 1) {

            starsh_dense_dlrrsdd(rows, cols, (double *) ADiagonal, lda, (double *) AU, ldu, (double *) AV, ldv, &rank,
                                 maxrank, oversample, tol, (double *) work, lwork, iwork);

            if (rank == -1) { //means that tile is dense.
                rank = rows;
                fprintf(stderr, "%s %s %d: Dense off-diagonal block (%d,%d). maxrank:%d\n", __FILE__, __func__,
                        __LINE__,
                        aTileRowIdx, aTileColIdx, maxrank);
                exit(0);
            }
            if (rank == 0) rank = 1;
            ARK[0] = rank;
            assert(ADiagonal != saveAD);
            if (A_diagonal_allocated) {
                free(ADiagonal);
            }
        } else {
            ARK[0] = rows;
        }

        comp_tile.ReadjustTileRank(ARK[0], aContext);

        free(work);
        free(iwork);

        return flops;
    }


    template<typename T>
    size_t TileOperations<T>::LaCpy(int aRows, int aCols, const hcorepp::operators::Tile<T> &aA,
                                    hcorepp::operators::Tile<T> &aB,
                                    const hcorepp::kernels::RunContext &aContext) {

        auto data_a = aA.GetTileSubMatrix(0);
        auto lda = aA.GetLeadingDim();
        auto data_b = aB.GetTileSubMatrix(0);
        auto ldb = aB.GetLeadingDim();
        hcorepp::kernels::HCoreKernels<T>::LaCpy(hcorepp::common::MatrixType::General, aRows, aCols, data_a, lda,
                                                 data_b,
                                                 ldb, aContext);

        return 0;
    }

    HICMAPP_INSTANTIATE_CLASS(TileOperations)
}