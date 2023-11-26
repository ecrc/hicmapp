
#include <hicmapp/primitives/matrix.hpp>
#include <atomic>
#include <hcorepp/kernels/memory.hpp>
#include "hcorepp/operators/interface/TilePacker.hpp"
#include "hicmapp/runtime/interface/RunTimeSingleton.hpp"

namespace hicmapp::primitives {

    size_t GenerateMatrixId() {
        static std::atomic_size_t matrix_id = 0;
        const size_t id = matrix_id;
        matrix_id++;
        return id;
    }

    template<typename T>
    Matrix<T>::Matrix(T *apMatrixData, size_t aTotalGlobalNumOfRows, size_t aTotalGlobalNumOfCols,
                      size_t aTileNumOfRows, size_t aTileNumOfCols, common::StorageLayout aStorageLayout,
                      runtime::HicmaContext &aContext, size_t aRank, bool aDiagonalMatrix) : mStorageLayout{
            aStorageLayout}, mContext{aContext}, mDiagonalMatrix(aDiagonalMatrix) {

        auto world_size = 1;
#ifdef HICMAPP_USE_MPI
        MPI_Comm_size(aContext.GetCommunicator().GetMPICommunicatior(), &world_size);
#endif
        SlowestDimDecomposer decomposer(world_size, aStorageLayout);
        mDecomposerType = SLOWESTDIM;

        Initialize(apMatrixData, aTotalGlobalNumOfRows, aTotalGlobalNumOfCols, aTileNumOfRows, aTileNumOfCols,
                   aStorageLayout, decomposer, aContext, aRank);
    }

    template<typename T>
    Matrix<T>::Matrix(T *apMatrixData, size_t aTotalGlobalNumOfRows, size_t aTotalGlobalNumOfCols,
                      size_t aTileNumOfRows, size_t aTileNumOfCols, common::StorageLayout aStorageLayout,
                      MatrixDecomposer &aMatrixDecomposer, runtime::HicmaContext &aContext, size_t aRank,
                      bool aDiagonalMatrix): mStorageLayout{aStorageLayout}, mContext{aContext},
                                             mDiagonalMatrix(aDiagonalMatrix) {

        mDecomposerType = aMatrixDecomposer.GetType();
        Initialize(apMatrixData, aTotalGlobalNumOfRows, aTotalGlobalNumOfCols, aTileNumOfRows, aTileNumOfCols,
                   aStorageLayout, aMatrixDecomposer, aContext, aRank);
    }

    template<typename T>
    Matrix<T>::Matrix(T *apMatrixData, size_t aTotalGlobalNumOfRows, size_t aTotalGlobalNumOfCols,
                      size_t aTileNumOfRows, size_t aTileNumOfCols, common::StorageLayout aStorageLayout,
                      MatrixDecomposer &aMatrixDecomposer, runtime::HicmaContext &aContext,
                      const CompressionParameters &aParams) : mStorageLayout{
            aStorageLayout}, mContext{aContext} {

        mDecomposerType = aMatrixDecomposer.GetType();
        mDiagonalMatrix = false;
        Initialize(apMatrixData, aTotalGlobalNumOfRows, aTotalGlobalNumOfCols, aTileNumOfRows, aTileNumOfCols,
                   aStorageLayout, aMatrixDecomposer, aContext, aParams);
    }

    template<typename T>
    Matrix<T>::Matrix(T *apMatrixData, size_t aTotalGlobalNumOfRows, size_t aTotalGlobalNumOfCols,
                      size_t aTileNumOfRows, size_t aTileNumOfCols, common::StorageLayout aStorageLayout,
                      runtime::HicmaContext &aContext, const CompressionParameters &aParams) : mStorageLayout{
            aStorageLayout}, mContext{aContext} {
        mDiagonalMatrix = false;
        auto world_size = 1;
#ifdef HICMAPP_USE_MPI
        MPI_Comm_size(aContext.GetCommunicator().GetMPICommunicatior(), &world_size);
#endif
        SlowestDimDecomposer decomposer(world_size, aStorageLayout);
        mDecomposerType = SLOWESTDIM;
        Initialize(apMatrixData, aTotalGlobalNumOfRows, aTotalGlobalNumOfCols, aTileNumOfRows, aTileNumOfCols,
                   aStorageLayout, decomposer, aContext, aParams);
    }

    template<typename T>
    Matrix<T>::~Matrix() {
        for (auto sub_matrix: mSubMatrices) {
            delete sub_matrix;
        }
        mSubMatrices.clear();
    }

    template<typename T>
    size_t Matrix<T>::GetMatrixId() const {
        return mMatrixId;
    }

    template<typename T>
    size_t Matrix<T>::GetNumOfSubMatrices() const {
        return mSubMatrices.size();
    }

    template<typename T>
    size_t Matrix<T>::GetTotalNumOfSubMatrices() const {
        return mSpecs.size();
    }

    template<typename T>
    std::vector<SubMatrix<T> *> &
    Matrix<T>::GetSubMatrices() {
        return mSubMatrices;
    }

    template<typename T>
    SubMatrix<T> &Matrix<T>::GetSubMatrix(size_t aSubMatrixIndex) const {
        if (aSubMatrixIndex >= mSubMatrices.size()) {
            throw std::out_of_range("SubMatrix Index greater than number of Submatrices ");
        }
        return *mSubMatrices[aSubMatrixIndex];
    }

    template<typename T>
    void Matrix<T>::Initialize(T *apMatrixData, size_t aTotalGlobalNumOfRows, size_t aTotalGlobalNumOfCols,
                               size_t aTileNumOfRows, size_t aTileNumOfCols, common::StorageLayout aStorageLayout,
                               MatrixDecomposer &aMatrixDecomposer, hicmapp::runtime::HicmaContext &aContext,
                               size_t aRank) {

        this->mGlobalNumOfRowsInMatrix = aTotalGlobalNumOfRows;
        this->mGlobalNumOfColsInMatrix = aTotalGlobalNumOfCols;
        this->mGlobalNumOfRowsInTile = aTileNumOfRows;
        this->mGlobalNumOfColsInTile = aTileNumOfCols;
        this->mStorageLayout = aStorageLayout;
        this->mFixedRank = aRank;
//        auto RunTime_instance = hicmapp::runtime::RunTimeSingleton<T>::GetRunTimeInstance();
        if (aRank > 0) {
            this->mTileType = COMPRESSED;
        } else {
            this->mTileType = DENSE;
        }

        if (aRank < 0 || !IsMatrixValid()) {
            throw std::invalid_argument("Matrix::Initialize Invalid Matrix Initialization");
        }

        auto process_id = 0;
        auto number_of_processes = 1;

#ifdef HICMAPP_USE_MPI
        MPI_Comm_rank(mContext.GetCommunicator().GetMPICommunicatior(), &process_id);
        if (process_id == 0) {
            this->mMatrixId = GenerateMatrixId();
        }

        MPI_Bcast(&this->mMatrixId, 1, MPI_UNSIGNED_LONG, 0, mContext.GetCommunicator().GetMPICommunicatior());
        MPI_Comm_size(mContext.GetCommunicator().GetMPICommunicatior(), &number_of_processes);
#else
        this->mMatrixId = GenerateMatrixId();
#endif

        if (mGlobalNumOfColsInTile == 0 || mGlobalNumOfRowsInTile == 0) throw std::runtime_error("Division By zero");

        mGlobalNumOfTilesInRows = (mGlobalNumOfRowsInMatrix + mGlobalNumOfRowsInTile - 1) / mGlobalNumOfRowsInTile;
        mGlobalNumOfTilesInCols = (mGlobalNumOfColsInMatrix + mGlobalNumOfColsInTile - 1) / mGlobalNumOfColsInTile;

        auto specs = aMatrixDecomposer.Decompose(mGlobalNumOfTilesInRows,
                                                 mGlobalNumOfTilesInCols,
                                                 mDiagonalMatrix);

        for (size_t i = 0; i < specs.size(); i++) {

            /// calculating the first and last element index in the submatrix in terms of rows.
            auto sub_matrix_st_idx_in_rows = specs[i].GetStartingIndexInRows() * mGlobalNumOfRowsInTile;
            auto sub_matrix_end_idx_in_rows =
                    sub_matrix_st_idx_in_rows + specs[i].GetNumOfTilesInRow() * mGlobalNumOfRowsInTile;

            sub_matrix_end_idx_in_rows = std::min(sub_matrix_end_idx_in_rows, mGlobalNumOfRowsInMatrix);

            auto num_of_elements_in_rows = sub_matrix_end_idx_in_rows - sub_matrix_st_idx_in_rows;

            auto sub_matrix_st_idx_in_cols = specs[i].GetStartingIndexInCols() * mGlobalNumOfColsInTile;
            auto sub_matrix_end_idx_in_cols =
                    sub_matrix_st_idx_in_cols + specs[i].GetNumOfTilesInCol() * mGlobalNumOfColsInTile;

            sub_matrix_end_idx_in_cols = std::min(sub_matrix_end_idx_in_cols, mGlobalNumOfColsInMatrix);

            auto num_of_elements_in_cols = sub_matrix_end_idx_in_cols - sub_matrix_st_idx_in_cols;

            specs[i].SetTotalSubMatrixNumOfElementsInRows(num_of_elements_in_rows);
            specs[i].SetTotalSubMatrixNumOfElementsInCols(num_of_elements_in_cols);

            if (specs[i].GetOwnerId() != process_id) {
                continue;
            }

            if ((specs[i].GetNumOfTilesInRow() == 0) || (specs[i].GetNumOfTilesInCol() == 0)) {
                continue;
            }
            if (apMatrixData == nullptr) {
                mSubMatrices.push_back(
                        new SubMatrix<T>(nullptr, mGlobalNumOfRowsInTile, mGlobalNumOfColsInTile,
                                         mGlobalNumOfRowsInMatrix,
                                         mGlobalNumOfColsInMatrix, num_of_elements_in_rows, num_of_elements_in_cols,
                                         specs[i].GetStartingIndexInRows(), specs[i].GetStartingIndexInCols(),
                                         aStorageLayout, process_id, mContext, aRank));
                continue;
            }
            mSubMatrices.push_back(
                    new SubMatrix<T>(apMatrixData, mGlobalNumOfRowsInTile, mGlobalNumOfColsInTile,
                                     mGlobalNumOfRowsInMatrix, mGlobalNumOfColsInMatrix, num_of_elements_in_rows,
                                     num_of_elements_in_cols, specs[i].GetStartingIndexInRows(),
                                     specs[i].GetStartingIndexInCols(), aStorageLayout, process_id, mContext, aRank));
        }

        mMemory = 0;
        mSpecs = specs;
        for (auto &submatrix: mSubMatrices) {
            mMemory += submatrix->GetMemoryFootprint();
        }

//        RunTime_instance->RegisterHandles(this);
    }

    template<typename T>
    void Matrix<T>::Initialize(T *apMatrixData, size_t aTotalGlobalNumOfRows, size_t aTotalGlobalNumOfCols,
                               size_t aTileNumOfRows, size_t aTileNumOfCols, common::StorageLayout aStorageLayout,
                               MatrixDecomposer &aMatrixDecomposer, hicmapp::runtime::HicmaContext &aContext,
                               const CompressionParameters &aParams) {
        this->mGlobalNumOfRowsInMatrix = aTotalGlobalNumOfRows;
        this->mGlobalNumOfColsInMatrix = aTotalGlobalNumOfCols;
        this->mGlobalNumOfRowsInTile = aTileNumOfRows;
        this->mGlobalNumOfColsInTile = aTileNumOfCols;
        this->mStorageLayout = aStorageLayout;
        this->mTileType = COMPRESSED;

        if (aParams.GetFixedRank() < 0 || !IsMatrixValid()) {
            throw std::invalid_argument("Matrix::Initialize Invalid Matrix Initialization");
        }

        auto process_id = 0;
        auto number_of_processes = 1;

#ifdef HICMAPP_USE_MPI
        MPI_Comm_rank(mContext.GetCommunicator().GetMPICommunicatior(), &process_id);
        if (process_id == 0) {
            this->mMatrixId = GenerateMatrixId();
        }

        MPI_Bcast(&this->mMatrixId, 1, MPI_UNSIGNED_LONG, 0, mContext.GetCommunicator().GetMPICommunicatior());
        MPI_Comm_size(mContext.GetCommunicator().GetMPICommunicatior(), &number_of_processes);

#else
        this->mMatrixId = GenerateMatrixId();
#endif


        if (mGlobalNumOfColsInTile == 0 || mGlobalNumOfRowsInTile == 0) throw std::runtime_error("Division By zero");

        mGlobalNumOfTilesInRows = (mGlobalNumOfRowsInMatrix + mGlobalNumOfRowsInTile - 1) / mGlobalNumOfRowsInTile;
        mGlobalNumOfTilesInCols = (mGlobalNumOfColsInMatrix + mGlobalNumOfColsInTile - 1) / mGlobalNumOfColsInTile;


        auto specs = aMatrixDecomposer.Decompose(mGlobalNumOfTilesInRows, mGlobalNumOfTilesInCols, mDiagonalMatrix);

        size_t sub_matrix_idx = 0;
        for (size_t i = 0; i < specs.size(); i++) {

            /// calculating the first and last element index in the submatrix in terms of rows.
            auto sub_matrix_st_idx_in_rows = specs[i].GetStartingIndexInRows() * mGlobalNumOfRowsInTile;
            auto sub_matrix_end_idx_in_rows =
                    sub_matrix_st_idx_in_rows + specs[i].GetNumOfTilesInRow() * mGlobalNumOfRowsInTile;

            sub_matrix_end_idx_in_rows = std::min(sub_matrix_end_idx_in_rows, mGlobalNumOfRowsInMatrix);

            auto num_of_elements_in_rows = sub_matrix_end_idx_in_rows - sub_matrix_st_idx_in_rows;

            auto sub_matrix_st_idx_in_cols = specs[i].GetStartingIndexInCols() * mGlobalNumOfColsInTile;
            auto sub_matrix_end_idx_in_cols =
                    sub_matrix_st_idx_in_cols + specs[i].GetNumOfTilesInCol() * mGlobalNumOfColsInTile;

            sub_matrix_end_idx_in_cols = std::min(sub_matrix_end_idx_in_cols, mGlobalNumOfColsInMatrix);

            auto num_of_elements_in_cols = sub_matrix_end_idx_in_cols - sub_matrix_st_idx_in_cols;

            specs[i].SetTotalSubMatrixNumOfElementsInRows(num_of_elements_in_rows);
            specs[i].SetTotalSubMatrixNumOfElementsInCols(num_of_elements_in_cols);

            if (specs[i].GetOwnerId() != process_id) {
                continue;
            }

            if ((specs[i].GetNumOfTilesInRow() == 0) || (specs[i].GetNumOfTilesInCol() == 0)) {
                continue;
            }
            if (apMatrixData == nullptr) {
                mSubMatrices.push_back(
                        new SubMatrix<T>(nullptr, mGlobalNumOfRowsInTile, mGlobalNumOfColsInTile,
                                         mGlobalNumOfRowsInMatrix,
                                         mGlobalNumOfColsInMatrix, num_of_elements_in_rows, num_of_elements_in_cols,
                                         specs[i].GetStartingIndexInRows(), specs[i].GetStartingIndexInCols(),
                                         aStorageLayout, process_id, mContext, aParams));
                continue;
            }

            mSubMatrices.push_back(
                    new SubMatrix<T>(apMatrixData, mGlobalNumOfRowsInTile, mGlobalNumOfColsInTile,
                                     mGlobalNumOfRowsInMatrix, mGlobalNumOfColsInMatrix, num_of_elements_in_rows,
                                     num_of_elements_in_cols, specs[i].GetStartingIndexInRows(),
                                     specs[i].GetStartingIndexInCols(), aStorageLayout, process_id, mContext, aParams));
        }

        mSpecs = specs;
        mMemory = 0;
        for (auto &submatrix: mSubMatrices) {
            mMemory += submatrix->GetMemoryFootprint();
        }

    }

    template<typename T>
    size_t Matrix<T>::GetNumOfGlobalTilesInRows() const {
        return mGlobalNumOfTilesInRows;
    }

    template<typename T>
    size_t Matrix<T>::GetNumOfGlobalTilesInCols() const {
        return mGlobalNumOfTilesInCols;
    }

    template<typename T>
    bool Matrix<T>::ContainsTile(size_t aTileIdxInRows, size_t aTileIdxInCols) const {
        for (auto &subMatrix: mSubMatrices) {
            if (subMatrix->ContainsTile(aTileIdxInRows, aTileIdxInCols)) {
                return true;
            }
        }
        return false;
    }

    //todo throw out of range exception
    template<typename T>
    Tile<T> *Matrix<T>::GetTilePointer(size_t aTileIdxInRows, size_t aTileIdxInCols) {
        for (auto subMatrix: mSubMatrices) {
            if (subMatrix->ContainsTile(aTileIdxInRows, aTileIdxInCols)) {
                return subMatrix->GetTilePointer(aTileIdxInRows, aTileIdxInCols);
            }
        }
        throw std::out_of_range("Matrix::GetTilePointer, out of range tile.\n");
    }

    template<typename T>
    common::StorageLayout Matrix<T>::GetStorageLayout() const {
        return mStorageLayout;
    }

    template<typename T>
    void Matrix<T>::Print(std::ostream &aOutStream) {
        aOutStream << "Matrix :" << std::endl;
        for (size_t cols = 0; cols < this->GetGlobalNumOfColsInMatrix(); cols += this->GetNumOfColsInTile()) {
            for (size_t rows = 0;
                 rows < this->GetGlobalNumOfRowsInMatrix(); rows += this->GetNumOfRowsInTile()) {
                auto tile_index_r = rows / this->GetNumOfRowsInTile();
                auto tile_index_c = cols / this->GetNumOfColsInTile();
                if (this->ContainsTile(tile_index_r, tile_index_c)) {
                    aOutStream << "Tile( " << tile_index_r << "," << tile_index_c << ")" << std::endl;
                    this->GetTilePointer(tile_index_r, tile_index_c)->Print(aOutStream);
                }
            }
        }
    }

    template<typename T>
    hcorepp::helpers::RawMatrix<T> Matrix<T>::ToRawMatrix(runtime::HicmaContext &aContext) {
        int process_id = 0;
#ifdef HICMAPP_USE_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
#endif
        size_t full_array_index;
        size_t tile_index_r;
        size_t tile_index_c;
        size_t index_in_tile;
        hcorepp::helpers::RawMatrix<T> ret(this->mGlobalNumOfRowsInMatrix, this->mGlobalNumOfColsInMatrix);
        auto data_ptr = ret.GetData();
        for (size_t cols = 0; cols < mGlobalNumOfColsInMatrix; cols += mGlobalNumOfColsInTile) {
            for (size_t rows = 0; rows < mGlobalNumOfRowsInMatrix; rows += mGlobalNumOfRowsInTile) {
                size_t tile_rows = std::min(mGlobalNumOfRowsInTile, mGlobalNumOfRowsInMatrix - rows);
                size_t tile_cols = std::min(mGlobalNumOfColsInTile, mGlobalNumOfColsInMatrix - cols);
                tile_index_r = rows / mGlobalNumOfRowsInTile;
                tile_index_c = cols / mGlobalNumOfColsInTile;
                T *temp;

                auto tile_idx = tile_index_r + (tile_index_c * mGlobalNumOfTilesInRows);
                if (!this->ContainsTile(tile_index_r, tile_index_c)) {
#ifdef HICMAPP_USE_MPI
                    temp = new T[tile_rows * tile_cols];
#endif
                } else {
                    auto *tile = this->GetTilePointer(tile_index_r, tile_index_c);
                    if (tile->isDense()) {
                        auto &sub_matrix = tile->GetDataHolder().get();
                        size_t n = sub_matrix.GetNumOfCols();
                        size_t m = sub_matrix.GetNumOfRows();
                        temp = new T[n * m];
                        hcorepp::memory::Memcpy<T>(temp, sub_matrix.GetData(), n * m,
                                                   aContext.GetMainContext(),
                                                   hcorepp::memory::MemoryTransfer::DEVICE_TO_HOST);
                        aContext.SyncMainContext();
                    } else {
                        auto *comp_tile = static_cast<CompressedTile<T> *>(tile);
                        auto m = comp_tile->GetNumOfRows();
                        auto n = comp_tile->GetNumOfCols();
                        auto rank = comp_tile->GetTileRank();
                        size_t num_elements = m * rank;
                        T *cu = new T[num_elements];
                        hcorepp::memory::Memcpy<T>(cu, comp_tile->GetUMatrix(), num_elements,
                                                   aContext.GetMainContext(),
                                                   hcorepp::memory::MemoryTransfer::DEVICE_TO_HOST);
                        num_elements = rank * n;
                        T *cv = new T[num_elements];
                        hcorepp::memory::Memcpy<T>(cv, comp_tile->GetVMatrix(), num_elements,
                                                   aContext.GetMainContext(),
                                                   hcorepp::memory::MemoryTransfer::DEVICE_TO_HOST);
                        aContext.SyncMainContext();
                        temp = new T[n * m];
                        memset(temp, 0, m * n * sizeof(T));

                        blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                                   m, n, rank, 1.0, cu,
                                   comp_tile->GetULeadingDim(), cv,
                                   comp_tile->GetVLeadingDim(), 0.0, temp, m);
                        delete[] cu;
                        delete[] cv;
                    }
                }
#ifdef HICMAPP_USE_MPI
                MPI_Bcast((char *)temp, tile_rows * tile_cols * sizeof(T), MPI_CHAR,
                          GetTileOwnerId(tile_index_r, tile_index_c),
                          MPI_COMM_WORLD);
#endif

                for (size_t i = 0; i < tile_cols; i++) {
                    for (size_t j = 0; j < tile_rows; j++) {
                        index_in_tile = i * tile_rows + j;
                        full_array_index = rows + j + ((cols + i) * mGlobalNumOfRowsInMatrix);
                        data_ptr[full_array_index] = temp[index_in_tile];
                    }
                }
                delete[] temp;
            }
        }
        return ret;
    }

    template<typename T>
    int Matrix<T>::GetSubMatrixOwnerId(size_t aTileIdxInRows, size_t aTileIdxInCols) const {
        for (auto &subMatrix: mSubMatrices) {
            if (subMatrix->ContainsTile(aTileIdxInRows, aTileIdxInCols)) {
                return subMatrix->GetSubMatrixOwnerId();
            }
        }
        return -1;
    }

    template<typename T>
    int Matrix<T>::GetTileOwnerId(size_t aTileIdxInRows, size_t aTileIdxInCols) const {
        for (auto &sub_matrix_spec: mSpecs) {
            auto st_idx_in_rows = sub_matrix_spec.GetStartingIndexInRows();
            auto st_idx_in_cols = sub_matrix_spec.GetStartingIndexInCols();
            auto end_idx_in_rows = sub_matrix_spec.GetStartingIndexInRows() + sub_matrix_spec.GetNumOfTilesInRow();
            auto end_idx_in_cols = sub_matrix_spec.GetStartingIndexInCols() + sub_matrix_spec.GetNumOfTilesInCol();

            if (aTileIdxInRows >= st_idx_in_rows && aTileIdxInRows < end_idx_in_rows) {
                if (aTileIdxInCols >= st_idx_in_cols && aTileIdxInCols < end_idx_in_cols) {
                    return sub_matrix_spec.GetOwnerId();
                }
            }
        }
        return -1;
    }

    template<typename T>
    runtime::HicmaContext &Matrix<T>::GetContext() const {
        return mContext;
    }

    template<typename T>
    bool Matrix<T>::IsMatrixValid() const {

        if (!(std::is_floating_point<T>::value && (sizeof(T) == sizeof(float) || sizeof(T) == sizeof(double)))) {
            return false;
        }
        if (mGlobalNumOfRowsInTile <= 0 || mGlobalNumOfColsInTile <= 0) {
            return false;
        }
        if (mGlobalNumOfRowsInMatrix <= 0 || mGlobalNumOfColsInMatrix <= 0) {
            return false;
        }
        for (auto sub_matrix: mSubMatrices) {
            if (!sub_matrix->IsValid()) {
                return false;
            }
        }
        return true;
    }

    template<typename T>
    size_t Matrix<T>::GetNumOfRowsInTile() const {
        return mGlobalNumOfRowsInTile;
    }

    template<typename T>
    size_t Matrix<T>::GetNumOfColsInTile() const {
        return mGlobalNumOfColsInTile;
    }

    template<typename T>
    size_t Matrix<T>::GetGlobalNumOfRowsInMatrix() const {
        return mGlobalNumOfRowsInMatrix;
    }

    template<typename T>
    size_t Matrix<T>::GetGlobalNumOfColsInMatrix() const {
        return mGlobalNumOfColsInMatrix;
    }

    template<typename T>
    size_t Matrix<T>::GetMemoryFootprint() {
        return mMemory;
    }

    template<typename T>
    size_t Matrix<T>::GetTileLeadingDim(size_t aTileIdx) {
        size_t leading_dim = 0;
        if (mStorageLayout == common::StorageLayout::HicmaCM) {
            leading_dim = mGlobalNumOfRowsInTile;
            if (aTileIdx == mGlobalNumOfTilesInRows) {
                leading_dim = mGlobalNumOfRowsInMatrix % mGlobalNumOfRowsInTile;
            }
        } else if (mStorageLayout == common::StorageLayout::HicmaRM) {
            leading_dim = mGlobalNumOfColsInTile;
            if (aTileIdx == mGlobalNumOfTilesInCols) {
                leading_dim = mGlobalNumOfColsInMatrix % mGlobalNumOfColsInTile;
            }
        }
        return leading_dim;
    }

    template<typename T>
    TileMetadata *Matrix<T>::GetTileMetadata(size_t aTileRowIdx, size_t aTileColIdx) {
        int owner = this->GetTileOwnerId(aTileRowIdx, aTileColIdx);
        int myrank = 0;
#ifdef HICMAPP_USE_MPI
        MPI_Comm_rank(this->GetContext().GetCommunicator().GetMPICommunicatior(), &myrank);
#endif
        if (myrank == owner) {
            auto *tile = this->GetTilePointer(aTileRowIdx, aTileColIdx);
            auto metadata_data = hcorepp::operators::TilePacker<T>::UnPackTile(*tile,
                                                                               hcorepp::kernels::ContextManager::GetInstance().GetContext());

            return metadata_data.first;
        } else {
            size_t tile_rows = mGlobalNumOfRowsInTile;
            size_t tile_cols = mGlobalNumOfColsInTile;
            for (size_t i = 0; i < mSpecs.size(); i++) {
                auto sub_matrix_spec = mSpecs[i];
                auto mNumOfTilesinRows =
                        (sub_matrix_spec.GetTotalSubMatrixNumOfElementsInRows() + tile_rows - 1) / tile_rows;
                auto mNumOfTilesinCols =
                        (sub_matrix_spec.GetTotalSubMatrixNumOfElementsInCols() + tile_cols - 1) / tile_cols;

                auto sub_matrix_st_idx_in_rows = sub_matrix_spec.GetStartingIndexInRows() * mGlobalNumOfRowsInTile;
                auto sub_matrix_end_idx_in_rows =
                        sub_matrix_st_idx_in_rows + sub_matrix_spec.GetNumOfTilesInRow() * mGlobalNumOfRowsInTile;

                sub_matrix_end_idx_in_rows = std::min(sub_matrix_end_idx_in_rows, mGlobalNumOfRowsInMatrix);

                size_t sub_matrix_tile_st_idx_rows = sub_matrix_spec.GetStartingIndexInRows();
                size_t sub_matrix_tile_end_idx_rows = sub_matrix_spec.GetStartingIndexInRows() + mNumOfTilesinRows;
                size_t sub_matrix_tile_st_idx_cols = sub_matrix_spec.GetStartingIndexInCols();
                size_t sub_matrix_tile_end_idx_cols = sub_matrix_spec.GetStartingIndexInCols() + mNumOfTilesinCols;

                bool row_check = (aTileRowIdx < sub_matrix_tile_end_idx_rows &&
                                  aTileRowIdx >= sub_matrix_tile_st_idx_rows);
                bool column_check = (aTileColIdx < sub_matrix_tile_end_idx_cols &&
                                     aTileColIdx >= sub_matrix_tile_st_idx_cols);
                if (!row_check || !column_check) {
                    continue;
                }

                auto remainder_rows = (sub_matrix_spec.GetTotalSubMatrixNumOfElementsInRows() % tile_rows != 0);
                auto remainder_cols = (sub_matrix_spec.GetTotalSubMatrixNumOfElementsInCols() % tile_cols != 0);

                if (remainder_rows && aTileRowIdx == sub_matrix_tile_end_idx_rows - 1) {
                    /// remainder tile in rows
                    tile_rows = std::min(tile_rows,
                                         sub_matrix_spec.GetTotalSubMatrixNumOfElementsInRows() -
                                         aTileRowIdx * tile_rows);
                }
                if (remainder_cols && aTileColIdx == sub_matrix_tile_end_idx_cols - 1) {
                    /// remainder tile in cols
                    tile_cols = std::min(tile_cols,
                                         sub_matrix_spec.GetTotalSubMatrixNumOfElementsInCols() -
                                         aTileColIdx * tile_cols);
                }

                auto tile_layout = (blas::Layout) mStorageLayout;
                auto tile_leading_dim = (tile_layout == blas::Layout::ColMajor) ? tile_rows : tile_cols;
                auto tile_matrix_rank = mFixedRank;
                auto tile_dense = mTileType;
                auto max_rank = std::max(std::min(tile_rows, tile_cols) / MAX_RANK_RATIO, 1UL);
                TileMetadata *metadata = new TileMetadata(tile_rows, tile_cols, tile_matrix_rank, max_rank,
                                                          tile_leading_dim,
                                                          tile_layout, tile_dense);

                return metadata;
            }

        }

        return nullptr;
    }

    HICMAPP_INSTANTIATE_CLASS(Matrix)
}