
#ifndef HICMAPP_PRIMITIVES_CONCRETE_MATRIX_TWO_DIM_CYCLIC_DECOMPOSER_HPP
#define HICMAPP_PRIMITIVES_CONCRETE_MATRIX_TWO_DIM_CYCLIC_DECOMPOSER_HPP

#include <cstddef>
#include <vector>

#include "hicmapp/primitives/decomposer/matrix_decomposer.hpp"
#include "hicmapp/runtime/interface/HicmaContext.hpp"

namespace hicmapp::primitives {

        class TwoDimCyclicDecomposer : public MatrixDecomposer {

        public:

            /**
             * @brief TwoDimCyclicDecomposer constructor
             *
             * @param aNumOfProcessesInRows
             * Number of processes across global matrix rows.
             * @param aNumOfProcessesInCols
             * Number of processes across global matrix columns.
             */
            TwoDimCyclicDecomposer(size_t aNumOfProcessesInRows, size_t aNumOfProcessesInCols);

            /**
             * @brief Decomposes the Matrix into sub-matrices, where every sub-matrix is assigned to a process.
             * When number of sub-matrices > number of processes the assignment is done in a round-robin schedule.
             * The TwoDimCyclicDecomposer supports RowMajor only.(TODO: add ColumnMajor support.)
             *
             * @param aGlobalMatrixTilesInRows
             * Number of tiles across global matrix rows.
             * @param aGlobalMatrixTilesInCols
             * Number of tiles across global matrix columns.
             */
            std::vector<MatrixSpecifications>
            Decompose(size_t aGlobalMatrixTilesInRows, size_t aGlobalMatrixTilesInCols,
                      bool aDiagonalMatrix = false) override;

            DecomposerType
            GetType() override {
                return CYCLIC2D;
            };

        private:
            size_t mNumOfProcessesInRows;
            size_t mNumOfProcessesInCols;
        };
    }
#endif //HICMAPP_PRIMITIVES_CONCRETE_MATRIX_TWO_DIM_CYCLIC_DECOMPOSER_HPP
