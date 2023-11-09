
#ifndef HICMAPP_PRIMITIVES_CONCRETE_MATRIX_SLOWEST_DIM_DECOMPOSER_HPP
#define HICMAPP_PRIMITIVES_CONCRETE_MATRIX_SLOWEST_DIM_DECOMPOSER_HPP

#include <cstddef>
#include <vector>

#include "hicmapp/primitives/decomposer/matrix_decomposer.hpp"
#include "hicmapp/common/definitions.h"
#include "hicmapp/runtime/interface/HicmaContext.hpp"

namespace hicmapp {
    namespace primitives {

        /**
         * Slowest Dimension decomposer will be decomposing the matrix tiles equally across the number of given
         * sub-matrices according to the slowest dimension.
         */
        class SlowestDimDecomposer : public MatrixDecomposer {

        public:
            /**
             * @brief SlowestDimDecomposer constructor
             *
             * @param aNumOfSubMatrices
             * Number of sub-matrices across global matrix
             * @param aStorageLayout
             * Global matrix layout, either RowMajor or ColMajor.
             */
            SlowestDimDecomposer(size_t aNumOfSubMatrices, common::StorageLayout aStorageLayout);

            /**
             * @brief Decompose the matrix into sub-matrices,
             * by dividing the number of tiles by the number of sub-matrices.
             *
             * @param aGlobalMatrixTilesInRows
             * Number of tiles across the global matrix rows.
             * @param aGlobalMatrixTilesInCols
             * Number of tiles across the global matrix cols.
             * @return
             */
            std::vector<MatrixSpecifications>
            Decompose(size_t aGlobalMatrixTilesInRows, size_t aGlobalMatrixTilesInCols,
                      bool aDiagonalMatrix = false) override;

            DecomposerType
            GetType() override {
                return SLOWESTDIM;
            };

        private:
            common::StorageLayout mStorageLayout;
            size_t mNumOfSubMatrices;
        };
    }
}
#endif //HICMAPP_PRIMITIVES_CONCRETE_MATRIX_SLOWEST_DIM_DECOMPOSER_HPP
