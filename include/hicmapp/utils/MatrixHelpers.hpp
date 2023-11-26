#ifndef HICMAPP_UTILS_MATRIX_HELPERS_HPP
#define HICMAPP_UTILS_MATRIX_HELPERS_HPP

#include <hicmapp/primitives/matrix.hpp>

/** This file is for Matrix Utilities. These helper functions are to be used in tests and examples for logging and debugging,
 * and is dedicated for utilities that should not necessarily be part of the Matrix Class API
 */

namespace hicmapp::utils {
        template<typename T>
        class MatrixHelpers {

        public:

            /**
             * This converts a Matrix Object to a contiguous array in memory. NOTE: This does not support MPI and the
             * decomposition associated with an MPI workflow. Please use ToRawMatrix function instead.
             * @param[in] aMatrix Matrix to be copied
             * @param[out] aArray Allocated array of size equal to the matrix being copied
             */
            static void
            MatrixToArray(primitives::Matrix<T> &aMatrix, T *&aArray);

            /**
             * Print Utility to print a two-dimensional array based on data layout
             * @param aArray Array to be printed
             * @param aRows Number of rows
             * @param aCols Number of Columns
             * @param aLayout Layout (ColMajor or RowMajor)
             */
            static void
            PrintArray(T *&aArray, size_t aRows, size_t aCols, hicmapp::common::StorageLayout aLayout);
        };

    }
#endif //HICMAPP_UTILS_MATRIX_HELPERS_HPP
