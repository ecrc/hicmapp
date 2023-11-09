#ifndef HICMAPP_COMMON_DEFINITIONS_HPP
#define HICMAPP_COMMON_DEFINITIONS_HPP

// Macro definition to instantiate the HiCMA template classes with supported types.
#define HICMAPP_INSTANTIATE_CLASS(TEMPLATE_CLASS)   template class TEMPLATE_CLASS<float>;  \
                                                    template class TEMPLATE_CLASS<double>;

namespace hicmapp::common {
        /**
         * @brief
         * Enum denoting the storage layout of a matrix or tile.
         */
        enum class StorageLayout {
            HicmaCM = 101,
            HicmaRM = 102,
            HicmaCCRB = 103,
            HicmaCRRB = 104,
            HicmaRCRB = 105,
            HicmaRRRB = 106
        };

        /**
         * @brief
         * Enum denoting the Data Type used.
         */
        enum class DataType {
            HicmaByte = 0,
            HicmaInteger = 1,
            HicmaRealFloat = 2,
            HicmaRealDouble = 3,
            HicmaComplexFloat = 4,
            HicmaComplexDouble = 5
        };

        /**
         * @brief
         * Enum denoting the Problem Type used.
         */
        enum class ProblemType {
            PROBLEM_TYPE_RND = 1,
            PROBLEM_TYPE_SS = 2,
            PROBLEM_TYPE_RNDUSR = 3,
            PROBLEM_TYPE_FILE = 4,
            PROBLEM_TYPE_GEOSTAT = 5,
            PROBLEM_TYPE_EDSIN = 6,
            PROBLEM_TYPE_GEOSTAT_POINT = 7,
            PROBLEM_TYPE_ST_3D_EXP = 8,
            PROBLEM_TYPE_ST_3D_SQEXP = 9,
            PROBLEM_TYPE_3D_RBF_VIRUS = 12,
            PROBLEM_TYPE_3D_RBF_CUBE = 13,
            PROBLEM_TYPE_AC_3D = 14,
            PROBLEM_TYPE_ST_2D_EXP = 15,
            PROBLEM_TYPE_GEOSTAT_PARSIMONIOUS_BIVARIATE = 108,
            PROBLEM_TYPE_GEOSTAT_PARSIMONIOUS_BIVARIATE_POINT = 109,
            PROBLEM_TYPE_GEOSTAT_PARSIMONIOUS2_BIVARIATE = 110,
            PROBLEM_TYPE_GEOSTAT_PARSIMONIOUS2_BIVARIATE_POINT = 111
        };

        /// @Todo : should be replaced with an enum in HCOREPP to avoid redundancy.
        /**
         * @brief
         * Enum denoting Uplo
         */
        enum class Uplo {
            HicmaUpper = 121,
            HicmaLower = 122,
            HicmaUpperLower = 123
        };

        enum class RunTimeLibrary{
            DEFAULT,
            STARPU
        };
    }//namespace hicmapp

#endif //HICMAPP_COMMON_DEFINITIONS_HPP