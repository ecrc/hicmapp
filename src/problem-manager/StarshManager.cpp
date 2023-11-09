#include "hicmapp/problem-manager/StarshManager.hpp"
#include "hicmapp/primitives/ProblemManager.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <variant>
#include "starsh-spatial.h"
#include "starsh-randtlr.h"
#include "iostream"

namespace hicmapp::operations {
    STARSH_blrf *StarsHManager::starsh_format = nullptr;

    STARSH_blrf *StarsHManager::GetStarsHFormat() {
        if (starsh_format != nullptr) {
            return starsh_format;
        } else {
            // throw exception..
        }
    }

    void StarsHManager::SetStarsHFormat(primitives::ProblemManager &aProblemManager) {

        int info = 0;
        STARSH_problem *problem = nullptr;
        void *data = nullptr;
        STARSH_kernel *kernel = nullptr;
        char dtype = 'd';

        auto N = aProblemManager.GetProblemProperty<int>(primitives::ProblemProperty::HICMA_PROB_PROPERTY_N);
        auto Ndim = aProblemManager.GetProblemProperty<int>(primitives::ProblemProperty::HICMA_PROB_PROPERTY_NDIM);
        auto block_size = aProblemManager.GetProblemProperty<int>(primitives::ProblemProperty::HICMA_PROB_PROPERTY_BLOCK_SIZE);
        char sym = aProblemManager.GetProblemProperty<char>(primitives::ProblemProperty::HICMA_PROB_PROPERTY_SYM);

        std::string problem_str;
        if (aProblemManager.GetProblemType() == hicmapp::common::ProblemType::PROBLEM_TYPE_SS) {
            auto Nu = aProblemManager.GetProblemProperty<double>(primitives::ProblemProperty::HICMA_PROB_PROPERTY_NU);
            auto Beta = aProblemManager.GetProblemProperty<double>(primitives::ProblemProperty::HICMA_PROB_PROPERTY_BETA);
            auto Noise = aProblemManager.GetProblemProperty<double>(primitives::ProblemProperty::HICMA_PROB_PROPERTY_NOISE);


            int kernel_type = STARSH_SPATIAL_SQREXP_SIMD;
            srand(0); // FIXME
            enum STARSH_PARTICLES_PLACEMENT place = STARSH_PARTICLES_UNIFORM;
            info = starsh_application((void **) &data, &kernel, N,
                                      dtype, STARSH_SPATIAL, kernel_type,
                                      STARSH_SPATIAL_NDIM, Ndim, STARSH_SPATIAL_BETA,
                                      Beta, STARSH_SPATIAL_NU, Nu,
                                      STARSH_SPATIAL_NOISE, Noise, STARSH_SPATIAL_PLACE, place,
                                      0);
            problem_str = "ST_2D_SQEXP";
        } else if (aProblemManager.GetProblemType() == hicmapp::common::ProblemType::PROBLEM_TYPE_RND) {
            auto Noise = aProblemManager.GetProblemProperty<double>(primitives::ProblemProperty::HICMA_PROB_PROPERTY_NOISE);
            auto decay = aProblemManager.GetProblemProperty<double>(primitives::ProblemProperty::HICMA_PROB_PROPERTY_DECAY);

            int kernel_type = STARSH_RANDTLR_KERNEL1;
            info = starsh_application((void **) &data, &kernel, N, dtype,
                                      STARSH_RANDTLR, kernel_type, STARSH_RANDTLR_NB, block_size,
                                      STARSH_RANDTLR_DECAY, decay, STARSH_RANDTLR_DIAG,
                                      Noise,
                                      0);
            problem_str = "Randomly generated matrix";
        } else {
            fprintf(stderr, "Unknown type of STARS-H problem:%d. Exiting...\n", aProblemManager.GetProblemType());
        }

        if (info != 0) {
            printf("wrong parameters for starsh_application()\n");
//            exit(info);
        }

        STARSH_int shape[] = {N, N};


        starsh_problem_new(&problem, Ndim,
                           shape, sym, dtype, data, data, kernel,
                           (char *) problem_str.c_str());
        STARSH_cluster *cluster;
        info = starsh_cluster_new_plain(&cluster, data, N, block_size);
        if (info != 0) {
            printf("Error in creation of cluster\n");
//            exit(info);
        }

        STARSH_blrf *F;
        info = starsh_blrf_new_tlr(&F, problem, sym, cluster, cluster);
        if (info != 0) {
            printf("Error in creation of format\n");
//            exit(info);
        }
        if (starsh_format != nullptr) {
            starsh_blrf_free(starsh_format);
            starsh_format = nullptr;
        }
        starsh_format = F;

    }

    void StarsHManager::DestroyStarsHManager() {
        if (starsh_format != nullptr) {
            starsh_blrf_free(starsh_format);
            starsh_format = nullptr;
        }
    }
}