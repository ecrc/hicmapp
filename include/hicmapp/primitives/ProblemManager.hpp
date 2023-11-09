#ifndef HICMAPP_PROBLEMMANAGER_HPP
#define HICMAPP_PROBLEMMANAGER_HPP
extern "C" {
#include "starsh.h"
}

#include <variant>
#include <string>
#include <unordered_map>
#include <hicmapp/common/definitions.h>
#include <memory>
#include <set>

namespace hicmapp {
    namespace primitives {

        enum ProblemProperty {
            HICMA_PROB_PROPERTY_MESH_POINTS,
            HICMA_PROB_PROPERTY_MORDERING,
            HICMA_PROB_PROPERTY_MESH_FILE,
            HICMA_PROB_PROPERTY_INTERPL_FILE,
            HICMA_PROB_PROPERTY_NTRIAN,
            HICMA_PROB_PROPERTY_NIPP,
            HICMA_PROB_PROPERTY_DIAG,
            HICMA_PROB_PROPERTY_WAVE_K,
            HICMA_PROB_PROPERTY_DECAY,
            HICMA_PROB_PROPERTY_NOISE,
            HICMA_PROB_PROPERTY_THETA,
            HICMA_PROB_PROPERTY_POINT,
            HICMA_PROB_PROPERTY_REG,
            HICMA_PROB_PROPERTY_ISREG,
            HICMA_PROB_PROPERTY_RAD,
            HICMA_PROB_PROPERTY_NUMOBJ,
            HICMA_PROB_PROPERTY_DENST,
            HICMA_PROB_PROPERTY_NU,
            HICMA_PROB_PROPERTY_BETA,
            HICMA_PROB_PROPERTY_NDIM,
            HICMA_PROB_PROPERTY_N,
            HICMA_PROB_PROPERTY_BLOCK_SIZE,
            HICMA_PROB_PROPERTY_SYM,
            HICMA_PROB_PROPERTY_MT,
            HICMA_PROB_PROPERTY_NT

        };

        class ProblemManager {
        public:
            explicit ProblemManager(hicmapp::common::ProblemType aProblemType);

            ~ProblemManager();

            ///
            void InitAC3DProblemMetadata();

            ///
            void InitEDSINProblemMetadata();

            ///
            void InitGeostatNonGaussianProblemMetadata();

            ///
            void InitGeostatNonGaussianPointProblemMetadata();

            ///
            void InitGeostatParsimoniousBivariatePointProblemMetadata();

            ///
            void InitGeostatParsimoniousBivariateProblemMetadata();

            ///
            void InitGeostatPointProblemMetadata();

            ///
            void InitGeostatProblemMetadata();

            ///
            void InitRBFCube3DProblemMetadata();

            ///
            void InitRBFVirus3DProblemMetadata();

            ///
            void InitRNDProblemMetadata();

            ///
            void InitSSProblemMetadata();

            ///
            void InitST2DExpProblemMetadata();

            ///
            void InitST3DExpProblemMetadata();

            ///
            void InitST3DSQExpProblemMetadata();

            common::ProblemType GetProblemType() const;
            //            template<typename T>
//            T SetProblemProperty(ProblemProperty aProperty, T aPropertyValue);

            template<typename T>
            T GetProblemProperty(ProblemProperty aProperty);

            void SetProblemProperty(ProblemProperty aProperty, int aPropertyValue);
            void SetProblemProperty(ProblemProperty aProperty, double aPropertyValue);
            void SetProblemProperty(ProblemProperty aProperty, double* aPropertyValue);
            void SetProblemProperty(ProblemProperty aProperty, char aPropertyValue);
            void SetProblemProperty(ProblemProperty aProperty, char* aPropertyValue);


        private:

            enum hicmapp::common::ProblemType mProblemType;
            std::set<ProblemProperty> mProblemProperties;
            std::unordered_map<ProblemProperty, char> mCharProperties;
            std::unordered_map<ProblemProperty, int> mIntProblemProperties;
            std::unordered_map<ProblemProperty, double> mDoubleProblemProperties;
            std::unordered_map<ProblemProperty, double *> mP2DoubleProblemProperties;
            std::unordered_map<ProblemProperty, char *> mStringProblemProperties;
        };
    }
}
#endif //HICMAPP_PROBLEMMANAGER_HPP
