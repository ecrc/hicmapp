extern "C" {
#include "starsh-spatial.h"
}

#include <hicmapp/primitives/ProblemManager.hpp>
#include <iostream>
#include <memory>

namespace hicmapp {
    namespace primitives {


        ProblemManager::ProblemManager(
                hicmapp::common::ProblemType aProblemType) {

            this->mProblemType = aProblemType;

            this->mCharProperties[HICMA_PROB_PROPERTY_SYM] = '\0';

            switch (aProblemType) {
                case common::ProblemType::PROBLEM_TYPE_RND:

                    InitRNDProblemMetadata();

                    break;
                case common::ProblemType::PROBLEM_TYPE_SS:

                    InitSSProblemMetadata();

                    break;

                case common::ProblemType::PROBLEM_TYPE_GEOSTAT:

                    InitGeostatProblemMetadata();

                    break;
                case common::ProblemType::PROBLEM_TYPE_EDSIN:

                    InitEDSINProblemMetadata();

                    break;
                case common::ProblemType::PROBLEM_TYPE_GEOSTAT_POINT:

                    InitGeostatPointProblemMetadata();

                    break;
                case common::ProblemType::PROBLEM_TYPE_ST_3D_EXP:

                    InitST3DExpProblemMetadata();

                    break;
                case common::ProblemType::PROBLEM_TYPE_ST_3D_SQEXP:

                    InitST3DSQExpProblemMetadata();

                    break;
                case common::ProblemType::PROBLEM_TYPE_3D_RBF_VIRUS:

                    InitRBFVirus3DProblemMetadata();

                    break;
                case common::ProblemType::PROBLEM_TYPE_3D_RBF_CUBE:

                    InitRBFCube3DProblemMetadata();

                    break;
                case common::ProblemType::PROBLEM_TYPE_AC_3D:

                    InitAC3DProblemMetadata();

                    break;
                case common::ProblemType::PROBLEM_TYPE_ST_2D_EXP:

                    InitST2DExpProblemMetadata();

                    break;
                case common::ProblemType::PROBLEM_TYPE_GEOSTAT_PARSIMONIOUS_BIVARIATE:

                    InitGeostatParsimoniousBivariateProblemMetadata();

                    break;
                case common::ProblemType::PROBLEM_TYPE_GEOSTAT_PARSIMONIOUS_BIVARIATE_POINT:

                    InitGeostatParsimoniousBivariatePointProblemMetadata();

                    break;
                case common::ProblemType::PROBLEM_TYPE_GEOSTAT_PARSIMONIOUS2_BIVARIATE:

                    InitGeostatNonGaussianProblemMetadata();

                    break;
                case common::ProblemType::PROBLEM_TYPE_GEOSTAT_PARSIMONIOUS2_BIVARIATE_POINT:

                    InitGeostatNonGaussianPointProblemMetadata();

                    break;
                case common::ProblemType::PROBLEM_TYPE_RNDUSR:
                    break;
                case common::ProblemType::PROBLEM_TYPE_FILE:
                    break;
            }
        }


        void ProblemManager::InitAC3DProblemMetadata() {
            mProblemProperties.insert(HICMA_PROB_PROPERTY_MESH_POINTS);
            mProblemProperties.insert(HICMA_PROB_PROPERTY_MORDERING);
            mProblemProperties.insert(HICMA_PROB_PROPERTY_MESH_FILE);
            mProblemProperties.insert(HICMA_PROB_PROPERTY_INTERPL_FILE);
            mProblemProperties.insert(HICMA_PROB_PROPERTY_NTRIAN);
            mProblemProperties.insert(HICMA_PROB_PROPERTY_NIPP);
        }


        void ProblemManager::InitEDSINProblemMetadata() {
            mProblemProperties.insert(HICMA_PROB_PROPERTY_DIAG);
            mProblemProperties.insert(HICMA_PROB_PROPERTY_WAVE_K);
            mProblemProperties.insert(HICMA_PROB_PROPERTY_DECAY);
        }


        void ProblemManager::InitGeostatNonGaussianProblemMetadata() {
            mProblemProperties.insert(HICMA_PROB_PROPERTY_THETA);
        }


        void ProblemManager::InitGeostatNonGaussianPointProblemMetadata() {
            mProblemProperties.insert(HICMA_PROB_PROPERTY_NOISE);
            mProblemProperties.insert(HICMA_PROB_PROPERTY_THETA);
            mProblemProperties.insert(HICMA_PROB_PROPERTY_POINT);
        }


        void
        ProblemManager::InitGeostatParsimoniousBivariatePointProblemMetadata() {
            mProblemProperties.insert(HICMA_PROB_PROPERTY_NOISE);
            mProblemProperties.insert(HICMA_PROB_PROPERTY_THETA);
            mProblemProperties.insert(HICMA_PROB_PROPERTY_POINT);
        }


        void ProblemManager::InitGeostatParsimoniousBivariateProblemMetadata() {
            mProblemProperties.insert(HICMA_PROB_PROPERTY_NOISE);
            mProblemProperties.insert(HICMA_PROB_PROPERTY_THETA);
        }


        void ProblemManager::InitGeostatPointProblemMetadata() {
            mProblemProperties.insert(HICMA_PROB_PROPERTY_NOISE);
            mProblemProperties.insert(HICMA_PROB_PROPERTY_THETA);
            mProblemProperties.insert(HICMA_PROB_PROPERTY_POINT);
        }


        void ProblemManager::InitGeostatProblemMetadata() {
            mProblemProperties.insert(HICMA_PROB_PROPERTY_NOISE);
            mProblemProperties.insert(HICMA_PROB_PROPERTY_THETA);
        }


        void ProblemManager::InitRBFCube3DProblemMetadata() {
            mProblemProperties.insert(HICMA_PROB_PROPERTY_REG);
            mProblemProperties.insert(HICMA_PROB_PROPERTY_ISREG);
            mProblemProperties.insert(HICMA_PROB_PROPERTY_RAD);
            mProblemProperties.insert(HICMA_PROB_PROPERTY_MESH_POINTS);
            mProblemProperties.insert(HICMA_PROB_PROPERTY_MORDERING);
            mProblemProperties.insert(HICMA_PROB_PROPERTY_MESH_FILE);
        }


        void ProblemManager::InitRBFVirus3DProblemMetadata() {
            mProblemProperties.insert(HICMA_PROB_PROPERTY_REG);
            mProblemProperties.insert(HICMA_PROB_PROPERTY_NUMOBJ);
            mProblemProperties.insert(HICMA_PROB_PROPERTY_ISREG);
            mProblemProperties.insert(HICMA_PROB_PROPERTY_RAD);
            mProblemProperties.insert(HICMA_PROB_PROPERTY_DENST);
            mProblemProperties.insert(HICMA_PROB_PROPERTY_MESH_POINTS);
            mProblemProperties.insert(HICMA_PROB_PROPERTY_MORDERING);
            mProblemProperties.insert(HICMA_PROB_PROPERTY_MESH_FILE);
        }


        void ProblemManager::InitRNDProblemMetadata() {
            mProblemProperties.insert(HICMA_PROB_PROPERTY_NDIM);
            mProblemProperties.insert(HICMA_PROB_PROPERTY_NOISE);
            mProblemProperties.insert(HICMA_PROB_PROPERTY_DECAY);
            mProblemProperties.insert(HICMA_PROB_PROPERTY_N);
            mProblemProperties.insert(HICMA_PROB_PROPERTY_BLOCK_SIZE);
            mProblemProperties.insert(HICMA_PROB_PROPERTY_MT);
            mProblemProperties.insert(HICMA_PROB_PROPERTY_NT);
            mProblemProperties.insert(HICMA_PROB_PROPERTY_SYM);
        }


        void ProblemManager::InitSSProblemMetadata() {
            mProblemProperties.insert(HICMA_PROB_PROPERTY_N);
            mProblemProperties.insert(HICMA_PROB_PROPERTY_NDIM);
            mProblemProperties.insert(HICMA_PROB_PROPERTY_BETA);
            mProblemProperties.insert(HICMA_PROB_PROPERTY_NU);
            mProblemProperties.insert(HICMA_PROB_PROPERTY_NOISE);
            mProblemProperties.insert(HICMA_PROB_PROPERTY_WAVE_K);
            mProblemProperties.insert(HICMA_PROB_PROPERTY_DIAG);
            mProblemProperties.insert(HICMA_PROB_PROPERTY_SYM);
            mProblemProperties.insert(HICMA_PROB_PROPERTY_BLOCK_SIZE);
            mProblemProperties.insert(HICMA_PROB_PROPERTY_DECAY);
            mProblemProperties.insert(HICMA_PROB_PROPERTY_MT);
            mProblemProperties.insert(HICMA_PROB_PROPERTY_NT);
        }


        void ProblemManager::InitST2DExpProblemMetadata() {
            mProblemProperties.insert(HICMA_PROB_PROPERTY_BETA);
            mProblemProperties.insert(HICMA_PROB_PROPERTY_NU);
            mProblemProperties.insert(HICMA_PROB_PROPERTY_NOISE);
        }


        void ProblemManager::InitST3DExpProblemMetadata() {
            mProblemProperties.insert(HICMA_PROB_PROPERTY_BETA);
            mProblemProperties.insert(HICMA_PROB_PROPERTY_NU);
            mProblemProperties.insert(HICMA_PROB_PROPERTY_NOISE);
        }


        void ProblemManager::InitST3DSQExpProblemMetadata() {
            mProblemProperties.insert(HICMA_PROB_PROPERTY_BETA);
            mProblemProperties.insert(HICMA_PROB_PROPERTY_NU);
            mProblemProperties.insert(HICMA_PROB_PROPERTY_NOISE);
        }


        template<>
        int
        ProblemManager::GetProblemProperty<int>(ProblemProperty aProperty) {
            if (this->mIntProblemProperties.find(aProperty) != this->mIntProblemProperties.end()) {
                return this->mIntProblemProperties.find(aProperty)->second;
            } else {
                throw std::invalid_argument( "Problem property was not initialized" );
            }
        }


        template<>
        double
        ProblemManager::GetProblemProperty<double>(ProblemProperty aProperty) {
            if (this->mDoubleProblemProperties.find(aProperty) != this->mDoubleProblemProperties.end()) {
                return this->mDoubleProblemProperties.find(aProperty)->second;
            } else {
                throw std::invalid_argument( "Problem property was not initialized" );
            }
        }


        template<>
        double *
        ProblemManager::GetProblemProperty<double *>(ProblemProperty aProperty) {
            if (this->mP2DoubleProblemProperties.find(aProperty) != this->mP2DoubleProblemProperties.end()) {
                return this->mP2DoubleProblemProperties.find(aProperty)->second;
            } else {
                throw std::invalid_argument( "Problem property was not initialized" );
            }
        }


        template<>
        char *
        ProblemManager::GetProblemProperty<char *>(ProblemProperty aProperty) {
            if (this->mStringProblemProperties.find(aProperty) != this->mStringProblemProperties.end()) {
                return this->mStringProblemProperties.find(aProperty)->second;
            } else {
                throw std::invalid_argument( "Problem property was not initialized" );
            }
        }


        template<>
        char
        ProblemManager::GetProblemProperty<char>(ProblemProperty aProperty) {
            if (this->mCharProperties.find(aProperty) != this->mCharProperties.end()) {
                return this->mCharProperties.find(aProperty)->second;
            } else {
                throw std::invalid_argument( "Problem property was not initialized" );
            }
        }


        common::ProblemType ProblemManager::GetProblemType() const {
            return mProblemType;
        }


        ProblemManager::~ProblemManager() {
            mDoubleProblemProperties.clear();
            mIntProblemProperties.clear();
            mCharProperties.clear();
            mStringProblemProperties.clear();
            mP2DoubleProblemProperties.clear();
        }


        void ProblemManager::SetProblemProperty(ProblemProperty aProperty, int aPropertyValue) {
            if(mProblemProperties.find(aProperty) != mProblemProperties.end()) {
                this->mIntProblemProperties.insert(std::make_pair(aProperty, aPropertyValue));
            } else {
                throw std::invalid_argument( "Not a problem property" );
            }
        }


        void ProblemManager::SetProblemProperty(ProblemProperty aProperty, double aPropertyValue) {
            if(mProblemProperties.find(aProperty) != mProblemProperties.end()) {
                this->mDoubleProblemProperties.insert(std::make_pair(aProperty, aPropertyValue));
            } else {
                throw std::invalid_argument( "Not a problem property" );
            }
        }


        void ProblemManager::SetProblemProperty(ProblemProperty aProperty, double *aPropertyValue) {
            if(mProblemProperties.find(aProperty) != mProblemProperties.end()) {
                this->mP2DoubleProblemProperties.insert(std::make_pair(aProperty, aPropertyValue));
            } else {
                throw std::invalid_argument( "Not a problem property" );
            }
        }


        void ProblemManager::SetProblemProperty(ProblemProperty aProperty, char aPropertyValue) {

            if(mProblemProperties.find(aProperty) != mProblemProperties.end()) {
                mCharProperties[aProperty] = aPropertyValue;
            } else {
                throw std::invalid_argument( "Not a problem property" );
            }
        }


        void ProblemManager::SetProblemProperty(ProblemProperty aProperty, char *aPropertyValue) {
            if(mProblemProperties.find(aProperty) != mProblemProperties.end()) {
                this->mStringProblemProperties.insert(std::make_pair(aProperty, aPropertyValue));
            } else {
                throw std::invalid_argument( "Not a problem property" );
            }
        }

    }
}