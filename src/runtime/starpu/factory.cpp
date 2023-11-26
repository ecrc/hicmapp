#include <hicmapp/runtime/concrete/starpu/factory.hpp>
#include <iostream>
#include "hicmapp/common/definitions.h"
#ifdef USE_OMP
#include <omp.h>
#endif

namespace hicmapp {
    namespace runtime {

        template<typename T>
        std::unordered_map<CodeletType, MakerInterface<T> *> &CodeletFactory<T>::GetMakersMap() {
            static auto mMakers = std::unordered_map<CodeletType, MakerInterface<T> *>();
            return mMakers;
        }

        template<typename T>
        std::unordered_map<CodeletType, StarpuCodelet *> &CodeletFactory<T>::GetCodeletsMap() {
            static auto mCodelets = std::unordered_map<CodeletType, StarpuCodelet *>();
            return mCodelets;

        }

        template<typename T>
        void CodeletFactory<T>::RegisterMaker(CodeletType aType, MakerInterface<T> *aMaker) {
            if (GetMakersMap().find(aType) == GetMakersMap().end()) {
                GetMakersMap().insert(std::make_pair(aType, aMaker));
            } else {
                std::cout << "The Maker passed is already registered. \n";
            }
        }

        template<typename T>
        StarpuCodelet *CodeletFactory<T>::CreateCodelet(CodeletType aType) {
            auto &makers_map = GetMakersMap();
            auto &codelets_map = GetCodeletsMap();

            if (makers_map.find(aType) != makers_map.end()) {
                if (codelets_map.find(aType) == codelets_map.end()) {
                    codelets_map[aType] = makers_map[aType]->CreateObject();
                }
                return codelets_map[aType];
            } else {
                throw std::runtime_error("Requested codelet does not have a registered maker");
            }
        }

        HICMAPP_INSTANTIATE_CLASS(CodeletFactory)
    }
}