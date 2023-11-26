#include <unordered_map>
#include <hicmapp/runtime/interface/StarpuCodelet.hpp>
#include "hicmapp/common/definitions.h"

#ifndef HICMAPP_FACTORY_HPP
#define HICMAPP_FACTORY_HPP
namespace hicmapp {
    namespace runtime {

        /***
         * Types of Supported Codelets
         */
        enum CodeletType {
            GENERATE_DENSE_DATA,
            GENERATE_COMPRESSED_DATA,
            GEMM,
            SYRK,
            POTRF,
            TRSM,
            DGYTLR,
            DGYTLR_DIAG,
            UNCOMPRESS,
            LACPY
        };

        template<typename T>
        class MakerInterface {
        public:
            MakerInterface() = default;

            virtual StarpuCodelet *CreateObject() = 0;

            virtual ~MakerInterface() = default;
        };

        template<typename T>
        class CodeletFactory {

        public:

            static void RegisterMaker(CodeletType aType, MakerInterface<T> *aMaker);

            static StarpuCodelet *CreateCodelet(CodeletType aType);

        private:
            static std::unordered_map<CodeletType, MakerInterface<T> *> &GetMakersMap();

            static std::unordered_map<CodeletType, StarpuCodelet *> &GetCodeletsMap();
        };

        template<template<typename T> typename Object, typename T>
        class CodeletMaker : public MakerInterface<T> {
        public:

            explicit CodeletMaker(const CodeletType aType) noexcept {
                CodeletFactory<T>::RegisterMaker(aType, this);
            }

            StarpuCodelet *CreateObject() override {
                return new Object<T>();
            }

            ~CodeletMaker() = default;
        };
    }

}

#endif //HICMAPP_FACTORY_HPP
