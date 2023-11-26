#include <iostream>
#include <starpu.h>

#ifndef HICMAPP_STARPUCODELET_HPP
#define HICMAPP_STARPUCODELET_HPP

using namespace std;

namespace hicmapp::runtime {

    /** Parent Abstract Class for all StarPu Codelets */
        class StarpuCodelet {

        public:

            /** Getter for starpu_codelet */
            virtual starpu_codelet *GetCodelet() = 0;

            /***
             * StarPuCodelet Destructor
             */
            virtual ~StarpuCodelet() = default;

        private:

        };
    }
#endif //HICMAPP_STARPUCODELET_HPP
