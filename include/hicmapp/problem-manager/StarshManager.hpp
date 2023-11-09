
#ifndef HICMAPP_TILE_OPERATIONS_STARSH_MANAGER_HPP
#define HICMAPP_TILE_OPERATIONS_STARSH_MANAGER_HPP

#include <hicmapp/common/definitions.h>
#include <hicmapp/primitives/ProblemManager.hpp>

#include <starsh.h>

namespace hicmapp {
    namespace operations {
        class StarsHManager {
        public:

            static void DestroyStarsHManager();

            static STARSH_blrf *GetStarsHFormat();

            static void SetStarsHFormat(primitives::ProblemManager &aProblemManager);

        private:
            StarsHManager();
            static STARSH_blrf *starsh_format;
        };
    }
}

#endif //HICMAPP_TILE_OPERATIONS_STARSH_MANAGER_HPP
