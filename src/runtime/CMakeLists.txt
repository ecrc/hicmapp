if (HICMAPP_SCHED_LOWER STREQUAL "starpu")
    add_subdirectory(starpu)
elseif(HICMAPP_SCHED_LOWER STREQUAL "default")
    add_subdirectory(default)
endif()

set(SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/RunTimeSingleton.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/HicmaContext.cpp
        ${SOURCES}
        PARENT_SCOPE
        )