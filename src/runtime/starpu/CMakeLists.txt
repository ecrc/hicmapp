add_subdirectory(starpu-codelets)
set(SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/starpu.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/factory.cpp
        ${SOURCES}
        PARENT_SCOPE
        )
