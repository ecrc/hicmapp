add_subdirectory(decomposer)

set(SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/ProblemManager.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/matrix.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/submatrix.cpp
        ${SOURCES}
        PARENT_SCOPE
        )
