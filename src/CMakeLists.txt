add_subdirectory(primitives)
add_subdirectory(matrix-operations)
add_subdirectory(runtime)
add_subdirectory(problem-manager)
add_subdirectory(tile-operations)
add_subdirectory(api)
add_subdirectory(utils)

set(LIB_NAME ${PROJECT_NAME})

add_library(${LIB_NAME}
        STATIC
        ${SOURCES}
        )

target_link_libraries(${LIB_NAME} ${LIBS})
target_compile_definitions(${LIB_NAME} PUBLIC ${COMPILE_DEFINITIONS})

## Installation steps for headers and libraries.
set_target_properties(${LIB_NAME}
        PROPERTIES
        VERSION ${${PROJECT_NAME}_VERSION}
        )
target_include_directories(${LIB_NAME}
        PUBLIC
        "$<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include>"
        )
install(TARGETS ${LIB_NAME}
        EXPORT ${LIB_NAME}CoreConfig
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        )
install(EXPORT ${LIB_NAME}CoreConfig DESTINATION lib/cmake/${PROJECT_NAME})

