cmake_minimum_required(VERSION 3.0.0)

include(CTest)
enable_testing()

add_library(LayerManager LayerManager.cpp)

target_link_libraries(LayerManager PUBLIC Tool)
target_link_libraries(LayerManager PUBLIC Layer)

target_include_directories(LayerManager PUBLIC "${CMAKE_SOURCE_DIR}/Tool")
target_include_directories(LayerManager PUBLIC "${CMAKE_SOURCE_DIR}/Layer")
target_link_directories(LayerManager PUBLIC "${CMAKE_SOURCE_DIR}/Layer")

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED True)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

install(TARGETS LayerManager DESTINATION lib)
install(FILES LayerManager.h DESTINATION include)