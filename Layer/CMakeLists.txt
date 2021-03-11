cmake_minimum_required(VERSION 3.0.0)

include(CTest)
enable_testing()

add_library(Layer Layer.cpp)

target_link_libraries(Layer PUBLIC Tool)
target_link_libraries(Layer PUBLIC ActivationFunction)
target_link_libraries(Layer PUBLIC Filter)
target_link_libraries(Layer PUBLIC Image)

target_include_directories(Layer PUBLIC "${CMAKE_SOURCE_DIR}/Tool")
target_include_directories(Layer PUBLIC "${CMAKE_SOURCE_DIR}/ActivationFunction")
target_include_directories(Layer PUBLIC "${CMAKE_SOURCE_DIR}/Filter")
target_include_directories(Layer PUBLIC "${CMAKE_SOURCE_DIR}/Image")
target_link_directories(Layer PUBLIC "${CMAKE_SOURCE_DIR}/ActivationFunction")
target_link_directories(Layer PUBLIC "${CMAKE_SOURCE_DIR}/Image")
target_link_directories(Layer PUBLIC "${CMAKE_SOURCE_DIR}/Filter")

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED True)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

install(TARGETS Layer DESTINATION lib)
install(FILES Layer.h DESTINATION include)