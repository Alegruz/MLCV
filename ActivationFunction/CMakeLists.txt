cmake_minimum_required(VERSION 3.0.0)

include(CTest)
enable_testing()

add_library(ActivationFunction ActivationFunction.cpp)

target_link_libraries(ActivationFunction PUBLIC Tool)
target_link_libraries(ActivationFunction PUBLIC Image)

target_include_directories(ActivationFunction PUBLIC "${CMAKE_SOURCE_DIR}/Tool")
target_include_directories(ActivationFunction PUBLIC "${CMAKE_SOURCE_DIR}")
target_include_directories(ActivationFunction PUBLIC "${CMAKE_SOURCE_DIR}/Image")
target_link_directories(ActivationFunction PUBLIC "${CMAKE_SOURCE_DIR}/Image")

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED True)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

install(TARGETS ActivationFunction DESTINATION lib)
install(FILES ActivationFunction.h DESTINATION include)