cmake_minimum_required(VERSION 3.0.0)

include(CTest)
enable_testing()

add_library(Filter Filter.cpp)

target_link_libraries(Filter PUBLIC Tool)

target_include_directories(Filter PUBLIC "${CMAKE_SOURCE_DIR}/Tool")
target_include_directories(Filter PUBLIC "${CMAKE_SOURCE_DIR}")

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED True)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

install(TARGETS Filter DESTINATION lib)
install(FILES Filter.h DESTINATION include)