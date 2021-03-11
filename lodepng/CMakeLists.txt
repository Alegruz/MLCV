cmake_minimum_required(VERSION 3.0.0)

include(CTest)
enable_testing()

add_library(lodepng lodepng.cpp)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED True)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

install(TARGETS lodepng DESTINATION lib)
install(FILES lodepng.h DESTINATION include)