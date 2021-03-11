cmake_minimum_required(VERSION 3.0.0)

include(CTest)
enable_testing()

add_library(Image Image.cpp)

target_link_libraries(Image PUBLIC Tool)

target_include_directories(Image PUBLIC "${CMAKE_SOURCE_DIR}/Tool")
target_include_directories(Image PUBLIC "${CMAKE_SOURCE_DIR}")

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED True)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

install(TARGETS Image DESTINATION lib)
install(FILES Image.h DESTINATION include)