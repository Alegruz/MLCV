cmake_minimum_required(VERSION 3.0.0)

include(CTest)
enable_testing()

add_library(DataReader DataReader.cpp)

target_link_libraries(DataReader PUBLIC Tool)
target_link_libraries(DataReader PUBLIC lodepng)
target_link_libraries(DataReader PUBLIC Image)

target_include_directories(DataReader PUBLIC "${CMAKE_SOURCE_DIR}/Tool")
target_include_directories(DataReader PUBLIC "${CMAKE_SOURCE_DIR}")
target_include_directories(DataReader PUBLIC "${CMAKE_SOURCE_DIR}/lodepng")
target_include_directories(DataReader PUBLIC "${CMAKE_SOURCE_DIR}/Image")
target_link_directories(DataReader PUBLIC "${CMAKE_SOURCE_DIR}/lodepng")
target_link_directories(DataReader PUBLIC "${CMAKE_SOURCE_DIR}/Image")

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED True)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

install(TARGETS DataReader DESTINATION lib)
install(FILES DataReader.h DESTINATION include)