cmake_minimum_required(VERSION 3.1)

add_subdirectory(libs)

project(aplusb)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(${PROJECT_NAME} src/main.cpp)
target_link_libraries(${PROJECT_NAME} libclew libutils)
