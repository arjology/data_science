cmake_minimum_required(VERSION 3.10)

# set the project name and version
project(NN_cpp VERSION 1.0 LANGUAGES CXX)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_library(netlib src/Network/Network.cpp src/Network/Network.h)
add_executable(nn src/main.cpp src/Matrix.h)
target_link_libraries(nn PRIVATE netlib)

