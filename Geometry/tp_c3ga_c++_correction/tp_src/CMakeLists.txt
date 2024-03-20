cmake_minimum_required(VERSION 3.13)

project(ga_view)

# set the module path for "findLib.cmake"
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/modules/")

# require Eigen
find_package(Eigen3 REQUIRED)
if(${EIGEN3_FOUND})
    message(STATUS "lib EIGEN3 found")
    message(STATUS "  version " ${EIGEN3_VERSION_STRING})
    message(STATUS "  include " ${EIGEN3_INCLUDE_DIR})
endif()

# compilation flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O2 -std=c++14")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")


# files to compile
file(GLOB_RECURSE source_files src/*.cpp src/*.hpp)
message(STATUS "  files " ${source_files})

# compile
add_executable(ga_view ${source_files})
#add_executable(ga_view src/Directory.cpp src/Directory.hpp src/Entry.hpp src/Entry.cpp src/main.cpp )

# includes
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${C3GA_INCLUDE_DIRS})

# link
target_link_libraries(ga_view ${C3GA_LIBRARIES})




