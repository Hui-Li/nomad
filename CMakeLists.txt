cmake_minimum_required(VERSION 2.6)

project(nomad CXX)

# Project version
set(NOMAD_MAJOR 1)
set(NOMAD_MINOR 0)

# find mpi
find_package(MPI REQUIRED)

set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})
# BUGBUG: is this quotation mark really needed? if I don't put it, it generates semicolons
set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS}")
include_directories(${MPI_INCLUDE_PATH})
link_libraries(${MPI_LIBRARIES})

set(CMAKE_CXX_FLAGS_DEBUG "-std=c++11 -g -Wall -fsigned-char -D TBB_USE_THREADING_TOOLS")
set(CMAKE_CXX_FLAGS_RELEASE "-std=c++11 -O3 -Wall -fsigned-char")

set(CMAKE_BUILD_TYPE Release)
# set(CMAKE_BUILD_TYPE Debug)


# Locate TBB

# Add FindTBB directory to CMake's module path
# This cmake module is from http://code.google.com/p/findtbb
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/CMake/FindTBB/")

find_package(TBB REQUIRED)

include_directories(${TBB_INCLUDE_DIRS})
link_libraries(${TBB_LIBRARIES})

# Locate Boost
find_package(Boost 1.49 COMPONENTS program_options REQUIRED )
include_directories( ${Boost_INCLUDE_DIR} )
link_libraries(${Boost_LIBRARIES} )


add_executable(nomad_float nomad_float.cpp)
add_executable(nomad_double nomad_double.cpp)
add_executable(synthgen synthgen.cpp)
add_executable(permute permute.cpp)
