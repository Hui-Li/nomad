cmake_minimum_required(VERSION 2.8)

project(nomad CXX)

set(CMAKE_CXX_FLAGS "-std=c++11 -DNDEBUG -O3 -Wall -fsigned-char")

# find mpi
find_package(MPI REQUIRED)

if (MPI_FOUND)
    message(STATUS "MPI found: " ${MPI_INCLUDE_PATH})
    set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})
    set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS}")
    include_directories(${MPI_INCLUDE_PATH})
    link_libraries(${MPI_LIBRARIES})
endif(MPI_FOUND)

#set(CMAKE_BUILD_TYPE Release)
# set(CMAKE_BUILD_TYPE Debug)


# find TBB

# Add FindTBB directory to CMake's module path
# This cmake module is from http://code.google.com/p/findtbb
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/CMake/FindTBB/")

#set(TBB_ROOT_DIR "/home/hli/Desktop/Workspace/nomad/tbb")
#set(TBB_INCLUDE_DIR "/home/hli/Desktop/Workspace/nomad/tbb/include/tbb")
#set(TBB_LIBRARY "/home/hli/Desktop/Workspace/nomad/tbb/lib/intel64/gcc4.7")

find_package(TBB REQUIRED)

if(TBB_FOUND)
    message(STATUS "TBB found: " ${TBB_INCLUDE_DIRS})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ltbb -ltbbmalloc")
    include_directories(${TBB_INCLUDE_DIRS})
    link_libraries(${TBB_LIBRARIES})
endif(TBB_FOUND)

# find Boost

find_package(Boost 1.49 COMPONENTS program_options REQUIRED)

if(Boost_FOUND)
    message(STATUS "Boost found: " ${Boost_INCLUDE_DIR})
    include_directories( ${Boost_INCLUDE_DIR})
    link_libraries(${Boost_LIBRARIES})
endif(Boost_FOUND)

message(STATUS "${CMAKE_CXX_FLAGS}")

#add_executable(nomad_float nomad_float.cpp)
add_executable(nomad_double nomad_double.cpp)
#add_executable(synthgen synthgen.cpp)
#add_executable(permute permute.cpp)
