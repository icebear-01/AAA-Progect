cmake_minimum_required(VERSION 3.0.2)
project(rtkplanner)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set( CMAKE_BUILD_TYPE Release )
set(CMAKE_BUILD_TYPE RelWithDebInfo) 


#set(CMAKE_VERBOSE_MAKEFILE "true")

# Compiler
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)

if (COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif (COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else ()
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_FLAGS} doesn't have C++11 support.
                         Please use a different C++ compiler")
endif ()

set(ADDITIONAL_CXX_FLAG "-Wall -O3 -march=native")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ADDITIONAL_CXX_FLAG}")

# catkin
find_package(
    catkin REQUIRED COMPONENTS
    nav_msgs
    roscpp
    tf
    planning_msgs
    pcl_ros
    sensor_msgs
)
find_package(yaml-cpp REQUIRED)

get_filename_component(PROJECT_PARENT_DIR ${PROJECT_SOURCE_DIR} DIRECTORY)


include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR} 

    ${PCL_INCLUDE_DIRS}
)

catkin_package(
    #  INCLUDE_DIRS include
    #  LIBRARIES path_searcher
    #  CATKIN_DEPENDS nav_mags roscpp tf
    #  DEPENDS system_lib
)

add_library(
    RTKPLANNER_LIB SHARED
    ${PROJECT_SOURCE_DIR}/src/rtk_planner.cpp
    # ${PROJECT_PARENT_DIR}/hybrid_aStar/src/rs_path.cpp
)

# link_directories(${PCL_LIBRARY_DIRS})
target_link_libraries(
    RTKPLANNER_LIB
    ${catkin_LIBRARIES}
)
add_dependencies(${catkin_EXPORTED_TARGETS} planning_msgs_generate_messages_cpp) 
add_executable(rtkplanner_run  src/main.cpp)

target_link_libraries(rtkplanner_run RTKPLANNER_LIB ${PCL_LIBRARIES} 
yaml-cpp)