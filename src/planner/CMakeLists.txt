cmake_minimum_required(VERSION 3.0.2)
project(planner)

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

# Eigen
include(cmake/FindEigen.cmake)
include_directories(${EIGEN_INCLUDE_DIR})
 
include(cmake/glog.cmake)
include_directories(${GLOG_INCLUDE_DIRS})

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
set(ONNXRUNTIME_DIR "/usr/local/onnxruntime")
set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_DIR}/include")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_DIR}/lib")

# message(STATUS "Parent directory of the project: ${PROJECT_PARENT_DIR}/emplanner")

include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR} 
    ${PROJECT_PARENT_DIR}/emplanner
    ${PROJECT_PARENT_DIR}/emplanner/RL_DP
    ${PROJECT_PARENT_DIR}/hybrid_aStar/include
    ${PROJECT_PARENT_DIR}/hybrid_aStar
    ${PROJECT_PARENT_DIR}/FixedRouteplanner

    ${PCL_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIR}
)

catkin_package(
    #  INCLUDE_DIRS include
    #  LIBRARIES path_searcher
    #  CATKIN_DEPENDS nav_mags roscpp tf
    #  DEPENDS system_lib
)

add_library(
    PLAN_SEARCHER_LIB SHARED
    ${PROJECT_PARENT_DIR}/hybrid_aStar/src/rs_path.cpp
    ${PROJECT_PARENT_DIR}/hybrid_aStar/src/costmap_subscriber.cpp
    ${PROJECT_PARENT_DIR}/hybrid_aStar/src/goal_pose_subscriber.cpp
    ${PROJECT_PARENT_DIR}/hybrid_aStar/src/init_pose_subscriber.cpp
    ${PROJECT_PARENT_DIR}/hybrid_aStar/src/hybrid_a_star.cpp
    ${PROJECT_PARENT_DIR}/hybrid_aStar/src/hybrid_a_star_flow.cpp
    ${PROJECT_PARENT_DIR}/emplanner/src/emplanner.cpp
    ${PROJECT_PARENT_DIR}/emplanner/RL_DP/rl_dp.cpp
    ${PROJECT_PARENT_DIR}/emplanner/RL_DP/dp_planner.cpp
    ${PROJECT_PARENT_DIR}/emplanner/RL_DP/dp_policy.cpp
    ${PROJECT_PARENT_DIR}/FixedRouteplanner/src/rtk_planner.cpp
    ${PROJECT_SOURCE_DIR}/src/planner.cpp 
    # src/smoother.cpp
)

link_directories(/usr/local/lib/Eigen3)
link_directories(/usr/local/lib/osqp-eigen)
link_directories(/usr/local/lib/osqp)
link_directories(${ONNXRUNTIME_LIB_DIR})
# link_directories(${PCL_LIBRARY_DIRS})
target_link_libraries(
    PLAN_SEARCHER_LIB
    ${catkin_LIBRARIES}
    ${GLOG_LIBRARIES}
    ${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so
)
# add_dependencies(run_hybrid_astar ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS} planning_msgs_generate_messages_cpp) 
add_executable(planner src/main.cpp)
# target_link_libraries(run_hybrid_astar PATH_SEARCHER_LIB ${PCL_LIBRARIES}  libosqp.so  
# libOsqpEigen.so 
# libqdldl.so
# libgflags.a libglog.a
# yaml-cpp )

target_link_libraries(planner PLAN_SEARCHER_LIB ${PCL_LIBRARIES}  libosqp.so  
libOsqpEigen.so 
libqdldl.so
gflags
glog 
yaml-cpp)
