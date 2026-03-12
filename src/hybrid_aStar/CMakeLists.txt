cmake_minimum_required(VERSION 3.0.2)
project(hybrid_a_star)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set( CMAKE_BUILD_TYPE Release )
set(CMAKE_BUILD_TYPE RelWithDebInfo) 
message(STATUS "Hybrid A star")
message(STATUS "Author: Zhang Zhimeng")


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

# glog
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
    geometry_msgs
    visualization_msgs
)
find_package(yaml-cpp REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBZIP REQUIRED libzip)

set(ONNXRUNTIME_DIR "/usr/local/onnxruntime")
set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_DIR}/include")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_DIR}/lib")


include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/../../../../devel/include
    ${PCL_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${LIBZIP_INCLUDE_DIRS}
)

catkin_package(
    CATKIN_DEPENDS nav_msgs roscpp tf planning_msgs pcl_ros sensor_msgs geometry_msgs visualization_msgs
)

add_library(
    PATH_SEARCHER_LIB SHARED
    src/rs_path.cpp
    src/costmap_subscriber.cpp
    src/goal_pose_subscriber.cpp
    src/init_pose_subscriber.cpp
    src/hybrid_a_star.cpp
    src/hybrid_a_star_flow.cpp
    src/guided_frontend_onnx.cpp
    # src/smoother.cpp
)

link_directories(/usr/local/lib/Eigen3)
link_directories(/usr/local/lib/osqp-eigen)
link_directories(/usr/local/lib/osqp)
# link_directories(${PCL_LIBRARY_DIRS})
target_link_libraries(
    PATH_SEARCHER_LIB
    ${catkin_LIBRARIES}
    ${GLOG_LIBRARIES}
    ${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so
)
# add_dependencies(run_hybrid_astar ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS} planning_msgs_generate_messages_cpp) 
add_executable(run_hybrid_astar app/run_hybrid_astar.cpp)
# target_link_libraries(run_hybrid_astar PATH_SEARCHER_LIB ${PCL_LIBRARIES}  libosqp.so  
# libOsqpEigen.so 
# libqdldl.so
# libgflags.a libglog.a
# yaml-cpp )

target_link_libraries(run_hybrid_astar PATH_SEARCHER_LIB ${PCL_LIBRARIES}  libosqp.so  
libOsqpEigen.so 
libqdldl.so
gflags
glog 
yaml-cpp)

add_executable(random_parking_env app/random_parking_env.cpp)
target_link_libraries(random_parking_env
    ${catkin_LIBRARIES}
)

add_executable(frontend_random_map_env app/frontend_random_map_env.cpp)
target_link_libraries(frontend_random_map_env
    ${catkin_LIBRARIES}
)

add_executable(frontend_street_map_env_cpp app/frontend_street_map_env.cpp)
target_link_libraries(frontend_street_map_env_cpp
    ${catkin_LIBRARIES}
    ${LIBZIP_LIBRARIES}
)

add_executable(smooth_path_cli app/smooth_path_cli.cpp)
target_link_libraries(smooth_path_cli
    PATH_SEARCHER_LIB
    ${PCL_LIBRARIES}
    libosqp.so
    libOsqpEigen.so
    libqdldl.so
    gflags
    glog
    yaml-cpp
)

add_executable(offline_guided_astar_cpp_demo app/offline_guided_astar_cpp_demo.cpp)
target_compile_definitions(offline_guided_astar_cpp_demo PRIVATE HYBRID_ASTAR_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
target_link_libraries(offline_guided_astar_cpp_demo
    PATH_SEARCHER_LIB
    ${PCL_LIBRARIES}
    libosqp.so
    libOsqpEigen.so
    libqdldl.so
    gflags
    glog
    yaml-cpp
    ${LIBZIP_LIBRARIES}
)
