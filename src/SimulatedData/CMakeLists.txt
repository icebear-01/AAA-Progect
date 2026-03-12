cmake_minimum_required(VERSION 3.0.2)
project(simulated_data)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set( CMAKE_BUILD_TYPE Release )
set(CMAKE_BUILD_TYPE RelWithDebInfo) 


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

# message(STATUS "Parent directory of the project: ${PROJECT_PARENT_DIR}/emplanner")

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
    SIM_SEARCHER_LIB SHARED

    ${PROJECT_SOURCE_DIR}/src/simulated_data.cpp
    # src/smoother.cpp
)

target_link_libraries(
    SIM_SEARCHER_LIB
    ${catkin_LIBRARIES}
)
# add_dependencies(run_hybrid_astar ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS} planning_msgs_generate_messages_cpp) 
add_executable(sim_data src/main.cpp)
# target_link_libraries(run_hybrid_astar PATH_SEARCHER_LIB ${PCL_LIBRARIES}  libosqp.so  
# libOsqpEigen.so 
# libqdldl.so
# libgflags.a libglog.a
# yaml-cpp )

target_link_libraries(sim_data SIM_SEARCHER_LIB ${PCL_LIBRARIES}  
yaml-cpp)