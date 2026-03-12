cmake_minimum_required(VERSION 3.0.2)
project(emplanner)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  pcl_ros
  geometry_msgs
  message_generation
  message_runtime
  planning_msgs
  perception_msgs
  
)
find_package(yaml-cpp REQUIRED)

find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)

string(REGEX REPLACE "/[^/]+/[^/]+/[^/]+/[^/]+$" "" PARENT_DIR ${PROJECT_SOURCE_DIR})
set(ONNXRUNTIME_DIR "/usr/local/onnxruntime")
set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_DIR}/include")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_DIR}/lib")

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/RL_DP
  ${PROJECT_SOURCE_DIR}
  # ${PROJECT_SOURCE_DIR}/../../../../devel/include
  ${PYTHON_INCLUDE_DIRS}
  ${ONNXRUNTIME_INCLUDE_DIR}
  ${PARENT_DIR}
)



# ## 配置 msg 源文件
# add_message_files(
#   FILES
#   car_path.msg #添加源文件
#   car_info.msg
#   path_point.msg
#   point.msg
#   Obstacle.msg
#   ObstacleList.msg
#   Pose2Dstamped.msg
# )
# generate_messages(
#   DEPENDENCIES
#   std_msgs 
# )

catkin_package(
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS  pcl_ros roscpp rospy sensor_msgs std_msgs geometry_msgs message_runtime message_generation planning_msgs perception_msgs
  DEPENDS yaml-cpp
)

add_library(${PROJECT_NAME}
  src/emplanner.cpp
  RL_DP/rl_dp.cpp
  RL_DP/dp_planner.cpp
  RL_DP/dp_policy.cpp
)
add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS} planning_msgs_generate_messages_cpp perception_msgs_generate_messages_cpp)
target_link_libraries(${PROJECT_NAME}
  PUBLIC
    ${catkin_LIBRARIES}
    ${PCL_LIBRARIES}
    yaml-cpp
    libosqp.so  
    libOsqpEigen.so 
    libqdldl.so
    ${PYTHON_LIBRARIES}
    ${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so
)
link_directories(${PCL_LIBRARY_DIRS})
link_directories(/usr/local/inclede/Eigen3)
link_directories(/usr/local/inclede/osqp-eigen)
link_directories(/usr/local/inclede/osqp)
link_directories(/usr/local/lib)
link_directories(${ONNXRUNTIME_LIB_DIR})

add_executable(emplanner_run src/main.cpp)
add_dependencies(emplanner_run ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS} planning_msgs_generate_messages_cpp perception_msgs_generate_messages_cpp) 
target_link_libraries(emplanner_run ${PROJECT_NAME})

add_executable(emplanner_compare_benchmark src/compare_benchmark.cpp)
add_dependencies(emplanner_compare_benchmark ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS} planning_msgs_generate_messages_cpp perception_msgs_generate_messages_cpp)
target_link_libraries(emplanner_compare_benchmark ${PROJECT_NAME})
