cmake_minimum_required(VERSION 3.0.2)
project(record_path)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
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
)

find_package(yaml-cpp REQUIRED)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/include

  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/../../../../devel/include
)

# add_message_files(
#   FILES
#   car_path.msg #添加源文件
# )
# generate_messages(
#   DEPENDENCIES
#   std_msgs 
# )

catkin_package(
  CATKIN_DEPENDS  pcl_ros roscpp rospy sensor_msgs std_msgs geometry_msgs message_generation message_runtime
)


add_library(${PROJECT_NAME}
  src/record_path.cpp
)
link_directories(${PCL_LIBRARY_DIRS})


add_executable(record_path1 src/main.cpp src/record_path.cpp)
add_dependencies(record_path1 ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS} planning_msgs_generate_messages_cpp perception_msgs_generate_messages_cpp)
target_link_libraries(record_path1  ${catkin_LIBRARIES} ${PCL_LIBRARIES} yaml-cpp)