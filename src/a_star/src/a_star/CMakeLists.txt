cmake_minimum_required(VERSION 3.0.2)
project(a_star)


 add_compile_options(-std=c++14)


find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  pcl_conversions
  pcl_ros
  geometry_msgs
  planning_msgs
)
find_package(yaml-cpp REQUIRED)
catkin_package(
)


include_directories(
 include
  ${catkin_INCLUDE_DIRS}
)
link_directories(
  ${catkin_LIB_DIRS} lib
)


add_executable(a_star src/a_star.cpp)
target_link_libraries(a_star
   ${catkin_LIBRARIES}
   yaml-cpp
)
