cmake_minimum_required(VERSION 3.0.2)
project(emplanner_rviz_sim)

add_compile_options(-std=c++14)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  geometry_msgs
  planning_msgs
  emplanner
  roslib
)

catkin_package(
  CATKIN_DEPENDS roscpp std_msgs geometry_msgs planning_msgs emplanner roslib
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/../emplanner
  ${PROJECT_SOURCE_DIR}/../emplanner/include
  ${PYTHON_INCLUDE_DIRS}
  ${PARENT_DIR}
)

add_executable(rviz_sim_node src/rviz_sim_node.cpp)
add_dependencies(rviz_sim_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(rviz_sim_node ${catkin_LIBRARIES} emplanner)
