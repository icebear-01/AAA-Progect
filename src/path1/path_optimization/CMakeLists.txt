cmake_minimum_required(VERSION 3.0.2)
project(optimizetion_path)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

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
  
)
find_package(yaml-cpp REQUIRED)

find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/include
  # ${PROJECT_SOURCE_DIR}/../../../../devel/include
  ${PYTHON_INCLUDE_DIRS}
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
  CATKIN_DEPENDS  pcl_ros roscpp rospy sensor_msgs std_msgs geometry_msgs message_runtime message_generation
)
add_library(${PROJECT_NAME}
src/path_optimization.cpp
)

link_directories(${PCL_LIBRARY_DIRS})
link_directories(/usr/local/lib/Eigen3)
link_directories(/usr/local/lib/osqp-eigen)
link_directories(/usr/local/lib/osqp)


add_executable(optimize_path src/main.cpp)
add_dependencies(optimize_path ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS}) 
target_link_libraries(optimize_path  ${catkin_LIBRARIES} ${PCL_LIBRARIES} yaml-cpp
libosqp.so  
libOsqpEigen.so 
libqdldl.so
${PYTHON_LIBRARIES}
)