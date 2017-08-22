cmake_minimum_required(VERSION 2.8.3)
project(farnn)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  sensor_msgs
  std_msgs
)

# find_package(Boost REQUIRED COMPONENTS system)


catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES farnn
  CATKIN_DEPENDS roscpp rospy sensor_msgs std_msgs torch-ros
#  DEPENDS system_lib
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)