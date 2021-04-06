cmake_minimum_required(VERSION 3.0.2)
project(car_racing_dev)


find_package(catkin REQUIRED COMPONENTS
  message_generation
  message_runtime
  geometry_msgs
  roscpp
  rospy
  std_msgs
)

 add_message_files(
   FILES
   VehicleControl.msg
   VehicleStateCurv.msg
   VehicleStateGlob.msg
   VehicleState.msg
 )

 generate_messages(
   DEPENDENCIES
   geometry_msgs
   std_msgs
 )

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES car_racing_dev
#  CATKIN_DEPENDS geometry_msgs roscpp rospy std_msgs
#  DEPENDS system_lib
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)