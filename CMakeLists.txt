cmake_minimum_required(VERSION 3.0.2)
project(person_detection_robocup)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  geometry_msgs
  message_filters
  rospy
  sensor_msgs
  std_msgs
  clf_person_recognition_msgs
)

catkin_python_setup()

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES person_detection_robocup
#  CATKIN_DEPENDS cv_bridge geometry_msgs message_filters rospy sensor_msgs std_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

#############
## Install ##
#############

# Mark executable scripts (Python etc.) for installation
# in contrast to setup.py, you can choose the destination
catkin_install_python(PROGRAMS
  src/person_detection_robocup/detection_node.py
  src/person_detection_robocup/template_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY models templates launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)



