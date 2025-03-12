cmake_minimum_required(VERSION 3.0.2)
project(person_detection_robocup)

find_package(catkin REQUIRED COMPONENTS
  dynamic_reconfigure
  cv_bridge
  geometry_msgs
  message_filters
  rospy
  sensor_msgs
  std_msgs
  clf_person_recognition_msgs
  people_msgs
)

catkin_python_setup()

## Gencfg
generate_dynamic_reconfigure_options(
  cfg/Tracker.cfg
)

catkin_package(
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



