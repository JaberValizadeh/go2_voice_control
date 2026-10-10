cmake_minimum_required(VERSION 3.0.2)
project(go2_voice_control)

find_package(catkin REQUIRED COMPONENTS
  gazebo_msgs
  geometry_msgs
  rospy
  sensor_msgs
  std_msgs
)

catkin_python_setup()

catkin_package(CATKIN_DEPENDS gazebo_msgs geometry_msgs rospy sensor_msgs
                              std_msgs)

catkin_install_python(PROGRAMS
  scripts/voice_recognizer.py
  scripts/voice_teleop.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch worlds tools xacro
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

if(CATKIN_ENABLE_TESTING)
  # Pure-logic tests: vocabulary and turn maths, no ROS master required.
  catkin_add_nosetests(test/test_interpreter.py)
  catkin_add_nosetests(test/test_stairs.py)
  catkin_add_nosetests(test/test_perception.py)
  catkin_add_nosetests(test/test_stepper.py)
  catkin_add_nosetests(test/test_teleop_stairs.py)

  # Integration test: runs voice_teleop.py and drives it over topics.
  find_package(rostest REQUIRED)
  add_rostest(test/voice_teleop.test)
endif()
