cmake_minimum_required(VERSION 3.0.2)
project(velo2cam_calibration)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -w ")

#find_package(OpenCV REQUIRED)
#set(OpenCV_DIR ~/Downloads/opencv/opencv/build)
#message(OPENCV_DIR)
#message("${OPENCV_DIR}")
#find_package(OpenCV REQUIRED)
#include_directories(${OpenCV_INCLUDE_DIRS})
#message(OpenCV_LIBRARIES="${OpenCV_LIBRARIES}")
#message(OpenCV_DIR="${OpenCV_DIR}")
#message(OpenCV_INCLUDE_DIRS="${OpenCV_INCLUDE_DIRS}")
#SET(OpenCV_DIR /home/rdv/Downloads/opencv/opencv/build)
#set(opencv_contrib ON)

# set(OpenCV_DIR /home/rdv/Downloads/opencv/opencv/build_4.2.0)
# set(OpenCV_LIBRARIES /home/rdv/Downloads/opencv/opencv/build_4.2.0/lib)


# find_package(OpenCV 4.2.0 REQUIRED)
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
#SET(cv_bridge_DIR /opt/ros/noetic/share/cv_bridge/cmake/)
#pkg-config --modversion opencv
#sudo apt-get purge libopencv* python-opencv
#set(OpenCV_LIBRARIES ${OpenCV_LIBS})

#export PATH=/home/rdv/Downloads/opencv/opencv/build/bin/:$PATH
#export LD_LIBRARY_PATH=/home/rdv/Downloads/opencv/opencv/build/lib/:$LD_LIBRARY_PATH
#cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D INSTALL_C_EXAMPLES=ON -D INSTALL_PYTHON_EXAMPLES=ON -D WITH_TBB=ON -D WITH_V4L=ON -D OPENCV_EXTRA_MODULES_PATH=~/Downloads/opencv/opencv_contrib/modules -D OPENCV_ENABLE_NONFREE=ON -D WITH_CUDA=OFF -D BUILD_EXAMPLES=ON ..

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_geometry
  image_transport
  message_filters
  roscpp
  pcl_ros
  sensor_msgs
  std_msgs
  stereo_msgs
  tf
  tf_conversions
  dynamic_reconfigure
  cmake_modules
)

find_package(TinyXML REQUIRED)

generate_dynamic_reconfigure_options(
  cfg/Lidar.cfg
  cfg/Stereo.cfg
  cfg/Monocular.cfg
  cfg/Plane.cfg
)

################################################
## Declare ROS messages, services and actions ##
################################################
## Generate messages in the 'msg' folder
add_message_files(
  FILES
  ClusterCentroids.msg
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
)

###################################
## catkin specific configuration ##
###################################
catkin_package(
  INCLUDE_DIRS include
  DEPENDS TinyXML
)


###########
## Build ##
###########
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS} #${OpenCV_LIBRARIES}
  ${TinyXML_INCLUDE_DIRS}
)

#link_directories()

#add_definitions()

## Declare cpp executables
add_executable(lidar_pattern src/lidar_pattern.cpp)
add_executable(stereo_pattern src/stereo_pattern.cpp)
add_executable(mono_qr_pattern src/mono_qr_pattern.cpp)
add_executable(velo2cam_calibration src/velo2cam_calibration.cpp)
add_executable(v2c_plane_segmentation src/plane.cpp)
add_executable(v2c_disp_masker src/disp_masker.cpp)

## Add cmake target dependencies of the executable/library
## as an example, message headers may need to be generated before nodes
add_dependencies(lidar_pattern
  ${catkin_EXPORTED_TARGETS}
  velo2cam_calibration_generate_messages_cpp
  ${PROJECT_NAME}_gencfg
)
add_dependencies(stereo_pattern
  ${catkin_EXPORTED_TARGETS}
  velo2cam_calibration_generate_messages_cpp
  ${PROJECT_NAME}_gencfg
)
add_dependencies(mono_qr_pattern
  ${catkin_EXPORTED_TARGETS}
  velo2cam_calibration_generate_messages_cpp
  ${PROJECT_NAME}_gencfg
)
add_dependencies(
  velo2cam_calibration
  ${catkin_EXPORTED_TARGETS}
  velo2cam_calibration_generate_messages_cpp
  ${PROJECT_NAME}_gencfg
)
add_dependencies(v2c_plane_segmentation ${PROJECT_NAME}_gencfg)

## Specify libraries to link a library or executable target against
target_link_libraries(stereo_pattern
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS} # ${OpenCV_LIBRARIES}
)
target_link_libraries(lidar_pattern
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS} # ${OpenCV_LIBRARIES}
)
target_link_libraries(mono_qr_pattern
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS} # ${OpenCV_LIBRARIES}
)
target_link_libraries(velo2cam_calibration
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS} # ${OpenCV_LIBRARIES}
  ${TinyXML_LIBRARIES}
)
target_link_libraries(v2c_plane_segmentation
  ${catkin_LIBRARIES}
)
target_link_libraries(v2c_disp_masker
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS} # ${OpenCV_LIBRARIES}
)
