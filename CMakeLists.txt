cmake_minimum_required(VERSION 2.8.12)
project(ackermann_msckf)

add_compile_options(-std=c++14)
set(DEFAULT_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-O3 -g")
if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE
            STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
            "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif ()

# Modify cmake module path if new .cmake files are required
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  tf
  nav_msgs
  sensor_msgs
  geometry_msgs
  eigen_conversions
  tf_conversions
  message_generation
  nodelet
  image_transport
  cv_bridge
  message_filters
  pcl_conversions
  pcl_ros
  std_srvs
  random_numbers
)

## System dependencies are found with CMake's conventions
find_package(Boost REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Sophus REQUIRED )
find_package(OpenCV REQUIRED)
find_package(SuiteSparse REQUIRED)

##################
## ROS messages ##
##################
add_message_files(
  FILES

  FeatureMeasurement.msg
  CameraMeasurement.msg
  TrackingInfo.msg

  AckermannDrive.msg 
  AckermannDriveStamped.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

###################################
## catkin specific configuration ##
###################################
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ackermann_msckf image_processor
  CATKIN_DEPENDS
    roscpp std_msgs tf nav_msgs sensor_msgs geometry_msgs
    eigen_conversions tf_conversions message_runtime
    image_transport cv_bridge message_filters pcl_conversions
    pcl_ros std_srvs random_numbers
  DEPENDS Boost EIGEN3 OpenCV SUITESPARSE
)

###########
## Build ##
###########

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${Boost_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${Sophus_INCLUDE_DIRS}
  ${SUITESPARSE_INCLUDE_DIRS}
)

add_library(ackermann_msckf
  src/ackermann_msckf.cpp
  src/utils.cpp
)
add_dependencies(ackermann_msckf
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(ackermann_msckf
  ${catkin_LIBRARIES}
  ${SUITESPARSE_LIBRARIES}
  ${Sophus_LIBRARIES}
)

# ackermann_msckf nodelet
add_library(ackermann_msckf_nodelet
  src/ackermann_msckf_nodelet.cpp
)
add_dependencies(ackermann_msckf_nodelet
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(ackermann_msckf_nodelet
  ackermann_msckf
  ${catkin_LIBRARIES}
  ${Sophus_LIBRARIES}
)

# Image processor
add_library(image_processor_ackermann
  src/image_processor_ackermann.cpp
  src/utils.cpp
)
add_dependencies(image_processor_ackermann
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(image_processor_ackermann
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${Sophus_LIBRARIES}
)

# Image processor nodelet
add_library(image_processor_ackermann_nodelet
  src/image_processor_ackermann_nodelet.cpp
)
add_dependencies(image_processor_ackermann_nodelet
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(image_processor_ackermann_nodelet
  image_processor_ackermann
  ${catkin_LIBRARIES}
  ${Sophus_LIBRARIES}
)

#############
## Install ##
#############

install(TARGETS
  ackermann_msckf ackermann_msckf_nodelet image_processor_ackermann image_processor_ackermann_nodelet
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  PATTERN "*_nodelet.h" EXCLUDE
)

install(FILES nodelets.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

#############
## Testing ##
#############

# if(CATKIN_ENABLE_TESTING)
#   # Feature initialization test
#   catkin_add_gtest(test_feature_init
#     test/feature_initialization_test.cpp
#   )
#   add_dependencies(test_feature_init
#     ${${PROJECT_NAME}_EXPORTED_TARGETS}
#     ${catkin_EXPORTED_TARGETS}
#   )
#   target_link_libraries(test_feature_init
#     ${catkin_LIBRARIES}
#   )

#   # Math utils test
#   catkin_add_gtest(test_math_utils
#     test/math_utils_test.cpp
#   )
# endif()