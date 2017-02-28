cmake_minimum_required(VERSION 2.8.3)
project(nn_controller)

find_package(PkgConfig)
pkg_check_modules( EIGEN3 REQUIRED eigen3 )
include_directories( ${EIGEN3_INCLUDE_DIRS} )


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(Caffe_DIR                     $ENV{CAFFE_ROOT})

OPTION(USE_CAFFE "Enable Caffe support" ON)
OPTION(USE_CAFFE_GPU "Enable Caffe GPU support" ON)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  roscpp
  rospy
  std_msgs
  message_generation
  ensenso
  Caffe
)

find_package(Boost COMPONENTS filesystem system serialization
thread REQUIRED)

# Generate messages in the 'msg' folder
add_message_files(
  FILES
  predictor.msg
)

## Generate services in the 'srv' folder
 add_service_files(
   FILES
   amfcError.srv
   controller.srv
   predictor_params.srv
 )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
 generate_messages(
   DEPENDENCIES
   geometry_msgs
   std_msgs
 )

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  CATKIN_DEPENDS geometry_msgs roscpp rospy std_msgs ensenso message_runtime 
  DEPENDS Boost
  LIBRARIES pathfinder
)

###########
## Build ##
###########

# Include Caffe
if (USE_CAFFE)
    # add definitions for the C++ code
    add_definitions(-DUSE_CAFFE)
    if (USE_CAFFE_GPU)
        add_definitions(-DCAFFE_GPU)
        find_package(CUDA 7.0 REQUIRED)
        include_directories(${CUDA_INCLUDE_DIRS})
    else (USE_CAFFE_GPU)
        add_definitions(-DCPU_ONLY)
    endif (USE_CAFFE_GPU)
    # add paths to CAFFE
    link_directories(${CAFFE_LIBRARY_PATH})  # Should be set to /path/to/caffe/build/lib
    include_directories(${CAFFE_INCLUDE_PATH})  # Should be set to /path/to/caffe/distribute/include
    # extra header paths
    include_directories(${CAFFE_EXTRA_PATH})
    # add neural network to DDP controller files
    #set(DDP_FILES ${DDP_FILES} src/neuralnetworkcaffe.cpp src/caffenncontroller.cpp)
    # compile Caffe test
    #rosbuild_add_executable(caffe_test src/caffe_test.cpp src/neural_network_caffe.cpp)
    #target_link_libraries(caffe_test caffe protobuf)

    # compile image processor node
    #rosbuild_add_executable(caffe_img_processor src/img_processor.cpp src/neural_network_caffe.cpp)
    #target_link_libraries(caffe_img_processor caffe protobuf)
endif (USE_CAFFE)

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(include)
include_directories(
  ${EIGEN3_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
)

add_definitions(${EIGEN_DEFINITIONS})

## Declare a C++ library
#add_library(nn_controller
#   src/nn_controller.cpp
# )

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
#add_dependencies(nn_controller ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})


## Declare a C++ executable
 add_executable(nn_controller src/nn_controller.cpp)
add_dependencies(nn_controller ${PROJECT_NAME}_gencpp)
add_dependencies(nn_controller ${catkin_EXPORTED_TARGETS})

## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(nn_controller_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

if (USE_CAFFE)
    target_link_libraries(nn_controller caffe protobuf)
endif (USE_CAFFE)

## Specify libraries to link a library or executable target against
 target_link_libraries(nn_controller
  ${catkin_LIBRARIES} ${Boost_FILESYSTEM_LIBRARY} -ldl /home/robotec/catkin_ws/devel/lib/libboostsender.so
 )

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
# install(TARGETS nn_controller nn_controller_node
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark cpp header files for installation
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
  PATTERN ".svn" EXCLUDE
)


## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_nn_controller.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
