cmake_minimum_required(VERSION 2.8.3)
project(pose_graph)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
#-DEIGEN_USE_MKL_ALL")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    nav_msgs
    camera_model
    cv_bridge
    roslib
)

find_package(OpenCV REQUIRED)

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "~/source_builds/libtorch")
find_package(Torch REQUIRED)

find_package(Ceres REQUIRED)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(GTSAM REQUIRED)

find_package(Eigen3 REQUIRED)

find_package(Boost REQUIRED)

include_directories(${catkin_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${GTSAM_INCLUDE_DIR})
link_directories(${GTSAM_LIBRARY_DIRS})

catkin_package()

add_executable(pose_graph
    src/pose_graph_node.cpp
    src/pose_graph.cpp
    src/keyframe.cpp
    src/netvlad.cpp
    src/utility/CameraPoseVisualization.cpp
    src/ThirdParty/DBoW/BowVector.cpp
    src/ThirdParty/DBoW/FBrief.cpp
    src/ThirdParty/DBoW/FeatureVector.cpp
    src/ThirdParty/DBoW/QueryResults.cpp
    src/ThirdParty/DBoW/ScoringObject.cpp
    src/ThirdParty/DUtils/Random.cpp
    src/ThirdParty/DUtils/Timestamp.cpp
    src/ThirdParty/DVision/BRIEF.cpp
    src/ThirdParty/VocabularyBinary.cpp
)


target_link_libraries(pose_graph ${catkin_LIBRARIES} ${CERES_LIBRARIES} ${OpenCV_LIBS} ${TORCH_LIBRARIES} gtsam)
# message("catkin_lib  ${catkin_LIBRARIES}")


# cmake_minimum_required(VERSION 2.8.3)
# project(pose_graph)

# # list(APPEND CMAKE_PREFIX_PATH "/home/devapi_ubuntu_20_04_5_lts/source_builds/libtorch")#/share/cmake/Torch")
# # set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/home/devapi_ubuntu_20_04_5_lts/source_builds/libtorch")
# message("cmake prefix path0:- " ${CMAKE_PREFIX_PATH} "\n\n")

# set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_CXX_FLAGS "-std=c++14")
# #-DEIGEN_USE_MKL_ALL")
# set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

# #set(Torch_DIR "/home/devapi_ubuntu_20_04_5_lts/source_builds/libtorch")


# find_package(catkin REQUIRED COMPONENTS
#     roscpp
#     std_msgs
#     nav_msgs
#     camera_model
#     cv_bridge
#     roslib
#     )

# find_package(OpenCV REQUIRED)
# message("cmake prefix path1:- " ${CMAKE_PREFIX_PATH} "\n\n")
# set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "~/source_builds/libtorch")
# message("cmake prefix path2:- " ${CMAKE_PREFIX_PATH} "\n\n")
# find_package(Torch REQUIRED)

# find_package(Ceres REQUIRED)

# set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
# find_package(Eigen3)
# find_package(Boost REQUIRED)
# find_package(GTSAM REQUIRED)
# # list(APPEND CMAKE_PREFIX_PATH "/home/devapi_ubuntu_20_04_5_lts/.local/lib/python3.8/site-packages/torch/share/cmake")
# # list(APPEND CMAKE_PREFIX_PATH "/home/devapi_ubuntu_20_04_5_lts/source_builds/libtorch")


# include_directories(${catkin_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS}  ${EIGEN3_INCLUDE_DIR} ${GTSAM_INCLUDE_DIR})
# link_directories(${GTSAM_LIBRARY_DIRS})

# catkin_package()

# add_executable(pose_graph
#     src/pose_graph_node.cpp
#     src/pose_graph.cpp
#     src/keyframe.cpp
#     src/netvlad.cpp
#     src/utility/CameraPoseVisualization.cpp
#     src/ThirdParty/DBoW/BowVector.cpp
#     src/ThirdParty/DBoW/FBrief.cpp
#     src/ThirdParty/DBoW/FeatureVector.cpp
#     src/ThirdParty/DBoW/QueryResults.cpp
#     src/ThirdParty/DBoW/ScoringObject.cpp
#     src/ThirdParty/DUtils/Random.cpp
#     src/ThirdParty/DUtils/Timestamp.cpp
#     src/ThirdParty/DVision/BRIEF.cpp
#     src/ThirdParty/VocabularyBinary.cpp
#     )

# target_link_libraries(pose_graph ${catkin_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES} ${TORCH_LIBRARIES} gtsam) 
# # message("catkin_lib  ${catkin_LIBRARIES}")
