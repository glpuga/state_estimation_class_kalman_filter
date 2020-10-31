cmake_minimum_required (VERSION 2.6)
project(depthEstimator)

set(CMAKE_BUILD_TYPE release) 

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

#find_package(OpenCV2.4.13 REQUIRED)
find_package(OpenCV REQUIRED )

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

include_directories("implementacion")
include_directories(
        ${PROJECT_SOURCE_DIR}/thirdparty/Sophus
        )

file(GLOB DepthEstimator_SRCS "implementacion/*.cpp"
                              "implementacion/CostVolume/*.cpp"
                              "implementacion/Common/*.cpp"
                              "implementacion/Utils/*.cpp"
)

file(GLOB TESTAPP_SRCS "implementacion/test/*.cpp"
                       "implementacion/test/utils/*.cpp")

# build shared library.
add_library(depthEstimator SHARED ${DepthEstimator_SRCS})
target_link_libraries(depthEstimator ${OpenCV_LIBS})

# build example app
add_executable (testapp implementacion/test/main.cpp ${TESTAPP_SRCS})
target_link_libraries(testapp depthEstimator ${OpenCV_LIBS})
