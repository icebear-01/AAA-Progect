cmake_minimum_required(VERSION 3.16)

project(hope_env LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(HOPE_ENV_BUILD_APPS "Build example apps" ON)

find_package(Boost REQUIRED)
find_package(OpenCV REQUIRED)

add_library(hope_env
    src/configs.cc
    src/geometry.cc
    src/random_utils.cc
    src/vehicle.cc
    src/parking_map_normal.cc
    src/parking_map_dlp.cc
    src/map_level.cc
    src/lidar_simulator.cc
    src/action_mask.cc
    src/reeds_shepp.cc
    src/env.cc
)

target_include_directories(hope_env PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(hope_env PUBLIC Boost::boost ${OpenCV_LIBS})

target_compile_definitions(hope_env PUBLIC HOPE_USE_OPENCV=1)

if(HOPE_ENV_BUILD_APPS)
    add_executable(hope_preview apps/preview_envs.cpp)
    target_link_libraries(hope_preview PRIVATE hope_env)
endif()
