cmake_minimum_required(VERSION 3.12)
project(onnx_infer_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 直接在此指定 onnxruntime 安装根目录（包含 include/ 与 lib 或 lib64）。
# 如需调整，修改下行路径即可，构建时无需再传 -DONNXRUNTIME_DIR。
set(ONNXRUNTIME_DIR "/home/wmd/onnxruntime" CACHE PATH "Path to onnxruntime installation (with include/ and lib/)" FORCE)

set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_DIR}/include")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_DIR}/lib")
if(NOT EXISTS "${ONNXRUNTIME_LIB_DIR}")
    set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_DIR}/lib")
endif()

set(ONNXRUNTIME_DIR "/usr/local/onnxruntime" CACHE PATH "Path to onnxruntime installation (with include/ and lib/)" FORCE)

set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_DIR}/include")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_DIR}/lib")

include_directories(
    ${ONNXRUNTIME_INCLUDE_DIR}
)

include_directories(
  ${PROJECT_SOURCE_DIR}/include
  /usr/local/cuda/include
  /usr/local/onnxruntime
  /usr/local/onnxruntime/include
  ${catkin_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/../../../../devel/include
)

link_directories(
    ${ONNXRUNTIME_LIB_DIR}
)

option(USE_MATPLOTLIBCPP "Enable matplotlibcpp visualization (requires Python.h)" OFF)

add_library(dp_policy STATIC dp_policy.cpp)
target_link_libraries(dp_policy PRIVATE onnxruntime)

add_library(dp_infer STATIC dp_infer.cpp)
target_link_libraries(dp_infer PRIVATE onnxruntime)
target_link_libraries(dp_infer PRIVATE /usr/local/onnxruntime/lib/libonnxruntime.so)

add_library(dp_planner STATIC dp_planner.cpp)
target_link_libraries(dp_planner PRIVATE dp_policy onnxruntime)
target_link_libraries(dp_planner PRIVATE /usr/local/onnxruntime/lib/libonnxruntime.so)

add_library(rl_dp STATIC rl_dp.cpp)
target_link_libraries(rl_dp PRIVATE dp_policy dp_planner onnxruntime)
target_link_libraries(rl_dp PRIVATE /usr/local/onnxruntime/lib/libonnxruntime.so)

add_executable(infer_demo cpp_onnx_infer_example.cpp)
target_link_libraries(infer_demo PRIVATE onnxruntime dp_policy dp_planner dp_infer)
target_link_libraries(infer_demo PRIVATE /usr/local/onnxruntime/lib/libonnxruntime.so)

add_executable(rollout_demo cpp_rollout_visualize.cpp)
target_link_libraries(rollout_demo PRIVATE onnxruntime dp_policy dp_planner)
target_link_libraries(rollout_demo PRIVATE /usr/local/onnxruntime/lib/libonnxruntime.so)

add_executable(rl_dp_test rl_dp_test.cpp)
target_link_libraries(rl_dp_test PRIVATE onnxruntime rl_dp)
target_link_libraries(rl_dp_test PRIVATE /usr/local/onnxruntime/lib/libonnxruntime.so)

if(USE_MATPLOTLIBCPP)
    set(MATPLOTLIBCPP_DIR "${PROJECT_SOURCE_DIR}/../third_party/matplotlib-cpp-master")
    if(NOT EXISTS "${MATPLOTLIBCPP_DIR}/matplotlibcpp.h")
        message(FATAL_ERROR "matplotlibcpp.h not found at ${MATPLOTLIBCPP_DIR}")
    endif()

    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    find_package(Python3 COMPONENTS NumPy)

    add_library(matplotlib_cpp INTERFACE)
    target_include_directories(matplotlib_cpp INTERFACE ${MATPLOTLIBCPP_DIR})
    target_link_libraries(matplotlib_cpp INTERFACE Python3::Python Python3::Module)
    if(Python3_NumPy_FOUND)
        target_link_libraries(matplotlib_cpp INTERFACE Python3::NumPy)
    else()
        target_compile_definitions(matplotlib_cpp INTERFACE WITHOUT_NUMPY)
    endif()

    target_link_libraries(infer_demo PRIVATE matplotlib_cpp)
    target_include_directories(infer_demo PRIVATE ${MATPLOTLIBCPP_DIR})
    target_compile_definitions(infer_demo PRIVATE USE_MATPLOTLIBCPP)
    message(STATUS "matplotlib-cpp enabled from ${MATPLOTLIBCPP_DIR}")

    target_link_libraries(rollout_demo PRIVATE matplotlib_cpp)
    target_include_directories(rollout_demo PRIVATE ${MATPLOTLIBCPP_DIR})
    target_compile_definitions(rollout_demo PRIVATE USE_MATPLOTLIBCPP)

    target_link_libraries(rl_dp_test PRIVATE matplotlib_cpp)
    target_include_directories(rl_dp_test PRIVATE ${MATPLOTLIBCPP_DIR})
    target_compile_definitions(rl_dp_test PRIVATE USE_MATPLOTLIBCPP)
endif()

# 运行期如需 rpath，可启用以下两行：
# set_target_properties(infer_demo PROPERTIES INSTALL_RPATH "${ONNXRUNTIME_LIB_DIR}")
# set_target_properties(infer_demo PROPERTIES BUILD_RPATH "${ONNXRUNTIME_LIB_DIR}")

message(STATUS "Using ONNXRUNTIME_INCLUDE_DIR=${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "Using ONNXRUNTIME_LIB_DIR=${ONNXRUNTIME_LIB_DIR}")
