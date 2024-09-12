cmake_minimum_required(VERSION 3.20)
project(rl_cpp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# ccache
set(CMAKE_CXX_COMPILER_LAUNCHER ccache)

set(CMAKE_BUILD_TYPE Debug)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-DDEBUGSTR)
endif()

# set(PYBIND11_PATH "$ENV{HOME}/anaconda3/envs/rl/lib/python3.8/site-packages/pybind11/share/cmake/pybind11")
# find_package(pybind11 REQUIRED PATHS ${PYBIND11_PATH})

file(GLOB PYBIND_SRC "${CMAKE_SOURCE_DIR}/src/*.cc")

add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(sim_env)
add_subdirectory(utils)
add_subdirectory(test)

# link_directories(${CMAKE_SOURCE_DIR}/third_party/rlenvs_from_cpp/install/dbg)

file(GLOB MAIN_SOURCE "${CMAKE_SOURCE_DIR}/main.cc")
file(GLOB ALGO_SRC 
  "${CMAKE_SOURCE_DIR}/src/qlearning_offpolicy.cc")
# list(REMOVE_ITEM ALGO_SRC "${CMAKE_SOURCE_DIR}/src/mcts.cc")
file (GLOB ENV_SRC "${CMAKE_SOURCE_DIR}/sim_env/src/*.cc")
file (GLOB UTILS_SRC "${CMAKE_SOURCE_DIR}/utils/*.cc")


add_executable(main ${MAIN_SOURCE} ${ALGO_SRC} ${ENV_SRC} ${UTILS_SRC})
target_include_directories(main PRIVATE ${CMAKE_SOURCE_DIR}/include
                                        ${CMAKE_SOURCE_DIR}/sim_env/include
                                        ${CMAKE_SOURCE_DIR}/utils)
                                        # ${CMAKE_SOURCE_DIR}/third_party/rlenvs_from_cpp/src
                                        # ${CMAKE_SOURCE_DIR}/third_party/rlenvs_from_cpp/src/rlenvs/envs/gymnasium/toy_text)
# target_link_libraries(main rlenvscpplib)

# set_target_properties(main PROPERTIES
#     INSTALL_RPATH_USE_LINK_PATH TRUE
#     INSTALL_RPATH "${CMAKE_SOURCE_DIR}/third_party/rlenvs_from_cpp/install/dbg"
# )
