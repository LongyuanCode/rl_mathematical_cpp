# set(PYBIND11_PATH "$ENV{HOME}/anaconda3/envs/rl/lib/python3.8/site-packages/pybind11/share/cmake/pybind11")
# find_package(pybind11 REQUIRED PATHS ${PYBIND11_PATH})

# file(GLOB PYBIND_SRC "${CMAKE_SOURCE_DIR}/src/*.cc")

# add_library(value_iteration MODULE ${PYBIND_SRC})
# target_include_directories(value_iteration PUBLIC ${CMAKE_SOURCE_DIR}/include)
# target_include_directories(value_iteration PUBLIC ${pybind11_INCLUDE_DIRS})

# target_link_libraries(value_iteration PRIVATE pybind11::module)

# # 设置生成的共享库的名称
# set_target_properties(value_iteration PROPERTIES PREFIX "" SUFFIX ".so")
