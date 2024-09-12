file(GLOB SIM_SOURCES "${CMAKE_SOURCE_DIR}/sim_env/src/*.cc")

add_library(sim_env ${SIM_SOURCES})
target_include_directories(sim_env PUBLIC ${CMAKE_SOURCE_DIR}/sim_env/include)
