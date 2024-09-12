set(GTEST_PATH "${CMAKE_SOURCE_DIR}/third_party/googletest/build/lib")
link_directories(${GTEST_PATH})

file(GLOB TEST_SRC "${CMAKE_CURRENT_SOURCE_DIR}/test_epsilongreedyselect.cc")

add_executable(my_test ${TEST_SRC})
target_include_directories(my_test PRIVATE ${CMAKE_SOURCE_DIR}/include
                                           ${CMAKE_SOURCE_DIR}/sim_env/include
                                           ${CMAKE_SOURCE_DIR}/utils
                                           ${CMAKE_SOURCE_DIR}/third_party/googletest/googletest/include
                                           ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(my_test gtest gtest_main)