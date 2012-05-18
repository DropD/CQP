cmake_minimum_required (VERSION 2.8)

project(CQP_COMMON)

add_executable(test test_common.cpp)

add_library(pycqp SHARED pycqp.cpp)

##### boost-python support

link_libraries(boost_python-mt)
include_directories("/Library/Frameworks/Python.framework/Versions/2.7/include/python2.7/")
link_directories("/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/")

#####

add_definitions(-DBOOST_DISABLE_ASSERTS)
set(cmake_release_flags "O2 -funroll-loops -fopenmp)
