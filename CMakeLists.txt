cmake_minimum_required(VERSION 3.0)

project (ReverseAD)
set(target ReverseAD)

set(FEMSOURCES
src/MainTest.cpp
)

set(FEMINCLUDES
include/BaseExpr.hpp
#include/Derivatives.hpp
include/ExprDerived.hpp
include/ExprOps.hpp
include/Trigfuncs.hpp
include/ReverseADStuff.hpp
)

include_directories(include) 

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -pg -std=c++17 -march=native")

add_executable (ReverseAD ${FEMSOURCES} ${FEMINCLUDES})

# linking
#target_link_libraries(ReverseAD) 

# install
install(TARGETS ReverseAD DESTINATION ${CMAKE_INSTALL_PREFIX})
