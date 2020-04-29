cmake_minimum_required( VERSION 3.5.1 )
project( wemd )

message( "** Using release mode." )
set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC -O3 -ansi -march=core2 -Wall -Wno-sign-compare" )
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -O3 -std=gnu++0x -ansi -march=core2 -Wall -Wno-sign-compare" )

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
include_directories(blitz++/include ${PROJECT_SOURCE_DIR})
add_library(wemd SHARED wemd_impl.cpp)
target_link_libraries(wemd -Lblitz++/lib/libblitz.la)