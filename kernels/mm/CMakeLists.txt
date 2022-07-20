cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(mm LANGUAGES CUDA CXX)

# include(CTest)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wunused-variable")
# set(CMAKE_CXX_FLAGS "-O2 -Wunused-variable")
# CMake will apply this to all targets, so this will work too.
# add_compile_options("-Werror")

add_executable(mm
  mm.cc
  mm.cu
  mm.h
)

set(CUDA_TOOLKIT_ROOT_DIR "${CMAKE_CUDA_COMPILER}")
get_filename_component(CUDA_TOOLKIT_ROOT_DIR "${CUDA_TOOLKIT_ROOT_DIR}" DIRECTORY)
get_filename_component(CUDA_TOOLKIT_ROOT_DIR "${CUDA_TOOLKIT_ROOT_DIR}" DIRECTORY)

find_library(CUDART_LIB cudart
             HINTS
             "${CUDA_TOOLKIT_ROOT_DIR}/lib64"
             "${CUDA_TOOLKIT_ROOT_DIR}/lib"
             "${CUDA_TOOLKIT_ROOT_DIR}"
             )

add_library(CUDA::cudart IMPORTED INTERFACE)
set_target_properties(CUDA::cudart PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES
    "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
  INTERFACE_LINK_LIBRARIES
    "${CUDART_LIB}")

target_link_libraries(mm PRIVATE CUDA::cudart)

# target_include_directories(
#   mm PUBLIC ${CUDA_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIRS})
# target_link_libraries(
#   mm ${CUDA_LIBRARIES})

# Request that particles be built with -std=c++11
# As this is a public compile feature anything that links to particles
# will also build with -std=c++11
# target_compile_features(particles PUBLIC cxx_std_11)

# We need to explicitly state that we need all CUDA files in the particle
# library to be built with -dc as the member functions could be called by
# other libraries and executables
# set_target_properties( particles
#                        PROPERTIES CUDA_SEPARABLE_COMPILATION ON
#                       )

# CUDA_ADD_EXECUTABLE(demo test.cu)

# cuda_add_executable(
#   hellocuda
#   test.cu)
