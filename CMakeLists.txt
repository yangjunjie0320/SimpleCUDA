cmake_minimum_required(VERSION 3.20)
project(simple_cuda LANGUAGES CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -w")

find_package(CUDAToolkit REQUIRED)
find_package(xtensor REQUIRED)

file(GLOB_RECURSE MAIN_CXX_FILES "content/*/main.cu")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
foreach(MAIN_CXX_FILE ${MAIN_CXX_FILES})
    get_filename_component(MAIN_DIR ${MAIN_CXX_FILE} DIRECTORY)
    get_filename_component(MODULE_NAME ${MAIN_DIR} NAME)

    set(EXEC_NAME "${MODULE_NAME}.x")
    add_executable(${EXEC_NAME} ${MAIN_CXX_FILE})

    target_link_libraries(
        ${EXEC_NAME}
        PRIVATE CUDA::cudart
    )

    target_include_directories(
        ${EXEC_NAME}
        PRIVATE ${xtensor_INCLUDE_DIRS}
    )
endforeach()
