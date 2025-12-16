cmake_minimum_required(VERSION 3.16)
project(simple_cuda LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

find_package(CUDAToolkit REQUIRED)
find_package(Armadillo REQUIRED)

file(GLOB_RECURSE MAIN_CXX_FILES "content/*/main.cu")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
foreach(MAIN_CXX_FILE ${MAIN_CXX_FILES})
    get_filename_component(MAIN_DIR ${MAIN_CXX_FILE} DIRECTORY)
    get_filename_component(MODULE_NAME ${MAIN_DIR} NAME)
    
    set(EXEC_NAME "${MODULE_NAME}.x")
    add_executable(${EXEC_NAME} ${MAIN_CXX_FILE})
    
    target_link_libraries(${EXEC_NAME}
        PRIVATE
            CUDA::cudart
            Armadillo::armadillo
    )
    target_include_directories(${EXEC_NAME}
        PUBLIC
            ${CUDAToolkit_INCLUDE_DIRS}
    )
endforeach()
