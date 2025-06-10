cmake_minimum_required(VERSION 3.16)
project(YOLOv8cs2)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
    message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
endif()

# ONNX Runtime paths
set(ONNXRUNTIME_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime)
set(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_ROOT_PATH}/include)
set(ONNXRUNTIME_LIB_DIRS ${ONNXRUNTIME_ROOT_PATH}/lib)

# Find ONNX Runtime libraries
find_library(ONNXRUNTIME_LIB onnxruntime HINTS ${ONNXRUNTIME_LIB_DIRS})
find_library(ONNXRUNTIME_PROVIDERS_CUDA_LIB onnxruntime_providers_cuda HINTS ${ONNXRUNTIME_LIB_DIRS})
find_library(ONNXRUNTIME_PROVIDERS_SHARED_LIB onnxruntime_providers_shared HINTS ${ONNXRUNTIME_LIB_DIRS})

if(NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR "ONNX Runtime library not found in ${ONNXRUNTIME_LIB_DIRS}")
endif()

message(STATUS "ONNX Runtime found: ${ONNXRUNTIME_LIB}")
if(ONNXRUNTIME_PROVIDERS_CUDA_LIB)
    message(STATUS "ONNX Runtime CUDA provider found: ${ONNXRUNTIME_PROVIDERS_CUDA_LIB}")
else()
    message(WARNING "ONNX Runtime CUDA provider not found - GPU acceleration may not work")
endif()

# Create executable
add_executable(${PROJECT_NAME} main.cpp)

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIRS}
)

# Link libraries
set(ONNX_LIBS ${ONNXRUNTIME_LIB})
if(ONNXRUNTIME_PROVIDERS_CUDA_LIB)
    list(APPEND ONNX_LIBS ${ONNXRUNTIME_PROVIDERS_CUDA_LIB})
endif()
if(ONNXRUNTIME_PROVIDERS_SHARED_LIB)
    list(APPEND ONNX_LIBS ${ONNXRUNTIME_PROVIDERS_SHARED_LIB})
endif()

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${ONNX_LIBS}
)

# Copy DLLs to output directory (Windows)
if(WIN32)
    # Copy ONNX Runtime DLLs
    file(GLOB ONNXRUNTIME_DLLS "${ONNXRUNTIME_ROOT_PATH}/bin/*.dll")
    foreach(DLL ${ONNXRUNTIME_DLLS})
        get_filename_component(DLL_NAME ${DLL} NAME)
        message(STATUS "Will copy DLL: ${DLL_NAME}")
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${DLL}
            $<TARGET_FILE_DIR:${PROJECT_NAME}>
        )
    endforeach()
   
    message(STATUS "ONNX Runtime DLLs will be copied to output directory")
endif()

# Compiler-specific options
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /W3)
    target_compile_definitions(${PROJECT_NAME} PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra)
endif()