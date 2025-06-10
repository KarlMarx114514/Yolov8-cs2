cmake_minimum_required(VERSION 3.16)
project(YOLOv8cs2)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Add compile options for better performance
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(MSVC)
        add_compile_options(/O2 /Ob2 /DNDEBUG)
    else()
        add_compile_options(-O3 -DNDEBUG -march=native)
    endif()
endif()

# Find OpenCV
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
    message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
else()
    message(FATAL_ERROR "OpenCV not found! Please install OpenCV")
endif()

# ONNX Runtime paths
set(ONNXRUNTIME_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime)
set(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_ROOT_PATH}/include)
set(ONNXRUNTIME_LIB_DIRS ${ONNXRUNTIME_ROOT_PATH}/lib)

# Check if ONNX Runtime directory exists
if(NOT EXISTS ${ONNXRUNTIME_ROOT_PATH})
    message(FATAL_ERROR "ONNX Runtime directory not found at ${ONNXRUNTIME_ROOT_PATH}")
endif()

if(NOT EXISTS ${ONNXRUNTIME_INCLUDE_DIRS})
    message(FATAL_ERROR "ONNX Runtime include directory not found at ${ONNXRUNTIME_INCLUDE_DIRS}")
endif()

if(NOT EXISTS ${ONNXRUNTIME_LIB_DIRS})
    message(FATAL_ERROR "ONNX Runtime lib directory not found at ${ONNXRUNTIME_LIB_DIRS}")
endif()

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

if(ONNXRUNTIME_PROVIDERS_SHARED_LIB)
    message(STATUS "ONNX Runtime shared providers found: ${ONNXRUNTIME_PROVIDERS_SHARED_LIB}")
endif()

# Create executable
add_executable(${PROJECT_NAME} main.cpp)

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIRS}
)

# Prepare ONNX Runtime libraries list
set(ONNX_LIBS ${ONNXRUNTIME_LIB})
if(ONNXRUNTIME_PROVIDERS_CUDA_LIB)
    list(APPEND ONNX_LIBS ${ONNXRUNTIME_PROVIDERS_CUDA_LIB})
endif()
if(ONNXRUNTIME_PROVIDERS_SHARED_LIB)
    list(APPEND ONNX_LIBS ${ONNXRUNTIME_PROVIDERS_SHARED_LIB})
endif()

# Windows-specific libraries for screen capture
if(WIN32)
    set(WINDOWS_LIBS d3d11 dxgi winmm)
    message(STATUS "Adding Windows libraries: ${WINDOWS_LIBS}")
else()
    set(WINDOWS_LIBS)
    message(STATUS "Non-Windows platform detected - screen capture features disabled")
endif()

# Link libraries
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${ONNX_LIBS}
    ${WINDOWS_LIBS}
)

# Windows-specific configurations
if(WIN32)
    # Copy ONNX Runtime DLLs to output directory
    file(GLOB ONNXRUNTIME_DLLS "${ONNXRUNTIME_ROOT_PATH}/bin/*.dll")
    
    if(ONNXRUNTIME_DLLS)
        foreach(DLL ${ONNXRUNTIME_DLLS})
            get_filename_component(DLL_NAME ${DLL} NAME)
            message(STATUS "Will copy DLL: ${DLL_NAME}")
            add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${DLL}
                $<TARGET_FILE_DIR:${PROJECT_NAME}>
                COMMENT "Copying ${DLL_NAME} to output directory"
            )
        endforeach()
        message(STATUS "ONNX Runtime DLLs will be copied to output directory")
    else()
        message(WARNING "No ONNX Runtime DLLs found in ${ONNXRUNTIME_ROOT_PATH}/bin/")
        message(WARNING "Make sure to manually copy DLLs to the executable directory")
    endif()
    
    # Copy OpenCV DLLs if available
    if(OpenCV_FOUND AND EXISTS "${OpenCV_DIR}/../bin")
        file(GLOB OPENCV_DLLS "${OpenCV_DIR}/../bin/opencv_*.dll")
        foreach(DLL ${OPENCV_DLLS})
            get_filename_component(DLL_NAME ${DLL} NAME)
            add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${DLL}
                $<TARGET_FILE_DIR:${PROJECT_NAME}>
                COMMENT "Copying ${DLL_NAME} to output directory"
            )
        endforeach()
    endif()
    
    # Set Windows-specific compile definitions
    target_compile_definitions(${PROJECT_NAME} PRIVATE 
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
        WIN32_LEAN_AND_MEAN
    )
endif()

# Compiler-specific options
if(MSVC)
    # MSVC-specific options
    target_compile_options(${PROJECT_NAME} PRIVATE 
        /W3          # Warning level 3
        /permissive- # Disable non-conforming code
        /Zc:__cplusplus # Enable correct __cplusplus macro
    )
    
    # Enable parallel compilation
    target_compile_options(${PROJECT_NAME} PRIVATE /MP)
    
    # Runtime library
    set_property(TARGET ${PROJECT_NAME} PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
        
else()
    # GCC/Clang options
    target_compile_options(${PROJECT_NAME} PRIVATE 
        -Wall 
        -Wextra 
        -Wpedantic
        -Wno-unused-parameter
    )
    
    # Enable threading support
    find_package(Threads REQUIRED)
    target_link_libraries(${PROJECT_NAME} Threads::Threads)
endif()

# Create a configuration summary
message(STATUS "")
message(STATUS "=== Configuration Summary ===")
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "ONNX Runtime path: ${ONNXRUNTIME_ROOT_PATH}")
message(STATUS "Platform: ${CMAKE_SYSTEM_NAME}")
if(WIN32)
    message(STATUS "Screen capture: Enabled (DirectX + GDI)")
else()
    message(STATUS "Screen capture: Disabled (Windows only)")
endif()
message(STATUS "GPU acceleration: ${ONNXRUNTIME_PROVIDERS_CUDA_LIB}")
message(STATUS "=============================")
message(STATUS "")

# Installation rules (optional)
install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin
    COMPONENT applications
)

if(WIN32 AND ONNXRUNTIME_DLLS)
    install(FILES ${ONNXRUNTIME_DLLS}
        DESTINATION bin
        COMPONENT runtime
    )
endif()

# Add custom target for cleaning generated files
add_custom_target(clean-all
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/CMakeFiles
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/CMakeCache.txt
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/cmake_install.cmake
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/Makefile
    COMMENT "Cleaning all generated files"
)

# Add custom target for model download (if needed)
add_custom_target(download-model
    COMMAND ${CMAKE_COMMAND} -E echo "Place your yolov8s_cs2.onnx model in the build directory"
    COMMENT "Model download reminder"
)

# Print build instructions
if(CMAKE_GENERATOR MATCHES "Visual Studio")
    message(STATUS "Build instructions for Visual Studio:")
    message(STATUS "  1. Open ${PROJECT_NAME}.sln")
    message(STATUS "  2. Set Release configuration")
    message(STATUS "  3. Build solution")
else()
    message(STATUS "Build instructions:")
    message(STATUS "  mkdir build && cd build")
    message(STATUS "  cmake .. -DCMAKE_BUILD_TYPE=Release")
    message(STATUS "  cmake --build . --config Release")
endif()

message(STATUS "Usage examples:")
message(STATUS "  ${PROJECT_NAME} --realtime")
message(STATUS "  ${PROJECT_NAME} --benchmark 20")
message(STATUS "  ${PROJECT_NAME} --conf 0.3 screenshot.jpg")