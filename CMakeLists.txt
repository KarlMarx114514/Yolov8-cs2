cmake_minimum_required(VERSION 3.16)
project(YOLOv8cs2)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Windows-only check
if(NOT WIN32)
    message(FATAL_ERROR "This project is designed for Windows only")
endif()

# Add compile options for better performance
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(MSVC)
        add_compile_options(/O2 /Ob2 /DNDEBUG /fp:fast)
    else()
        add_compile_options(-O3 -DNDEBUG -march=native -ffast-math)
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

# Find ONNX Runtime libraries
find_library(ONNXRUNTIME_LIB onnxruntime HINTS ${ONNXRUNTIME_LIB_DIRS})
find_library(ONNXRUNTIME_PROVIDERS_CUDA_LIB onnxruntime_providers_cuda HINTS ${ONNXRUNTIME_LIB_DIRS})
find_library(ONNXRUNTIME_PROVIDERS_SHARED_LIB onnxruntime_providers_shared HINTS ${ONNXRUNTIME_LIB_DIRS})

if(NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR "ONNX Runtime library not found in ${ONNXRUNTIME_LIB_DIRS}")
endif()

message(STATUS "ONNX Runtime found: ${ONNXRUNTIME_LIB}")

# DEBUG: Check if modular structure exists and find all files
message(STATUS "")
message(STATUS "=== DEBUGGING SOURCE FILE DETECTION ===")
message(STATUS "Current source directory: ${CMAKE_CURRENT_SOURCE_DIR}")

# Check if directories exist
set(REQUIRED_DIRS 
    "${CMAKE_CURRENT_SOURCE_DIR}/src/core"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/capture"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/control"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/display"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/export"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/realtime"
)

foreach(DIR ${REQUIRED_DIRS})
    if(EXISTS ${DIR})
        message(STATUS "✓ Directory exists: ${DIR}")
    else()
        message(STATUS "✗ Directory missing: ${DIR}")
    endif()
endforeach()

# Check for modular vs single file structure
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/core" AND 
   EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/capture" AND
   EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/control" AND
   EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/display" AND
   EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/export" AND
   EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/realtime")
    
    message(STATUS "Using MODULAR structure")
    
    # Collect source files by module with detailed logging
    file(GLOB CORE_SOURCES "src/core/*.cpp")
    file(GLOB CAPTURE_SOURCES "src/capture/*.cpp")
    file(GLOB CONTROL_SOURCES "src/control/*.cpp")
    file(GLOB DISPLAY_SOURCES "src/display/*.cpp")
    file(GLOB EXPORT_SOURCES "src/export/*.cpp")
    file(GLOB REALTIME_SOURCES "src/realtime/*.cpp")
    file(GLOB UTILS_SOURCES "src/utils/*.cpp")

    # Debug: Print found sources
    message(STATUS "")
    message(STATUS "=== FOUND SOURCE FILES ===")
    message(STATUS "Core sources: ${CORE_SOURCES}")
    message(STATUS "Capture sources: ${CAPTURE_SOURCES}")
    message(STATUS "Control sources: ${CONTROL_SOURCES}")
    message(STATUS "Display sources: ${DISPLAY_SOURCES}")
    message(STATUS "Export sources: ${EXPORT_SOURCES}")
    message(STATUS "Realtime sources: ${REALTIME_SOURCES}")
    message(STATUS "Utils sources: ${UTILS_SOURCES}")

    set(ALL_LIB_SOURCES 
        ${CORE_SOURCES}
        ${CAPTURE_SOURCES}
        ${CONTROL_SOURCES}
        ${DISPLAY_SOURCES}
        ${EXPORT_SOURCES}
        ${REALTIME_SOURCES}
        ${UTILS_SOURCES}
    )

    message(STATUS "")
    message(STATUS "Total library sources found: ${ALL_LIB_SOURCES}")
    message(STATUS "Total source count: ")
    list(LENGTH ALL_LIB_SOURCES SOURCE_COUNT)
    message(STATUS "  ${SOURCE_COUNT} files")

    set(MODULAR_BUILD TRUE)
else()
    message(STATUS "Using SINGLE-FILE structure with main.cpp")
    set(MODULAR_BUILD FALSE)
endif()

# Windows-specific libraries for screen capture
set(WINDOWS_LIBS 
    d3d11
    dxgi
    winmm
    user32
    gdi32
)

# Prepare ONNX Runtime libraries list
set(ONNX_LIBS ${ONNXRUNTIME_LIB})
if(ONNXRUNTIME_PROVIDERS_CUDA_LIB)
    list(APPEND ONNX_LIBS ${ONNXRUNTIME_PROVIDERS_CUDA_LIB})
endif()
if(ONNXRUNTIME_PROVIDERS_SHARED_LIB)
    list(APPEND ONNX_LIBS ${ONNXRUNTIME_PROVIDERS_SHARED_LIB})
endif()

# Create executable - either modular or single file
if(MODULAR_BUILD)
    message(STATUS "")
    message(STATUS "=== BUILDING MODULAR VERSION ===")
    
    # Ensure we have source files
    if(NOT ALL_LIB_SOURCES)
        message(FATAL_ERROR "No source files found for modular build!")
    endif()
    
    # Create static library for the CS2 detection system
    add_library(${PROJECT_NAME}_lib STATIC ${ALL_LIB_SOURCES})

    # Set include directories for the library
    target_include_directories(${PROJECT_NAME}_lib PUBLIC
        src/
        ${OpenCV_INCLUDE_DIRS}
        ${ONNXRUNTIME_INCLUDE_DIRS}
    )

    # Set compile definitions for the library
    target_compile_definitions(${PROJECT_NAME}_lib PUBLIC
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
        WIN32_LEAN_AND_MEAN
    )

    # Link libraries to the static library
    target_link_libraries(${PROJECT_NAME}_lib PUBLIC
        ${OpenCV_LIBS}
        ${ONNX_LIBS}
        ${WINDOWS_LIBS}
    )

    # Create main executable
    add_executable(${PROJECT_NAME} src/main.cpp)

    # Link the static library to the executable
    target_link_libraries(${PROJECT_NAME} PRIVATE ${PROJECT_NAME}_lib)
    
    message(STATUS "Modular build configured successfully")
else()
    message(STATUS "")
    message(STATUS "=== BUILDING SINGLE-FILE VERSION ===")
    
    # Single file build - create executable directly
    add_executable(${PROJECT_NAME} main.cpp)

    # Set include directories for the executable
    target_include_directories(${PROJECT_NAME} PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${ONNXRUNTIME_INCLUDE_DIRS}
    )

    # Set compile definitions for the executable
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
        WIN32_LEAN_AND_MEAN
    )

    # Link libraries directly to the executable
    target_link_libraries(${PROJECT_NAME} PRIVATE
        ${OpenCV_LIBS}
        ${ONNX_LIBS}
        ${WINDOWS_LIBS}
    )
endif()

# MSVC-specific options
if(MSVC)
    if(MODULAR_BUILD)
        target_compile_options(${PROJECT_NAME}_lib PUBLIC
            /W3 /permissive- /Zc:__cplusplus /MP
        )
        set_property(TARGET ${PROJECT_NAME}_lib PROPERTY
            MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    else()
        target_compile_options(${PROJECT_NAME} PRIVATE
            /W3 /permissive- /Zc:__cplusplus /MP
        )
    endif()
    
    set_property(TARGET ${PROJECT_NAME} PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
else()
    # GCC/Clang options (MinGW)
    if(MODULAR_BUILD)
        target_compile_options(${PROJECT_NAME}_lib PUBLIC -Wall -Wextra -Wpedantic)
        find_package(Threads REQUIRED)
        target_link_libraries(${PROJECT_NAME}_lib PUBLIC Threads::Threads)
    else()
        target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic)
        find_package(Threads REQUIRED)
        target_link_libraries(${PROJECT_NAME} PRIVATE Threads::Threads)
    endif()
endif()

# Copy ONNX Runtime DLLs to output directory
file(GLOB ONNXRUNTIME_DLLS "${ONNXRUNTIME_ROOT_PATH}/lib/*.dll")

if(ONNXRUNTIME_DLLS)
    foreach(DLL ${ONNXRUNTIME_DLLS})
        get_filename_component(DLL_NAME ${DLL} NAME)
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${DLL}
            $<TARGET_FILE_DIR:${PROJECT_NAME}>
            COMMENT "Copying ${DLL_NAME} to output directory"
        )
    endforeach()
    message(STATUS "ONNX Runtime DLLs will be copied to output directory")
else()
    message(WARNING "No ONNX Runtime DLLs found in ${ONNXRUNTIME_ROOT_PATH}/lib/")
endif()

# Optional: Create benchmark executable
if(MODULAR_BUILD)
    add_executable(${PROJECT_NAME}_benchmark src/main.cpp)
    target_link_libraries(${PROJECT_NAME}_benchmark PRIVATE ${PROJECT_NAME}_lib)
    target_compile_definitions(${PROJECT_NAME}_benchmark PRIVATE BENCHMARK_MODE)
endif()

# Custom target to copy model file
add_custom_target(copy-model ALL
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/yolov8s_cs2.onnx
    ${CMAKE_CURRENT_BINARY_DIR}/yolov8s_cs2.onnx
    COMMENT "Copying model file to build directory"
    VERBATIM
)

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/yolov8s_cs2.onnx
    $<TARGET_FILE_DIR:${PROJECT_NAME}>/yolov8s_cs2.onnx
    COMMENT "Copying model file to output directory"
    VERBATIM
)

# Print configuration summary
message(STATUS "")
message(STATUS "=== FINAL CONFIGURATION SUMMARY ===")
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "ONNX Runtime path: ${ONNXRUNTIME_ROOT_PATH}")
message(STATUS "Build mode: ${MODULAR_BUILD}")
if(MODULAR_BUILD)
    message(STATUS "Library sources: ${SOURCE_COUNT} files")
endif()
message(STATUS "=============================")
message(STATUS "")