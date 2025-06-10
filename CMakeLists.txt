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

# Check if we have modular source files or single main.cpp
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/core" AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/capture")
    # Modular structure exists - collect source files by module
    file(GLOB CORE_SOURCES "src/core/*.cpp")
    file(GLOB CAPTURE_SOURCES "src/capture/*.cpp")
    file(GLOB DISPLAY_SOURCES "src/display/*.cpp")
    file(GLOB EXPORT_SOURCES "src/export/*.cpp")
    file(GLOB REALTIME_SOURCES "src/realtime/*.cpp")
    file(GLOB UTILS_SOURCES "src/utils/*.cpp")

    set(ALL_LIB_SOURCES 
        ${CORE_SOURCES}
        ${CAPTURE_SOURCES}
        ${DISPLAY_SOURCES}
        ${EXPORT_SOURCES}
        ${REALTIME_SOURCES}
        ${UTILS_SOURCES}
    )

    # Print found sources for debugging
    message(STATUS "Using modular structure:")
    message(STATUS "Found core sources: ${CORE_SOURCES}")
    message(STATUS "Found capture sources: ${CAPTURE_SOURCES}")
    message(STATUS "Found display sources: ${DISPLAY_SOURCES}")
    message(STATUS "Found export sources: ${EXPORT_SOURCES}")
    message(STATUS "Found realtime sources: ${REALTIME_SOURCES}")
    message(STATUS "Found utils sources: ${UTILS_SOURCES}")

    set(MODULAR_BUILD TRUE)
else()
    # Single file structure - use main.cpp directly
    message(STATUS "Using single-file structure with main.cpp")
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
else()
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
    # MSVC compile options
    if(MODULAR_BUILD)
        target_compile_options(${PROJECT_NAME}_lib PUBLIC
            /W3          # Warning level 3
            /permissive- # Disable non-conforming code
            /Zc:__cplusplus # Enable correct __cplusplus macro
            /MP          # Parallel compilation
        )
        
        # Runtime library
        set_property(TARGET ${PROJECT_NAME}_lib PROPERTY
            MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    else()
        target_compile_options(${PROJECT_NAME} PRIVATE
            /W3          # Warning level 3
            /permissive- # Disable non-conforming code
            /Zc:__cplusplus # Enable correct __cplusplus macro
            /MP          # Parallel compilation
        )
    endif()
    
    set_property(TARGET ${PROJECT_NAME} PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
        
else()
    # GCC/Clang options (MinGW)
    if(MODULAR_BUILD)
        target_compile_options(${PROJECT_NAME}_lib PUBLIC
            -Wall 
            -Wextra 
            -Wpedantic
            -Wno-unused-parameter
        )
        
        # Enable threading support
        find_package(Threads REQUIRED)
        target_link_libraries(${PROJECT_NAME}_lib PUBLIC Threads::Threads)
    else()
        target_compile_options(${PROJECT_NAME} PRIVATE
            -Wall 
            -Wextra 
            -Wpedantic
            -Wno-unused-parameter
        )
        
        # Enable threading support
        find_package(Threads REQUIRED)
        target_link_libraries(${PROJECT_NAME} PRIVATE Threads::Threads)
    endif()
endif()

# Copy ONNX Runtime DLLs to output directory
file(GLOB ONNXRUNTIME_DLLS "${ONNXRUNTIME_ROOT_PATH}/lib/*.dll")

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
    message(WARNING "No ONNX Runtime DLLs found in ${ONNXRUNTIME_ROOT_PATH}/lib/")
    message(WARNING "Make sure to manually copy DLLs to the executable directory")
endif()

# Copy OpenCV DLLs if available
if(OpenCV_FOUND)
    # Try to find OpenCV DLLs in common locations
    set(OPENCV_POSSIBLE_DLL_DIRS
        "${OpenCV_DIR}/../bin"
        "${OpenCV_DIR}/../../bin"
        "${OpenCV_DIR}/../../../x64/vc16/bin"
        "${OpenCV_DIR}/../../../x64/vc15/bin"
        "${OpenCV_DIR}/../../../x64/vc14/bin"
    )
    
    foreach(DLL_DIR ${OPENCV_POSSIBLE_DLL_DIRS})
        if(EXISTS "${DLL_DIR}")
            file(GLOB OPENCV_DLLS "${DLL_DIR}/opencv_*.dll")
            if(OPENCV_DLLS)
                message(STATUS "Found OpenCV DLLs in: ${DLL_DIR}")
                foreach(DLL ${OPENCV_DLLS})
                    get_filename_component(DLL_NAME ${DLL} NAME)
                    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        ${DLL}
                        $<TARGET_FILE_DIR:${PROJECT_NAME}>
                        COMMENT "Copying ${DLL_NAME} to output directory"
                    )
                endforeach()
                break()
            endif()
        endif()
    endforeach()
endif()

# Optional: Create test executable (if tests exist)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests")
    file(GLOB_RECURSE TEST_SOURCES "tests/*.cpp")
    if(TEST_SOURCES AND MODULAR_BUILD)
        add_executable(${PROJECT_NAME}_tests ${TEST_SOURCES})
        target_link_libraries(${PROJECT_NAME}_tests PRIVATE ${PROJECT_NAME}_lib)
        message(STATUS "Test executable will be created")
    endif()
endif()

# Optional: Create benchmark executable
if(MODULAR_BUILD)
    add_executable(${PROJECT_NAME}_benchmark src/main.cpp)
    target_link_libraries(${PROJECT_NAME}_benchmark PRIVATE ${PROJECT_NAME}_lib)
    target_compile_definitions(${PROJECT_NAME}_benchmark PRIVATE BENCHMARK_MODE)
else()
    # For single file builds, the main executable IS the benchmark
    message(STATUS "Single file build - use main executable for benchmarking")
endif()

# Installation rules
install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin
    COMPONENT applications
)

if(MODULAR_BUILD)
    install(TARGETS ${PROJECT_NAME}_lib
        ARCHIVE DESTINATION lib
        COMPONENT libraries
    )
endif()

if(ONNXRUNTIME_DLLS)
    install(FILES ${ONNXRUNTIME_DLLS}
        DESTINATION bin
        COMPONENT runtime
    )
endif()

# Create a configuration summary
message(STATUS "")
message(STATUS "=== Configuration Summary ===")
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "ONNX Runtime path: ${ONNXRUNTIME_ROOT_PATH}")
message(STATUS "Platform: Windows (DirectX + GDI screen capture)")
message(STATUS "GPU acceleration: ${ONNXRUNTIME_PROVIDERS_CUDA_LIB}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Optimizations: Enabled")
else()
    message(STATUS "Optimizations: Disabled (Debug mode)")
endif()
message(STATUS "=============================")
message(STATUS "")

# Add custom targets for convenience
add_custom_target(clean-all
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/CMakeFiles
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/CMakeCache.txt
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/cmake_install.cmake
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/Makefile
    COMMENT "Cleaning all generated files"
)

# Custom target to copy model file (independent of main target)
add_custom_target(copy-model ALL
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/yolov8s_cs2.onnx
    ${CMAKE_CURRENT_BINARY_DIR}/yolov8s_cs2.onnx
    COMMENT "Copying model file to build directory"
    VERBATIM
)

# Also copy to the output directory after build
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/yolov8s_cs2.onnx
    $<TARGET_FILE_DIR:${PROJECT_NAME}>/yolov8s_cs2.onnx
    COMMENT "Copying model file to output directory"
    VERBATIM
)

# Print build instructions
message(STATUS "Build instructions:")
if(CMAKE_GENERATOR MATCHES "Visual Studio")
    message(STATUS "  1. Open ${PROJECT_NAME}.sln in Visual Studio")
    message(STATUS "  2. Set Release configuration")
    message(STATUS "  3. Build solution")
else()
    message(STATUS "  mkdir build && cd build")
    message(STATUS "  cmake .. -DCMAKE_BUILD_TYPE=Release")
    message(STATUS "  cmake --build . --config Release")
endif()

message(STATUS "")
if(MODULAR_BUILD)
    message(STATUS "Build mode: Modular (library + executable)")
    message(STATUS "Usage examples:")
    message(STATUS "  ${PROJECT_NAME}.exe --realtime")
    message(STATUS "  ${PROJECT_NAME}.exe --benchmark 20")
    message(STATUS "  ${PROJECT_NAME}.exe --conf 0.3 screenshot.jpg")
else()
    message(STATUS "Build mode: Single file")
    message(STATUS "Usage examples:")
    message(STATUS "  ${PROJECT_NAME}.exe --realtime")
    message(STATUS "  ${PROJECT_NAME}.exe --benchmark 20")
    message(STATUS "  ${PROJECT_NAME}.exe --conf 0.3 screenshot.jpg")
endif()