#include "realtime/realtime_detector.hpp"
#include "display/overlay.hpp"
#include "export/json_export.hpp"
#include <iostream>
#include <thread>
#include <chrono>
#include <iomanip>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>

#ifdef _WIN32

namespace cs2_realtime {

CS2RealTimeDetector::CS2RealTimeDetector(const std::string& model_path, bool use_gpu) 
    : running(false), save_next_frame(false), total_frames(0), detected_frames(0), 
      window_recheck_counter(0), paused(false) {
    
    // Initialize detector
    detector = std::make_unique<cs2_detection::YOLOv8Detector>(model_path, use_gpu);
    
    // Initialize screen capture
    screen_capture = cs2_detection::ScreenCaptureFactory::create(cs2_detection::CaptureMethod::AUTO);
    
    if (!screen_capture->initializeForCS2Window()) {
        std::cerr << "Failed to initialize screen capture for CS2" << std::endl;
        throw std::runtime_error("Screen capture initialization failed");
    }
    
    // Warm up the detector
    detector->warmup(3);
    
    // Setup mouse control
    mouse_integration.setupMouseControl();
    
    // Initialize display window
    display_window = std::make_unique<cs2_display::DisplayWindow>("CS2 Detection");
}

void CS2RealTimeDetector::runRealTimeDetection(float conf_threshold, float nms_threshold, 
                                              bool show_display, bool save_results) {
    running = true;
    session_start = std::chrono::high_resolution_clock::now();
                            
    std::cout << "\n=== STARTING MULTI-THREADED CS2 DETECTION ===" << std::endl;
    std::cout << "Confidence threshold: " << conf_threshold << std::endl;
    std::cout << "NMS threshold: " << nms_threshold << std::endl;
    std::cout << "Display overlay: " << (show_display ? "Enabled" : "Disabled") << std::endl;
    std::cout << "Multi-threading: ENABLED with main-thread GUI (OpenCV safe)" << std::endl;
                            
    if (show_display) {
        std::cout << "\nIMPORTANT: Position the detection window so it doesn't cover CS2!" << std::endl;
        printControls();
    }

    // Initialize display window if needed (MUST be on main thread)
    if (show_display) {
        setupDisplayWindow();
    }

    // Start worker threads (excluding display thread)
    std::thread capture_thread(&CS2RealTimeDetector::captureThread, this, conf_threshold, nms_threshold);
    std::thread detection_thread(&CS2RealTimeDetector::detectionThread, this, conf_threshold, nms_threshold);
    std::thread mouse_thread(&CS2RealTimeDetector::mouseControlThread, this);

    // Main thread handles display and input (OpenCV GUI operations)
    auto last_stats_time = std::chrono::high_resolution_clock::now();
    
    while (running) {
        // Handle display in main thread (OpenCV threading requirement)
        if (show_display) {
            handleDisplayAndInput(save_results);
        } else {
            // Non-display mode - just handle basic timing
            std::this_thread::sleep_for(std::chrono::milliseconds(16));
        }

        // Enhanced periodic window validation
        if (++window_recheck_counter % 1000 == 0) {
            if (!mouse_integration.recheckCS2Window()) {
                std::cout << "CS2 window validation failed, continuing detection..." << std::endl;
            }
        }

        // Print periodic stats
        auto now = std::chrono::high_resolution_clock::now();
        auto stats_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_stats_time);
        if (stats_elapsed.count() >= 10000) { // Every 10 seconds
            printLiveStats();
            last_stats_time = now;
        }
    }

    // Wait for threads to finish
    if (capture_thread.joinable()) capture_thread.join();
    if (detection_thread.joinable()) detection_thread.join();
    if (mouse_thread.joinable()) mouse_thread.join();

    printSessionStats();
}

void CS2RealTimeDetector::captureThread(float conf_threshold, float nms_threshold) {
    std::cout << "[CAPTURE THREAD] Started" << std::endl;
    
    while (running) {
        if (paused) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        // Capture screen
        auto capture_result = screen_capture->capture();
        cv::Mat frame = capture_result.first;
        auto capture_metrics = capture_result.second;

        if (frame.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        total_frames++;

        // Add to capture queue for detection
        {
            std::lock_guard<std::mutex> lock(capture_mutex);
            
            // Keep queue size reasonable to prevent memory buildup
            while (capture_queue.size() >= 3) {
                capture_queue.pop();
            }
            
            CaptureData data;
            data.frame = frame.clone();  // Important: clone for thread safety
            data.metrics = capture_metrics;
            data.timestamp = std::chrono::high_resolution_clock::now();
            
            capture_queue.push(data);
        }
        capture_cv.notify_one();

        // Target ~120 FPS for capture
        std::this_thread::sleep_for(std::chrono::microseconds(8333));
    }
    
    std::cout << "[CAPTURE THREAD] Stopped" << std::endl;
}

void CS2RealTimeDetector::detectionThread(float conf_threshold, float nms_threshold) {
    std::cout << "[DETECTION THREAD] Started" << std::endl;
    
    while (running) {
        CaptureData capture_data;
        
        // Wait for capture data
        {
            std::unique_lock<std::mutex> lock(capture_mutex);
            capture_cv.wait(lock, [this] { return !capture_queue.empty() || !running; });
            
            if (!running) break;
            
            capture_data = capture_queue.front();
            capture_queue.pop();
        }

        // Run detection
        auto detection_result = detector->detectWithMetrics(capture_data.frame, conf_threshold, nms_threshold);
        auto detections = detection_result.first;
        auto detection_metrics = detection_result.second;

        if (!detections.empty()) {
            detected_frames++;
        }

        // Add to mouse control queue (highest priority - smallest queue)
        {
            std::lock_guard<std::mutex> lock(mouse_mutex);
            
            // Keep only the latest detection for minimal latency
            while (!mouse_queue.empty()) {
                mouse_queue.pop();
            }
            
            mouse_queue.push(detections);
        }
        mouse_cv.notify_one();

        // Prepare display frame (but don't show it here)
        {
            std::lock_guard<std::mutex> lock(display_mutex);
            
            // Keep queue size reasonable
            while (display_queue.size() >= 2) {
                display_queue.pop();
            }
            
            cv::Mat display_frame = capture_data.frame.clone();
            if (!detections.empty()) {
                detector->drawResults(display_frame, detections);
            }
            
            // Count detections by class for overlay
            std::vector<int> class_counts(4, 0);
            for (const auto& det : detections) {
                if (det.class_id >= 0 && det.class_id < static_cast<int>(class_counts.size())) {
                    class_counts[det.class_id]++;
                }
            }

            // Add performance overlay (HUD)
            cs2_display::PerformanceOverlay::addPerformanceOverlay(
                display_frame, 
                capture_data.metrics, 
                detection_metrics,
                static_cast<int>(detections.size()),
                class_counts
            );
            
            DetectionFrame frame_data;
            frame_data.detections = detections;
            frame_data.display_frame = display_frame;
            frame_data.capture_metrics = capture_data.metrics;
            frame_data.detection_metrics = detection_metrics;
            frame_data.timestamp = capture_data.timestamp;
            
            display_queue.push(frame_data);
        }
        display_cv.notify_one();
    }
    
    std::cout << "[DETECTION THREAD] Stopped" << std::endl;
}

void CS2RealTimeDetector::mouseControlThread() {
    std::cout << "[MOUSE THREAD] Started with high priority" << std::endl;
    
    // Set high priority for mouse control thread
#ifdef _WIN32
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
#endif
    
    while (running) {
        std::vector<cs2_detection::Detection> detections;
        
        // Wait for detection data
        {
            std::unique_lock<std::mutex> lock(mouse_mutex);
            mouse_cv.wait(lock, [this] { return !mouse_queue.empty() || !running; });
            
            if (!running) break;
            
            detections = mouse_queue.front();
            mouse_queue.pop();
        }

        // Process detections with mouse control (critical path - minimal latency)
        mouse_integration.processDetections(detections);
    }
    
    std::cout << "[MOUSE THREAD] Stopped" << std::endl;
}

void CS2RealTimeDetector::handleDisplayAndInput(bool save_results) {
    // Handle display updates (main thread only - OpenCV requirement)
    DetectionFrame frame_data;
    bool has_frame = false;
    
    // Get latest frame if available (non-blocking)
    {
        std::lock_guard<std::mutex> lock(display_mutex);
        if (!display_queue.empty()) {
            frame_data = display_queue.front();
            display_queue.pop();
            has_frame = true;
        }
    }
    
    // Display frame if we have one
    if (has_frame && display_window && display_window->isEnabled()) {
        // Save screenshot if requested
        if (save_next_frame.exchange(false)) {
            auto now = std::chrono::system_clock::now();
            auto timestamp = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
            std::string screenshot_filename = "cs2_screenshot_" + std::to_string(timestamp) + ".jpg";

            cv::imwrite(screenshot_filename, frame_data.display_frame);
            std::cout << "\n[SAVED] Screenshot saved as: " << screenshot_filename << std::endl;

            if (!frame_data.detections.empty()) {
                std::string json_filename = "cs2_detections_" + std::to_string(timestamp) + ".json";
                cs2_export::exportToJSON(frame_data.detections, json_filename);
            }
        }

        display_window->showFrame(frame_data.display_frame);
    }
    
    // Handle input (main thread only - OpenCV requirement)
    if (display_window && display_window->isEnabled()) {
        bool paused_local = paused.load();
        char key = display_window->handleInput(paused_local, save_results, true);
        paused.store(paused_local);
        
        // Handle keys that weren't processed by DisplayWindow
        if (key == 'q' || key == 'Q' || key == 27) { // Quit
            running = false;
        } else if (key == 's' || key == 'S') { // Save
            if (save_results) {
                save_next_frame = true;
            }
        } else if (key == 'm' || key == 'M') { // Mouse control toggle
            mouse_integration.toggleMouseControl();
        } else if (key == 'v' || key == 'V') { // Debug mode
            mouse_integration.toggleDebugMode();
        } else if (key == 'c' || key == 'C') { // Refresh window detection
            mouse_integration.refreshWindowDetection();
        } else if (key == 'x' || key == 'X') { // Manual window selection
            mouse_integration.manualWindowSelection();
        } else if (key == 't' || key == 'T') { // Test input methods
            mouse_integration.testInputMethods();
        } else if (key == '1') { // Force raw input mode
            mouse_integration.setInputMode(1);
        } else if (key == '2') { // Force absolute mode
            mouse_integration.setInputMode(2);
        } else if (key == '3') { // Auto-detect mode
            mouse_integration.setInputMode(3);
        } else if (key == '4') { // Fast preset
            mouse_integration.setInputMode(4);
        } else if (key == '5') { // Medium preset
            mouse_integration.setInputMode(5);
        } else if (key == '6') { // Slow preset
            mouse_integration.setInputMode(6);
        } else if (key == '+' || key == '=') { // Increase sensitivity
            mouse_integration.adjustSensitivity(true);
        } else if (key == '-' || key == '_') { // Decrease sensitivity
            mouse_integration.adjustSensitivity(false);
        }
    } else {
        // Small delay when no display
        std::this_thread::sleep_for(std::chrono::milliseconds(16));
    }
}

void CS2RealTimeDetector::setupDisplayWindow() {
    if (!display_window) {
        return;
    }
    
    bool success = display_window->setupWindow();
    if (success) {
        // Pass the detection window handle to mouse controller
#ifdef _WIN32
        HWND cv_window = display_window->getWindowHandle();
        if (cv_window) {
            mouse_integration.setDetectionWindowHandle(cv_window);
        }
#endif
    }
}

void CS2RealTimeDetector::printControls() {
    std::cout << "\nControls:" << std::endl;
    std::cout << "  Q or ESC: Quit" << std::endl;
    std::cout << "  P: Pause/Resume" << std::endl;
    std::cout << "  S: Save current frame" << std::endl;
    std::cout << "  D: Toggle display on/off" << std::endl;
    std::cout << "  M: Toggle mouse control on/off" << std::endl;
    std::cout << "  V: Toggle debug mode" << std::endl;
    std::cout << "  C: Refresh window detection" << std::endl;
    std::cout << "  X: Manual window selection" << std::endl;
    std::cout << "  T: Test input methods" << std::endl;
    std::cout << "  1: Force raw input mode" << std::endl;
    std::cout << "  2: Force absolute mode" << std::endl;
    std::cout << "  3: Auto-detect mode" << std::endl;
    std::cout << "  +/-: Adjust sensitivity" << std::endl;
    std::cout << "  4/5/6: Fast/Medium/Slow presets" << std::endl;
}

void CS2RealTimeDetector::printLiveStats() {
    auto now = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - session_start);
    
    float detection_rate = total_frames > 0 ? (detected_frames * 100.0f / total_frames) : 0.0f;
    float avg_fps = elapsed.count() > 0 ? (total_frames / static_cast<float>(elapsed.count())) : 0.0f;
    
    // Get queue sizes for performance monitoring
    size_t capture_queue_size, mouse_queue_size, display_queue_size;
    {
        std::lock_guard<std::mutex> lock1(capture_mutex);
        std::lock_guard<std::mutex> lock2(mouse_mutex);
        std::lock_guard<std::mutex> lock3(display_mutex);
        capture_queue_size = capture_queue.size();
        mouse_queue_size = mouse_queue.size();
        display_queue_size = display_queue.size();
    }
    
    std::cout << "\n=== LIVE STATS (T+" << elapsed.count() << "s) ===" << std::endl;
    std::cout << "Avg FPS: " << std::fixed << std::setprecision(1) << avg_fps << std::endl;
    std::cout << "Total Frames: " << total_frames << " | Detection Rate: " << std::fixed << std::setprecision(1) << detection_rate << "%" << std::endl;
    std::cout << "Queue sizes - Capture: " << capture_queue_size 
              << " | Mouse: " << mouse_queue_size 
              << " | Display: " << display_queue_size << std::endl;
    std::cout << "Threading: [MAIN-THREAD GUI + WORKER THREADS]" << std::endl;
}

void CS2RealTimeDetector::printSessionStats() {
    auto session_end = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::seconds>(session_end - session_start);
    
    std::cout << "\n=== SESSION COMPLETE ===" << std::endl;
    std::cout << "Total time: " << total_time.count() << " seconds" << std::endl;
    std::cout << "Total frames: " << total_frames << std::endl;
    std::cout << "Frames with detections: " << detected_frames << std::endl;
    if (total_frames > 0) {
        std::cout << "Detection rate: " << std::fixed << std::setprecision(1) 
                  << (detected_frames * 100.0f / total_frames) << "%" << std::endl;
    }
    if (total_time.count() > 0) {
        std::cout << "Average FPS: " << std::fixed << std::setprecision(1) 
                  << (total_frames / static_cast<float>(total_time.count())) << std::endl;
    }
    std::cout << "Performance: OpenCV-safe threading with main-thread GUI" << std::endl;
}

} // namespace cs2_realtime

#endif // _WIN32