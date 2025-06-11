#include "realtime/realtime_detector.hpp"
#include "display/overlay.hpp"
#include "export/json_export.hpp"
#include <iostream>
#include <thread>
#include <chrono>
#include <iomanip>

#ifdef _WIN32

namespace cs2_realtime {

CS2RealTimeDetector::CS2RealTimeDetector(const std::string& model_path, bool use_gpu) 
    : running(false), save_next_frame(false), total_frames(0), detected_frames(0), window_recheck_counter(0) {
    
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
                            
    std::cout << "\n=== STARTING REAL-TIME CS2 DETECTION ===" << std::endl;
    std::cout << "Confidence threshold: " << conf_threshold << std::endl;
    std::cout << "NMS threshold: " << nms_threshold << std::endl;
    std::cout << "Display overlay: " << (show_display ? "Enabled" : "Disabled") << std::endl;
                            
    if (show_display) {
        std::cout << "\nIMPORTANT: Position the detection window so it doesn't cover CS2!" << std::endl;
        std::cout << "Controls:" << std::endl;
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

    bool paused = false;
    auto last_stats_time = std::chrono::high_resolution_clock::now();

    // Initialize display window if needed
    if (show_display) {
        setupDisplayWindow();
    }

    while (running) {
        if (!paused) {
            // Enhanced periodic window validation
            if (++window_recheck_counter % 100 == 0) {
                if (!mouse_integration.recheckCS2Window()) {
                    std::cout << "CS2 window validation failed, continuing detection..." << std::endl;
                }
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

            // Run detection
            auto detection_result = detector->detectWithMetrics(frame, conf_threshold, nms_threshold);
            auto detections = detection_result.first;
            auto detection_metrics = detection_result.second;

            // Process detections with mouse control
            mouse_integration.processDetections(detections);

            // Count detections by class
            std::vector<int> class_counts(4, 0);
            for (const auto& det : detections) {
                if (det.class_id >= 0 && det.class_id < static_cast<int>(class_counts.size())) {
                    class_counts[det.class_id]++;
                }
            }

            if (!detections.empty()) {
                detected_frames++;
                detector->drawResults(frame, detections);
            }

            // Save screenshot if requested
            if (save_next_frame) {
                auto now = std::chrono::system_clock::now();
                auto timestamp = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
                std::string screenshot_filename = "cs2_screenshot_" + std::to_string(timestamp) + ".jpg";

                cv::imwrite(screenshot_filename, frame);
                std::cout << "\n[SAVED] Screenshot saved as: " << screenshot_filename << std::endl;

                if (!detections.empty()) {
                    std::string json_filename = "cs2_detections_" + std::to_string(timestamp) + ".json";
                    cs2_export::exportToJSON(detections, json_filename);
                }

                save_next_frame = false;
            }

            // Add performance overlay (HUD)
            cs2_display::PerformanceOverlay::addPerformanceOverlay(frame, capture_metrics, detection_metrics, 
                                                                  detections.size(), class_counts);
            
            // Display frame (if enabled)
            if (show_display && display_window && display_window->isEnabled()) {
                showFrame(frame, save_results, paused);
            }

            // Print periodic stats
            auto now = std::chrono::high_resolution_clock::now();
            auto stats_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_stats_time);
            if (stats_elapsed.count() >= 10000) { // Every 10 seconds
                printLiveStats(capture_metrics, detection_metrics, detections.size());
                last_stats_time = now;
            }

        } else {
            // Paused - just handle input
            if (show_display && display_window && display_window->isEnabled()) {
                handleInput(paused, save_results, true);
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
    }

    printSessionStats();
    if (display_window && display_window->isEnabled()) {
        // DisplayWindow destructor will handle cleanup
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

void CS2RealTimeDetector::showFrame(const cv::Mat& frame, bool save_results, bool& paused) {
    if (!display_window) {
        return;
    }
    
    display_window->showFrame(frame);
    handleInput(paused, save_results, true);
}

void CS2RealTimeDetector::handleInput(bool& paused, bool save_results, bool update_display) {
    if (!display_window) {
        return;
    }
    
    char key = display_window->handleInput(paused, save_results, update_display);
    
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
}

void CS2RealTimeDetector::printLiveStats(const cs2_detection::CaptureMetrics& capture_metrics,
                                        const cs2_detection::PerformanceMetrics& detection_metrics,
                                        int current_detections) {
    auto now = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - session_start);
    
    float detection_rate = total_frames > 0 ? (detected_frames * 100.0f / total_frames) : 0.0f;
    float avg_fps = elapsed.count() > 0 ? (total_frames / static_cast<float>(elapsed.count())) : 0.0f;
    
    std::cout << "\n=== LIVE STATS (T+" << elapsed.count() << "s) ===" << std::endl;
    std::cout << "Current FPS: " << static_cast<int>(capture_metrics.fps) << " | Avg FPS: " << std::fixed << std::setprecision(1) << avg_fps << std::endl;
    std::cout << "Total Frames: " << total_frames << " | Detection Rate: " << std::fixed << std::setprecision(1) << detection_rate << "%" << std::endl;
    std::cout << "Current Players: " << current_detections << std::endl;
    std::cout << "Latency - Capture: " << std::fixed << std::setprecision(1) << capture_metrics.total_ms 
              << "ms | Detection: " << detection_metrics.total_ms << "ms" << std::endl;
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
}

} // namespace cs2_realtime

#endif // _WIN32