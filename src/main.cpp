#include "core/detector.hpp"
#include "capture/screen_capture.hpp"
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <thread>
#include <atomic>
#include <iomanip>
#include <chrono>

// Windows-specific includes
#ifdef _WIN32
#include <windows.h>
#endif

// Export to JSON function for detections
void exportToJSON(const std::vector<cs2_detection::Detection>& detections, const std::string& filename) {
    std::ofstream json_file(filename);
    if (!json_file.is_open()) {
        std::cerr << "Error: Cannot create JSON file " << filename << std::endl;
        return;
    }
    
    json_file << "{\n";
    json_file << "  \"detections\": [\n";
    
    for (int i = 0; i < static_cast<int>(detections.size()); i++) {
        const auto& det = detections[i];
        json_file << "    {\n";
        json_file << "      \"bbox\": {\n";
        json_file << "        \"x\": " << det.bbox.x << ",\n";
        json_file << "        \"y\": " << det.bbox.y << ",\n";
        json_file << "        \"width\": " << det.bbox.width << ",\n";
        json_file << "        \"height\": " << det.bbox.height << "\n";
        json_file << "      },\n";
        json_file << "      \"confidence\": " << det.confidence << ",\n";
        json_file << "      \"class_id\": " << det.class_id << ",\n";
        json_file << "      \"class_name\": \"" << det.class_name << "\"\n";
        json_file << "    }";
        if (i < static_cast<int>(detections.size()) - 1) json_file << ",";
        json_file << "\n";
    }
    
    json_file << "  ]\n";
    json_file << "}\n";
    
    json_file.close();
    std::cout << "CS2 detection data exported to: " << filename << std::endl;
}

// Real-time detector class with advanced features (Windows only)
#ifdef _WIN32
class CS2RealTimeDetector {
private:
    std::unique_ptr<cs2_detection::YOLOv8Detector> detector;
    std::unique_ptr<cs2_detection::WindowsScreenCapture> screen_capture;
    std::atomic<bool> running{false};
    std::atomic<bool> save_next_frame{false};
    
    // Performance tracking
    int total_frames = 0;
    int detected_frames = 0;
    std::chrono::high_resolution_clock::time_point session_start;
    
    // Display settings
    bool display_enabled = false;
    std::string window_name = "CS2 Detection";

public:
    CS2RealTimeDetector(const std::string& model_path, bool use_gpu = true) {
        detector = std::make_unique<cs2_detection::YOLOv8Detector>(model_path, use_gpu);
        screen_capture = cs2_detection::ScreenCaptureFactory::create(cs2_detection::CaptureMethod::AUTO);
        
        if (!screen_capture->initializeForCS2Window()) {
            std::cerr << "Failed to initialize screen capture for CS2" << std::endl;
            throw std::runtime_error("Screen capture initialization failed");
        }
        
        detector->warmup(3);
    }
    
    void runRealTimeDetection(float conf_threshold = 0.2f, float nms_threshold = 0.45f, 
                            bool show_display = true, bool save_results = false) {
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
        }

        bool paused = false;
        auto last_stats_time = std::chrono::high_resolution_clock::now();
        int window_check_counter = 0;

        // Initialize display window if needed
        if (show_display) {
            setupDisplayWindow();
        }

        while (running) {
            if (!paused) {
                // Periodically check if CS2 window is still valid (simplified for modular version)
                if (++window_check_counter % 100 == 0) {
                    // In the modular version, we rely on the capture method to handle window validation
                    // The screen capture implementation should handle window changes internally
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

                    // Also save detection data if there are any detections
                    if (!detections.empty()) {
                        std::string json_filename = "cs2_detections_" + std::to_string(timestamp) + ".json";
                        exportToJSON(detections, json_filename);
                    }

                    save_next_frame = false;
                }

                // Add performance overlay (HUD)
                addPerformanceOverlay(frame, capture_metrics, detection_metrics, 
                                    detections.size(), class_counts);
                
                // Display frame (if enabled)
                if (show_display && display_enabled) {
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
                if (show_display && display_enabled) {
                    handleInput(paused, save_results, true);
                } else {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
            }
        }

        printSessionStats();
        if (display_enabled) {
            cv::destroyAllWindows();
        }
    }
    
    void stop() {
        running = false;
    }
    
private:
    void setupDisplayWindow() {
        try {
            cv::namedWindow(window_name, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
            
            // Position window away from game area (bottom-right corner)
#ifdef _WIN32
            int screen_width = GetSystemMetrics(SM_CXSCREEN);
            int screen_height = GetSystemMetrics(SM_CYSCREEN);
#else
            int screen_width = 1920;  // Default fallback
            int screen_height = 1080;
#endif
            int window_width = 800;
            int window_height = 600;
            
            cv::moveWindow(window_name, screen_width - window_width - 50, screen_height - window_height - 100);
            cv::resizeWindow(window_name, window_width, window_height);
            
            // Set window to not be always on top to avoid interference
            cv::setWindowProperty(window_name, cv::WND_PROP_TOPMOST, 0);
            
            display_enabled = true;
            std::cout << "Display window created and positioned away from game area" << std::endl;
            
        } catch (const std::exception& e) {
            std::cout << "Could not create display window: " << e.what() << std::endl;
            display_enabled = false;
        }
    }
    
    void showFrame(const cv::Mat& frame, bool save_results, bool& paused) {
        try {
            cv::imshow(window_name, frame);
            handleInput(paused, save_results, true);
        } catch (const std::exception& e) {
            std::cout << "Display error: " << e.what() << std::endl;
            display_enabled = false;
        }
    }
    
    void handleInput(bool& paused, bool save_results, bool update_display) {
        char key = cv::waitKey(update_display ? 1 : 50) & 0xFF;
        
        if (key == 'q' || key == 'Q' || key == 27) { // 'q', 'Q' or ESC
            running = false;
        } else if (key == 'p' || key == 'P') {
            paused = !paused;
            std::cout << (paused ? "\n[PAUSED] - Press P to resume" : "\n[RESUMED]") << std::endl;
        } else if (key == 's' || key == 'S') {
            if (save_results) {
                save_next_frame = true;
                std::cout << "\n[SAVE] - Screenshot will be saved on next frame" << std::endl;
            }
        } else if (key == 'd' || key == 'D') {
            display_enabled = !display_enabled;
            if (!display_enabled) {
                cv::destroyAllWindows();
                std::cout << "\n[DISPLAY OFF] - Press D to turn back on" << std::endl;
            } else {
                setupDisplayWindow();
                std::cout << "\n[DISPLAY ON]" << std::endl;
            }
        }
    }
    
    void addPerformanceOverlay(cv::Mat& frame, const cs2_detection::CaptureMetrics& capture_metrics,
                              const cs2_detection::PerformanceMetrics& detection_metrics,
                              int detection_count, const std::vector<int>& class_counts) {
        
        // Create semi-transparent overlay
        cv::Mat overlay = frame.clone();
        cv::rectangle(overlay, cv::Point(10, 10), cv::Point(450, 180), cv::Scalar(0, 0, 0), -1);
        cv::addWeighted(frame, 0.75, overlay, 0.25, 0, frame);
        
        // Performance metrics
        cv::putText(frame, "CS2 REAL-TIME DETECTION", 
                   cv::Point(20, 35), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
        
        cv::putText(frame, "FPS: " + std::to_string(static_cast<int>(capture_metrics.fps)), 
                   cv::Point(20, 60), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
        
        cv::putText(frame, "Latency: " + std::to_string(static_cast<int>(capture_metrics.total_ms + detection_metrics.total_ms)) + "ms",
                   cv::Point(20, 85), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        
        cv::putText(frame, "Players: " + std::to_string(detection_count),
                   cv::Point(20, 110), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);
        
        // Class breakdown (compact)
        if (detection_count > 0 && class_counts.size() >= 4) {
            std::string class_breakdown = "CT:" + std::to_string(class_counts[0]) + 
                                        " T:" + std::to_string(class_counts[2]) + 
                                        " +H:" + std::to_string(class_counts[1] + class_counts[3]);
            cv::putText(frame, class_breakdown,
                       cv::Point(20, 135), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);
        }
        
        // Performance indicator
        float total_latency = capture_metrics.total_ms + detection_metrics.total_ms;
        cv::Scalar perf_color = total_latency < 20 ? cv::Scalar(0, 255, 0) : 
                               total_latency < 33 ? cv::Scalar(0, 255, 255) : cv::Scalar(0, 0, 255);
        cv::circle(frame, cv::Point(400, 35), 10, perf_color, -1);
        
        // Controls hint (smaller)
        cv::putText(frame, "Q=Quit P=Pause D=Display S=Save",
                   cv::Point(20, 160), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(150, 150, 150), 1);
    }
    
    void printLiveStats(const cs2_detection::CaptureMetrics& capture_metrics,
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
    
    void printSessionStats() {
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
};
#endif // _WIN32

int main(int argc, char* argv[]) {
    try {
        std::string image_path = "cs2_screenshot.jpg";
        std::string model_path = "yolov8s_cs2.onnx";
        bool run_benchmark = false;
        bool run_realtime = false;
        int benchmark_runs = 10;
        float conf_threshold = 0.2f;
        float nms_threshold = 0.45f;
        
        // Parse command line arguments
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            
            if (arg == "--benchmark") {
                run_benchmark = true;
                if (i + 1 < argc && argv[i + 1][0] != '-') {
                    benchmark_runs = std::atoi(argv[i + 1]);
                    i++;
                }
            } else if (arg == "--realtime" || arg == "-r") {
                run_realtime = true;
            } else if (arg == "--model" || arg == "-m") {
                if (i + 1 < argc) {
                    model_path = argv[i + 1];
                    i++;
                }
            } else if (arg == "--conf") {
                if (i + 1 < argc) {
                    conf_threshold = std::stof(argv[i + 1]);
                    i++;
                }
            } else if (arg == "--nms") {
                if (i + 1 < argc) {
                    nms_threshold = std::stof(argv[i + 1]);
                    i++;
                }
            } else if (arg == "--help" || arg == "-h") {
                std::cout << "CS2 Player Detection - Usage:" << std::endl;
                std::cout << argv[0] << " [options] [image_path]" << std::endl;
                std::cout << "\nOptions:" << std::endl;
                std::cout << "  --realtime, -r      : Run real-time detection on CS2 window" << std::endl;
                std::cout << "  --benchmark [runs]  : Run benchmark mode (default: 10 runs)" << std::endl;
                std::cout << "  --model, -m [path]  : Model path (default: yolov8s_cs2.onnx)" << std::endl;
                std::cout << "  --conf [threshold]  : Confidence threshold (default: 0.2)" << std::endl;
                std::cout << "  --nms [threshold]   : NMS threshold (default: 0.45)" << std::endl;
                std::cout << "  --help, -h          : Show this help" << std::endl;
                std::cout << "\nClasses: CT, CT+Helmet, T, T+Helmet" << std::endl;
                std::cout << "\nExamples:" << std::endl;
                std::cout << "  " << argv[0] << " --realtime" << std::endl;
                std::cout << "  " << argv[0] << " --benchmark 20" << std::endl;
                std::cout << "  " << argv[0] << " --conf 0.3 screenshot.jpg" << std::endl;
                return 0;
            } else if (arg[0] != '-') {
                image_path = arg;
            }
        }
        
        std::cout << "=== CS2 PLAYER DETECTION SYSTEM ===" << std::endl;
        std::cout << "Model: " << model_path << std::endl;
        std::cout << "Confidence threshold: " << conf_threshold << std::endl;
        std::cout << "NMS threshold: " << nms_threshold << std::endl;
        
        if (run_realtime) {
            std::cout << "\n=== REAL-TIME DETECTION MODE ===" << std::endl;
            
#ifdef _WIN32
            try {
                CS2RealTimeDetector realtime_detector(model_path, true);
                realtime_detector.runRealTimeDetection(conf_threshold, nms_threshold, true, true);
            } catch (const std::exception& e) {
                std::cerr << "Real-time detection failed: " << e.what() << std::endl;
                return -1;
            }
#else
            std::cerr << "Real-time detection is only supported on Windows" << std::endl;
            return -1;
#endif
            
        } else if (run_benchmark) {
            std::cout << "\n=== BENCHMARK MODE ===" << std::endl;
            std::cout << "Loading image: " << image_path << std::endl;
            
            cv::Mat image = cv::imread(image_path);
            if (image.empty()) {
                std::cerr << "Error: Cannot load image " << image_path << std::endl;
                std::cerr << "Make sure the image file exists and is a valid format" << std::endl;
                return -1;
            }
            
            std::cout << "Image loaded! Size: " << image.cols << "x" << image.rows << std::endl;
            
            cs2_detection::YOLOv8Detector detector(model_path, true);
            detector.warmup(3);
            
            auto avg_metrics = detector.benchmark(image, benchmark_runs, conf_threshold, nms_threshold);
            
            std::cout << "\n=== PERFORMANCE ANALYSIS ===" << std::endl;
            if (avg_metrics.inference_ms < 20) {
                std::cout << "EXCELLENT: GPU acceleration working perfectly!" << std::endl;
            } else if (avg_metrics.inference_ms < 50) {
                std::cout << "GOOD: GPU acceleration working" << std::endl;
            } else {
                std::cout << "WARNING: Performance suboptimal - check GPU drivers" << std::endl;
            }
            
        } else {
            std::cout << "\n=== STATIC IMAGE DETECTION MODE ===" << std::endl;
            std::cout << "Loading image: " << image_path << std::endl;
            
            cv::Mat image = cv::imread(image_path);
            if (image.empty()) {
                std::cerr << "Error: Cannot load image " << image_path << std::endl;
                std::cerr << "Make sure the image file exists and is a valid format" << std::endl;
                return -1;
            }
            
            std::cout << "Image loaded! Size: " << image.cols << "x" << image.rows << std::endl;
            
            cs2_detection::YOLOv8Detector detector(model_path, true);
            detector.warmup(3);
            
            std::cout << "\n=== DETECTING CS2 PLAYERS ===" << std::endl;
            auto result = detector.detectWithMetrics(image, conf_threshold, nms_threshold);
            auto detections = result.first;
            auto metrics = result.second;
            
            metrics.print();
            
            cv::Mat result_image = image.clone();
            if (!detections.empty()) {
                detector.drawResults(result_image, detections);
                
                std::cout << "\n=== DETECTION RESULTS ===" << std::endl;
                std::cout << "Detected " << detections.size() << " CS2 players!" << std::endl;
                
                // Count by class
                std::vector<int> class_counts(4, 0);
                for (const auto& det : detections) {
                    if (det.class_id >= 0 && det.class_id < static_cast<int>(class_counts.size())) {
                        class_counts[det.class_id]++;
                    }
                    std::cout << "  " << det.class_name << " (conf: " 
                              << std::fixed << std::setprecision(3) << det.confidence 
                              << ") at [" << det.bbox.x << "," << det.bbox.y 
                              << "," << det.bbox.width << "," << det.bbox.height << "]" << std::endl;
                }
                
                std::cout << "\nClass breakdown:" << std::endl;
                std::vector<std::string> class_names = {"CT", "CT+Helmet", "T", "T+Helmet"};
                for (int i = 0; i < static_cast<int>(class_names.size()); i++) {
                    if (i < static_cast<int>(class_counts.size()) && class_counts[i] > 0) {
                        std::cout << "  " << class_names[i] << ": " << class_counts[i] << std::endl;
                    }
                }
                
            } else {
                std::cout << "\nNo CS2 players detected" << std::endl;
                std::cout << "Try lowering the confidence threshold with --conf 0.1" << std::endl;
            }
            
            // Save results
            std::string output_path = "cs2_result_" + 
                std::to_string(std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count()) + ".jpg";
            cv::imwrite(output_path, result_image);
            std::cout << "\nResult saved as: " << output_path << std::endl;
            
            if (!detections.empty()) {
                exportToJSON(detections, "cs2_detections.json");
            }
            
            // Performance analysis
            std::cout << "\n=== PERFORMANCE ANALYSIS ===" << std::endl;
            if (metrics.inference_ms < 20) {
                std::cout << "EXCELLENT: GPU acceleration working perfectly!" << std::endl;
            } else if (metrics.inference_ms < 50) {
                std::cout << "GOOD: GPU acceleration working" << std::endl;
            } else {
                std::cout << "WARNING: Performance suboptimal - check GPU drivers" << std::endl;
            }
            
            // Display
            try {
                cv::imshow("CS2 Player Detection Results", result_image);
                std::cout << "\nPress any key to exit..." << std::endl;
                cv::waitKey(0);
                cv::destroyAllWindows();
            } catch (const std::exception&) {
                std::cout << "Display not available (headless mode)" << std::endl;
            }
        }
        
        std::cout << "\n=== CS2 DETECTION COMPLETE ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        std::cerr << "Please check:" << std::endl;
        std::cerr << "  1. Model file exists: yolov8s_cs2.onnx" << std::endl;
        std::cerr << "  2. ONNX Runtime libraries are properly installed" << std::endl;
        std::cerr << "  3. CUDA drivers are installed (for GPU acceleration)" << std::endl;
        return -1;
    }
    
    return 0;
}