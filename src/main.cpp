#include "core/detector.hpp"
#include "capture/screen_capture.hpp"
#include "export/json_export.hpp"
#include "realtime/realtime_detector.hpp"
#include <iostream>
#include <string>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <opencv2/opencv.hpp>

/**
 * @brief Print usage information
 */
void printUsage(const char* program_name) {
    std::cout << "CS2 Player Detection with Mouse Control - Usage:" << std::endl;
    std::cout << program_name << " [options] [image_path]" << std::endl;
    std::cout << "\nOptions:" << std::endl;
    std::cout << "  --realtime, -r      : Run real-time detection on CS2 window" << std::endl;
    std::cout << "  --benchmark [runs]  : Run benchmark mode (default: 10 runs)" << std::endl;
    std::cout << "  --model, -m [path]  : Model path (default: yolov8s_cs2.onnx)" << std::endl;
    std::cout << "  --conf [threshold]  : Confidence threshold (default: 0.2)" << std::endl;
    std::cout << "  --nms [threshold]   : NMS threshold (default: 0.45)" << std::endl;
    std::cout << "  --no-display       : Disable display window (realtime mode only)" << std::endl;
    std::cout << "  --no-save          : Disable save functionality" << std::endl;
    std::cout << "  --help, -h          : Show this help" << std::endl;
    std::cout << "\nDetection Classes: CT, CT+Helmet, T, T+Helmet" << std::endl;
    std::cout << "\nExamples:" << std::endl;
    std::cout << "  " << program_name << " --realtime" << std::endl;
    std::cout << "  " << program_name << " --realtime --no-display" << std::endl;
    std::cout << "  " << program_name << " --benchmark 20" << std::endl;
    std::cout << "  " << program_name << " --conf 0.3 screenshot.jpg" << std::endl;
    std::cout << "  " << program_name << " --model custom_model.onnx --realtime" << std::endl;
}

/**
 * @brief Run benchmark mode on a static image
 */
int runBenchmark(const std::string& image_path, const std::string& model_path, 
                int benchmark_runs, float conf_threshold, float nms_threshold) {
    
    std::cout << "\n=== BENCHMARK MODE ===" << std::endl;
    std::cout << "Loading image: " << image_path << std::endl;
    
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Error: Cannot load image " << image_path << std::endl;
        std::cerr << "Make sure the image file exists and is a valid format" << std::endl;
        return -1;
    }
    
    std::cout << "Image loaded! Size: " << image.cols << "x" << image.rows << std::endl;
    
    try {
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
        
    } catch (const std::exception& e) {
        std::cerr << "Benchmark failed: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}

/**
 * @brief Run static image detection mode
 */
int runStaticDetection(const std::string& image_path, const std::string& model_path,
                      float conf_threshold, float nms_threshold) {
    
    std::cout << "\n=== STATIC IMAGE DETECTION MODE ===" << std::endl;
    std::cout << "Loading image: " << image_path << std::endl;
    
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Error: Cannot load image " << image_path << std::endl;
        std::cerr << "Make sure the image file exists and is a valid format" << std::endl;
        return -1;
    }
    
    std::cout << "Image loaded! Size: " << image.cols << "x" << image.rows << std::endl;
    
    try {
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
            
            // Count detections by class
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
        
        // Save result image
        std::string output_path = "cs2_result_" + 
            std::to_string(std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count()) + ".jpg";
        cv::imwrite(output_path, result_image);
        std::cout << "\nResult saved as: " << output_path << std::endl;
        
        // Export detections to JSON
        if (!detections.empty()) {
            cs2_export::exportToJSON(detections, "cs2_detections.json");
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
        
        // Display result if possible
        try {
            cv::imshow("CS2 Player Detection Results", result_image);
            std::cout << "\nPress any key to exit..." << std::endl;
            cv::waitKey(0);
            cv::destroyAllWindows();
        } catch (const std::exception&) {
            std::cout << "Display not available (headless mode)" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Detection failed: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}

/**
 * @brief Run real-time detection mode
 */
int runRealTimeDetection(const std::string& model_path, float conf_threshold, 
                        float nms_threshold, bool show_display, bool save_results) {
    
    std::cout << "\n=== REAL-TIME DETECTION MODE ===" << std::endl;
    
#ifdef _WIN32
    try {
        cs2_realtime::CS2RealTimeDetector realtime_detector(model_path, true);
        realtime_detector.runRealTimeDetection(conf_threshold, nms_threshold, show_display, save_results);
    } catch (const std::exception& e) {
        std::cerr << "Real-time detection failed: " << e.what() << std::endl;
        std::cerr << "Please check:" << std::endl;
        std::cerr << "  1. CS2 is running and visible" << std::endl;
        std::cerr << "  2. Model file exists: " << model_path << std::endl;
        std::cerr << "  3. ONNX Runtime libraries are available" << std::endl;
        std::cerr << "  4. No antivirus blocking screen capture" << std::endl;
        return -1;
    }
#else
    std::cerr << "Real-time detection is only supported on Windows" << std::endl;
    std::cerr << "This feature requires Windows-specific APIs for:" << std::endl;
    std::cerr << "  - DirectX/GDI screen capture" << std::endl;
    std::cerr << "  - Mouse control integration" << std::endl;
    std::cerr << "  - CS2 window detection" << std::endl;
    return -1;
#endif
    
    return 0;
}

/**
 * @brief Main entry point
 */
int main(int argc, char* argv[]) {
    try {
        // Default parameters
        std::string image_path = "cs2_screenshot.jpg";
        std::string model_path = "yolov8s_cs2.onnx";
        bool run_benchmark = false;
        bool run_realtime = false;
        int benchmark_runs = 10;
        float conf_threshold = 0.2f;
        float nms_threshold = 0.45f;
        bool show_display = true;
        bool save_results = true;
        
        // Parse command line arguments
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            
            if (arg == "--help" || arg == "-h") {
                printUsage(argv[0]);
                return 0;
            } else if (arg == "--benchmark") {
                run_benchmark = true;
                // Check for optional run count
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
                } else {
                    std::cerr << "Error: --model requires a path argument" << std::endl;
                    return -1;
                }
            } else if (arg == "--conf") {
                if (i + 1 < argc) {
                    conf_threshold = std::stof(argv[i + 1]);
                    if (conf_threshold < 0.0f || conf_threshold > 1.0f) {
                        std::cerr << "Error: Confidence threshold must be between 0.0 and 1.0" << std::endl;
                        return -1;
                    }
                    i++;
                } else {
                    std::cerr << "Error: --conf requires a threshold value" << std::endl;
                    return -1;
                }
            } else if (arg == "--nms") {
                if (i + 1 < argc) {
                    nms_threshold = std::stof(argv[i + 1]);
                    if (nms_threshold < 0.0f || nms_threshold > 1.0f) {
                        std::cerr << "Error: NMS threshold must be between 0.0 and 1.0" << std::endl;
                        return -1;
                    }
                    i++;
                } else {
                    std::cerr << "Error: --nms requires a threshold value" << std::endl;
                    return -1;
                }
            } else if (arg == "--no-display") {
                show_display = false;
            } else if (arg == "--no-save") {
                save_results = false;
            } else if (arg[0] != '-') {
                // Non-option argument - treat as image path
                image_path = arg;
            } else {
                std::cerr << "Error: Unknown option: " << arg << std::endl;
                std::cerr << "Use --help for usage information" << std::endl;
                return -1;
            }
        }
        
        // Validate mutually exclusive options
        if (run_benchmark && run_realtime) {
            std::cerr << "Error: Cannot run both benchmark and real-time modes simultaneously" << std::endl;
            return -1;
        }
        
        // Print configuration
        std::cout << "=== CS2 PLAYER DETECTION SYSTEM WITH MOUSE CONTROL ===" << std::endl;
        std::cout << "Model: " << model_path << std::endl;
        std::cout << "Confidence threshold: " << conf_threshold << std::endl;
        std::cout << "NMS threshold: " << nms_threshold << std::endl;
        std::cout << "Version: Modular Architecture" << std::endl;
        
        // Validate model file exists
        cv::Mat test_read = cv::imread(model_path, cv::IMREAD_UNCHANGED);
        if (test_read.empty()) {
            // This is not a perfect check for ONNX files, but it's a basic existence check
            std::ifstream model_file(model_path);
            if (!model_file.good()) {
                std::cerr << "Warning: Model file may not exist: " << model_path << std::endl;
                std::cerr << "Make sure the ONNX model file is in the correct location" << std::endl;
            }
        }
        
        // Run the appropriate mode
        int result = 0;
        
        if (run_realtime) {
            result = runRealTimeDetection(model_path, conf_threshold, nms_threshold, show_display, save_results);
        } else if (run_benchmark) {
            result = runBenchmark(image_path, model_path, benchmark_runs, conf_threshold, nms_threshold);
        } else {
            result = runStaticDetection(image_path, model_path, conf_threshold, nms_threshold);
        }
        
        if (result == 0) {
            std::cout << "\n=== CS2 DETECTION COMPLETE ===" << std::endl;
        }
        
        return result;
        
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        std::cerr << "\nPlease check:" << std::endl;
        std::cerr << "  1. Model file exists: yolov8s_cs2.onnx" << std::endl;
        std::cerr << "  2. ONNX Runtime libraries are properly installed" << std::endl;
        std::cerr << "  3. CUDA drivers are installed (for GPU acceleration)" << std::endl;
        std::cerr << "  4. OpenCV libraries are available" << std::endl;
        std::cerr << "  5. Required DLLs are in the executable directory" << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown fatal error occurred" << std::endl;
        return -1;
    }
}