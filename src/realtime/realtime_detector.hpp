#pragma once

#include "core/detector.hpp"
#include "capture/screen_capture.hpp"
#include "control/mouse_controller.hpp"
#include "display/overlay.hpp"
#include <memory>
#include <atomic>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>

#ifdef _WIN32

namespace cs2_realtime {

// Thread communication structures
struct CaptureData {
    cv::Mat frame;
    cs2_detection::CaptureMetrics metrics;
    std::chrono::high_resolution_clock::time_point timestamp;
};

struct DetectionFrame {
    std::vector<cs2_detection::Detection> detections;
    cv::Mat display_frame;
    cs2_detection::CaptureMetrics capture_metrics;
    cs2_detection::PerformanceMetrics detection_metrics;
    std::chrono::high_resolution_clock::time_point timestamp;
};

/**
 * @brief Multi-threaded real-time CS2 player detection system with optimized mouse control
 * 
 * This class orchestrates a high-performance multi-threaded detection pipeline:
 * - Capture Thread: High-frequency screen capture (~120 FPS)
 * - Detection Thread: AI-based player detection processing  
 * - Mouse Thread: High-priority mouse control for minimal latency
 * - Main Thread: OpenCV GUI operations (display/input) and coordination
 * 
 * Key optimizations:
 * - Separated mouse control from heavy AI processing
 * - OpenCV GUI operations kept on main thread (thread-safe)
 * - Lock-free queues with size limits to prevent memory buildup
 * - Thread priorities to ensure mouse responsiveness
 * - Minimal synchronization overhead
 */
class CS2RealTimeDetector {
public:
    /**
     * @brief Constructor
     * @param model_path Path to the ONNX model file
     * @param use_gpu Whether to use GPU acceleration (requires CUDA)
     * @throws std::runtime_error if initialization fails
     */
    CS2RealTimeDetector(const std::string& model_path, bool use_gpu = true);
    
    /**
     * @brief Run the multi-threaded real-time detection loop
     * @param conf_threshold Confidence threshold for detections (0.0-1.0)
     * @param nms_threshold Non-maximum suppression threshold (0.0-1.0)
     * @param show_display Whether to show the detection overlay window
     * @param save_results Whether to enable screenshot/JSON export functionality
     */
    void runRealTimeDetection(float conf_threshold = 0.2f, 
                            float nms_threshold = 0.45f, 
                            bool show_display = true, 
                            bool save_results = false);
    
private:
    // Core detection components
    std::unique_ptr<cs2_detection::YOLOv8Detector> detector;
    std::unique_ptr<cs2_detection::WindowsScreenCapture> screen_capture;
    
    // Control and display components
    cs2_control::CS2MouseIntegration mouse_integration;
    std::unique_ptr<cs2_display::DisplayWindow> display_window;
    
    // Threading and state management
    std::atomic<bool> running;
    std::atomic<bool> save_next_frame;
    std::atomic<bool> paused;
    int window_recheck_counter;
    
    // Performance tracking
    std::atomic<int> total_frames;
    std::atomic<int> detected_frames;
    std::chrono::high_resolution_clock::time_point session_start;
    
    // Thread communication - Capture to Detection
    std::queue<CaptureData> capture_queue;
    std::mutex capture_mutex;
    std::condition_variable capture_cv;
    
    // Thread communication - Detection to Mouse Control
    std::queue<std::vector<cs2_detection::Detection>> mouse_queue;
    std::mutex mouse_mutex;
    std::condition_variable mouse_cv;
    
    // Thread communication - Detection to Main Thread (Display)
    std::queue<DetectionFrame> display_queue;
    std::mutex display_mutex;
    std::condition_variable display_cv;
    
    /**
     * @brief Screen capture thread - runs at high frequency
     * @param conf_threshold Detection confidence threshold
     * @param nms_threshold NMS threshold
     */
    void captureThread(float conf_threshold, float nms_threshold);
    
    /**
     * @brief AI detection processing thread
     * @param conf_threshold Detection confidence threshold  
     * @param nms_threshold NMS threshold
     */
    void detectionThread(float conf_threshold, float nms_threshold);
    
    /**
     * @brief High-priority mouse control thread for minimal latency
     */
    void mouseControlThread();
    
    /**
     * @brief Handle display and input in main thread (OpenCV thread-safe)
     * @param save_results Whether save functionality is enabled
     */
    void handleDisplayAndInput(bool save_results);
    
    /**
     * @brief Initialize the display window and link it to mouse controller
     */
    void setupDisplayWindow();
    
    /**
     * @brief Handle user input for all system controls (runs in main thread)
     * @param save_results Whether save functionality is enabled
     * @param update_display Whether to update display (affects input polling)
     */
    void handleInput(bool save_results, bool update_display);
    
    /**
     * @brief Print control help information
     */
    void printControls();
    
    /**
     * @brief Print live performance statistics (called periodically)
     */
    void printLiveStats();
    
    /**
     * @brief Print final session statistics (called at end)
     */
    void printSessionStats();
    
    // Disable copy constructor and assignment operator
    CS2RealTimeDetector(const CS2RealTimeDetector&) = delete;
    CS2RealTimeDetector& operator=(const CS2RealTimeDetector&) = delete;
};

} // namespace cs2_realtime

#endif // _WIN32