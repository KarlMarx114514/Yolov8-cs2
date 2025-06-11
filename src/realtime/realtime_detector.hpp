#pragma once

#include "core/detector.hpp"
#include "capture/screen_capture.hpp"
#include "control/mouse_controller.hpp"
#include "display/overlay.hpp"
#include <memory>
#include <atomic>
#include <chrono>

#ifdef _WIN32

namespace cs2_realtime {

/**
 * @brief Real-time CS2 player detection system with mouse control integration
 * 
 * This class orchestrates the entire real-time detection pipeline:
 * - Screen capture from CS2 window
 * - AI-based player detection
 * - Mouse control for aiming assistance
 * - Performance monitoring and display
 * - User input handling
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
     * @brief Run the real-time detection loop
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
    int window_recheck_counter;
    
    // Performance tracking
    int total_frames;
    int detected_frames;
    std::chrono::high_resolution_clock::time_point session_start;
    
    /**
     * @brief Initialize the display window and link it to mouse controller
     */
    void setupDisplayWindow();
    
    /**
     * @brief Display a frame and handle basic input
     * @param frame The frame to display
     * @param save_results Whether save functionality is enabled
     * @param paused Reference to pause state
     */
    void showFrame(const cv::Mat& frame, bool save_results, bool& paused);
    
    /**
     * @brief Handle user input for all system controls
     * @param paused Reference to pause state (modified by function)
     * @param save_results Whether save functionality is enabled
     * @param update_display Whether to update display (affects input polling)
     */
    void handleInput(bool& paused, bool save_results, bool update_display);
    
    /**
     * @brief Print live performance statistics (called periodically)
     * @param capture_metrics Screen capture performance data
     * @param detection_metrics AI detection performance data
     * @param current_detections Number of current detections
     */
    void printLiveStats(const cs2_detection::CaptureMetrics& capture_metrics,
                       const cs2_detection::PerformanceMetrics& detection_metrics,
                       int current_detections);
    
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