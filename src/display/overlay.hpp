#pragma once
#include "core/detection_types.hpp"
#include "capture/screen_capture.hpp"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <mutex>

#ifdef _WIN32
#include <windows.h>
#endif

namespace cs2_display {

/**
 * @brief Utility class for adding performance overlays to detection frames
 * 
 * Optimized for multi-threaded environments when called from main thread
 */
class PerformanceOverlay {
public:
    /**
     * @brief Add performance HUD overlay to a frame
     * @param frame The frame to add overlay to (modified in place)
     * @param capture_metrics Screen capture performance metrics
     * @param detection_metrics AI detection performance metrics
     * @param detection_count Current number of detections
     * @param class_counts Vector containing count of each detection class
     * 
     * @note Should be called from main thread when using with OpenCV windows
     * @note Will gracefully handle OpenCV exceptions to prevent crashes
     */
    static void addPerformanceOverlay(cv::Mat& frame, 
                                    const cs2_detection::CaptureMetrics& capture_metrics,
                                    const cs2_detection::PerformanceMetrics& detection_metrics,
                                    int detection_count, 
                                    const std::vector<int>& class_counts);
};

/**
 * @brief OpenCV display window manager for real-time detection
 * 
 * Handles window lifecycle, input processing, and frame display.
 * Designed for main-thread usage to avoid OpenCV threading issues.
 */
class DisplayWindow {
public:
    /**
     * @brief Constructor
     * @param window_name Name/title of the OpenCV window
     */
    explicit DisplayWindow(const std::string& window_name);
    
    /**
     * @brief Destructor - automatically closes window if open (thread-safe)
     */
    ~DisplayWindow();
    
    /**
     * @brief Initialize and setup the display window (main thread)
     * @return true if window was created successfully, false otherwise
     */
    bool setupWindow();
    
    /**
     * @brief Display a frame in the window (main thread)
     * @param frame The frame to display
     * 
     * @note Should be called from main thread to avoid OpenCV threading issues
     * @note Will handle OpenCV exceptions gracefully to prevent crashes
     */
    void showFrame(const cv::Mat& frame);
    
    /**
     * @brief Handle keyboard input and basic window controls (main thread)
     * @param paused Reference to pause state (modified by this function)
     * @param save_results Whether save functionality is enabled
     * @param update_display Whether to update display (affects input polling)
     * @return The key that was pressed (for handling by other modules)
     * 
     * @note Should be called from main thread for proper OpenCV event handling
     */
    char handleInput(bool& paused, bool save_results, bool update_display = true);
    
    /**
     * @brief Toggle display window on/off (main thread)
     * 
     * @note When disabled, provides performance optimization for mouse-only mode
     */
    void toggleDisplay();
    
    /**
     * @brief Check if display is currently enabled (thread-safe)
     * @return true if display window is active
     */
    bool isEnabled() const { 
        std::lock_guard<std::mutex> lock(window_mutex);
        return display_enabled; 
    }
    
#ifdef _WIN32
    /**
     * @brief Get Windows handle to the OpenCV window (thread-safe)
     * @return HWND handle, or nullptr if window not found/created
     */
    HWND getWindowHandle() const;
#endif
    
private:
    std::string window_name;
    bool display_enabled;
    mutable std::mutex window_mutex;  // Protects all window operations
    
    // Disable copy constructor and assignment operator
    DisplayWindow(const DisplayWindow&) = delete;
    DisplayWindow& operator=(const DisplayWindow&) = delete;
};

} // namespace cs2_display