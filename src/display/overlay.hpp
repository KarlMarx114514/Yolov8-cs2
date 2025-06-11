#pragma once
#include "core/detection_types.hpp"
#include "capture/screen_capture.hpp"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

namespace cs2_display {

/**
 * @brief Utility class for adding performance overlays to detection frames
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
     */
    static void addPerformanceOverlay(cv::Mat& frame, 
                                    const cs2_detection::CaptureMetrics& capture_metrics,
                                    const cs2_detection::PerformanceMetrics& detection_metrics,
                                    int detection_count, 
                                    const std::vector<int>& class_counts);
};

/**
 * @brief Manages the OpenCV display window for real-time detection
 */
class DisplayWindow {
public:
    /**
     * @brief Constructor
     * @param window_name Name/title of the OpenCV window
     */
    explicit DisplayWindow(const std::string& window_name);
    
    /**
     * @brief Destructor - automatically closes window if open
     */
    ~DisplayWindow();
    
    /**
     * @brief Initialize and setup the display window
     * @return true if window was created successfully, false otherwise
     */
    bool setupWindow();
    
    /**
     * @brief Display a frame in the window
     * @param frame The frame to display
     */
    void showFrame(const cv::Mat& frame);
    
    /**
     * @brief Handle keyboard input and basic window controls
     * @param paused Reference to pause state (modified by this function)
     * @param save_results Whether save functionality is enabled
     * @param update_display Whether to update display (affects input polling)
     * @return The key that was pressed (for handling by other modules)
     */
    char handleInput(bool& paused, bool save_results, bool update_display = true);
    
    /**
     * @brief Toggle display window on/off
     */
    void toggleDisplay();
    
    /**
     * @brief Check if display is currently enabled
     * @return true if display window is active
     */
    bool isEnabled() const { 
        return display_enabled; 
    }
    
#ifdef _WIN32
    /**
     * @brief Get Windows handle to the OpenCV window
     * @return HWND handle, or nullptr if window not found/created
     */
    HWND getWindowHandle() const;
#endif
    
private:
    std::string window_name;
    bool display_enabled;
    
    // Disable copy constructor and assignment operator
    DisplayWindow(const DisplayWindow&) = delete;
    DisplayWindow& operator=(const DisplayWindow&) = delete;
};

} // namespace cs2_display