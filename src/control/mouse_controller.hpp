#pragma once
#include "core/detection_types.hpp"
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

#ifdef _WIN32
#include <windows.h>

namespace cs2_control {
    
class MouseController {
public:
    enum class TargetPriority {
        CLOSEST_TO_CENTER,
        CLOSEST_TO_CURSOR,
        HIGHEST_CONFIDENCE,
        NEAREST_ENEMY
    };
    
    enum class InputMethod {
        AUTO_DETECT,
        RAW_INPUT_ONLY,
        ABSOLUTE_ONLY,
        HYBRID
    };
    
    MouseController();
    
    // Core functionality
    void setActive(bool active);
    void aimAtTarget(const std::vector<cs2_detection::Detection>& detections, 
                    TargetPriority priority = TargetPriority::CLOSEST_TO_CENTER);
    
    // Configuration
    void setSensitivity(float scale);
    void setInputMethod(InputMethod method);
    void setSmoothness(int steps, int delay_ms);
    void setTargetOffset(float offset_x, float offset_y);
    void setDetectionWindow(HWND window);
    
    // Window management
    bool findCS2Window();
    bool recheckCS2Window();
    void manualWindowSelection();
    
    // Testing and debugging
    void testInputMethods();
    void testBasicMovement();
    void testScreenCenter();
    void toggleDebugMode();
    
private:
    // Window handles and state
    HWND cs2_window;
    HWND detection_window;
    cv::Rect capture_region;
    bool is_active;
    bool debug_mode;
    
    // Input method detection
    bool use_raw_input;
    bool force_absolute_mode;
    
    // Movement parameters
    int movement_steps;
    int step_delay_ms;
    
    // Target selection parameters
    float target_offset_x;
    float target_offset_y;
    
    // Mouse sensitivity calibration
    float mouse_sensitivity_scale;
    
    // Window validation
    DWORD cs2_process_id;
    std::string cs2_window_title;
    
    // Movement tracking for relative input
    cv::Point2f last_target;
    
    // Private methods
    bool validateCS2Window();
    bool isInGameplay();
    void updateCaptureRegion();
    cv::Point2f detectionToScreen(const cs2_detection::Detection& detection);
    bool moveMouseRawInput(cv::Point2f target);
    bool moveMouseAbsolute(cv::Point2f target);
    bool moveMouseHybrid(cv::Point2f target);
    void smoothMoveTo(cv::Point2f target);
    cs2_detection::Detection selectTarget(const std::vector<cs2_detection::Detection>& detections, 
                                        TargetPriority priority);
    void calibrateMouseSensitivity();
};

class CS2MouseIntegration {
public:
    CS2MouseIntegration();
    void toggleMouseControl();
    void processDetections(const std::vector<cs2_detection::Detection>& detections);
    void setupMouseControl();
    void setDetectionWindowHandle(HWND window_handle);
    bool recheckCS2Window();
    void refreshWindowDetection();
    void manualWindowSelection();
    void testInputMethods();
    void setInputMode(int mode);
    void adjustSensitivity(bool increase);
    void setSensitivityPreset(float sensitivity);
    void toggleDebugMode();
    
private:
    MouseController mouse_controller;
    std::vector<cs2_detection::Detection> previous_detections;
    bool mouse_enabled;
};

} // namespace cs2_control

#endif // _WIN32