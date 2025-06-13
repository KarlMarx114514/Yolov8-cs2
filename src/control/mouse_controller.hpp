#pragma once
#include "core/detection_types.hpp"
#include <vector>
#include <string>
#include <chrono>
#include <opencv2/opencv.hpp>

#ifdef _WIN32
#include <windows.h>

namespace cs2_control {

// Team enumeration
enum class Team {
    NONE = -1,
    CT = 0,
    T = 2  // T starts at class_id 2
};

// PID Controller for smooth mouse movement
class PIDController {
public:
    PIDController(float kp = 1.0f, float ki = 0.0f, float kd = 0.1f);
    cv::Point2f update(cv::Point2f error, float dt);
    void reset();
    void setGains(float kp, float ki, float kd);
    void getGains(float& kp, float& ki, float& kd) const;
    
private:
    float kp_, ki_, kd_;
    cv::Point2f previous_error_;
    cv::Point2f integral_;
    bool first_update_;
};
    
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
    
    // Team management
    void setPlayerTeam(Team team);
    Team getPlayerTeam() const { return player_team_; }
    void toggleTeam();
    
    // Configuration
    void setSensitivity(float scale);
    void setInputMethod(InputMethod method);
    void setSmoothness(int steps, int delay_ms);
    void setTargetOffset(float offset_x, float offset_y);
    void setDetectionWindow(HWND window);
    
    // PID Controller configuration
    void adjustPIDGains(char adjustment); // 'p', 'i', 'd' for increase, 'P', 'I', 'D' for decrease
    void resetPID();
    void printPIDStatus();
    
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
    
    // Team tracking
    Team player_team_;
    cs2_detection::Detection current_target_;
    float target_switch_threshold_;
    
    // Performance optimization - window validation caching
    bool window_validated;
    std::chrono::steady_clock::time_point last_window_check;
    
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
    
    // PID Controller for smooth movement
    PIDController pid_controller_;
    std::chrono::high_resolution_clock::time_point last_update_time_;
    bool pid_initialized_;
    
    // Movement tracking for relative input
    cv::Point2f last_target;
    cv::Point2f current_position_;
    
    // Window validation
    DWORD cs2_process_id;
    std::string cs2_window_title;
    
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
    cv::Point2f getCurrentMousePosition();
    
    // Team-based filtering and prioritization
    std::vector<cs2_detection::Detection> filterEnemyTargets(const std::vector<cs2_detection::Detection>& detections);
    bool isEnemyTarget(const cs2_detection::Detection& detection);
    bool hasHelmet(const cs2_detection::Detection& detection);
    float calculateTargetPriority(const cs2_detection::Detection& detection, cv::Point2f screen_center);
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
    void adjustPIDGains(char adjustment);
    void resetPID();
    void setPlayerTeam(Team team);
    void toggleTeam();
    
private:
    MouseController mouse_controller;
    std::vector<cs2_detection::Detection> previous_detections;
    bool mouse_enabled;
};

} // namespace cs2_control

#endif // _WIN32