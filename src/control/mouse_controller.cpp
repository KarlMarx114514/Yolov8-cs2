#include "control/mouse_controller.hpp"
#include <iostream>
#include <thread>
#include <chrono>
#include <algorithm>
#include <limits>
#include <iomanip>

#ifdef _WIN32

namespace cs2_control {

// PID Controller implementation
PIDController::PIDController(float kp, float ki, float kd) 
    : kp_(kp), ki_(ki), kd_(kd), previous_error_(0, 0), integral_(0, 0), first_update_(true) {
}

cv::Point2f PIDController::update(cv::Point2f error, float dt) {
    if (first_update_) {
        previous_error_ = error;
        first_update_ = false;
        return error * kp_; // Only proportional term on first update
    }
    
    // Proportional term
    cv::Point2f proportional = error * kp_;
    
    // Integral term with windup protection
    integral_ += error * dt;
    
    // Limit integral windup (prevent accumulation when error is large)
    float integral_limit = 50.0f;
    if (cv::norm(integral_) > integral_limit) {
        integral_ = integral_ * (integral_limit / cv::norm(integral_));
    }
    cv::Point2f integral_term = integral_ * ki_;
    
    // Derivative term
    cv::Point2f derivative = (error - previous_error_) / dt * kd_;
    
    previous_error_ = error;
    
    return proportional + integral_term + derivative;
}

void PIDController::reset() {
    previous_error_ = cv::Point2f(0, 0);
    integral_ = cv::Point2f(0, 0);
    first_update_ = true;
}

void PIDController::setGains(float kp, float ki, float kd) {
    kp_ = kp;
    ki_ = ki;
    kd_ = kd;
}

void PIDController::getGains(float& kp, float& ki, float& kd) const {
    kp = kp_;
    ki = ki_;
    kd = kd_;
}

// MouseController implementation
MouseController::MouseController() 
    : cs2_window(nullptr)
    , detection_window(nullptr)
    , is_active(false)
    , debug_mode(false)  // Start with debug off for performance
    , player_team_(Team::NONE)
    , target_switch_threshold_(0.05f)  // 5% threshold for target switching
    , auto_fire_enabled_(false)
    , fire_cooldown_ms_(800.0f)  // 0.8 seconds
    , fire_distance_threshold_(15.0f)  // pixels - very close to target
    , use_raw_input(true)
    , force_absolute_mode(false)
    , movement_steps(2)   // Reduced steps for faster movement
    , step_delay_ms(0)    // No delay for maximum speed
    , target_offset_x(0.0f)
    , target_offset_y(-0.2f)
    , mouse_sensitivity_scale(10.0f)
    , cs2_process_id(0)
    , last_target(-1, -1)
    , window_validated(false)
    , last_window_check(std::chrono::steady_clock::now())
    , pid_controller_(0.84f, 3.84f, 0.01f)  // Optimized PID gains for mouse control
    , pid_initialized_(false)
    , direct_move_done_(false)
    , last_target_position_(-1, -1)
    , current_position_(0, 0)
{
    findCS2Window();
    calibrateMouseSensitivity();
    
    // Initialize empty current target
    current_target_.confidence = 0.0f;
    current_target_.class_id = -1;
}

void MouseController::setPlayerTeam(Team team) {
    player_team_ = team;
    current_target_.confidence = 0.0f; // Reset current target when team changes
    
    std::string team_name = (team == Team::CT) ? "CT" : (team == Team::T) ? "T" : "NONE";
    std::cout << "Player team set to: " << team_name << std::endl;
    if (debug_mode && team != Team::NONE) {
        std::cout << "Now targeting: " << ((team == Team::CT) ? "T and T+Helmet" : "CT and CT+Helmet") << std::endl;
    }
}

void MouseController::toggleTeam() {
    if (player_team_ == Team::CT) {
        setPlayerTeam(Team::T);
    } else if (player_team_ == Team::T) {
        setPlayerTeam(Team::CT);
    } else {
        setPlayerTeam(Team::CT); // Default to CT if none selected
    }
}

std::vector<cs2_detection::Detection> MouseController::filterEnemyTargets(const std::vector<cs2_detection::Detection>& detections) {
    if (player_team_ == Team::NONE) {
        return detections; // Return all if no team set
    }
    
    std::vector<cs2_detection::Detection> enemies;
    for (const auto& detection : detections) {
        if (isEnemyTarget(detection)) {
            enemies.push_back(detection);
        }
    }
    
    if (debug_mode && !enemies.empty()) {
        std::cout << "Filtered " << enemies.size() << " enemy targets from " << detections.size() << " total detections" << std::endl;
    }
    
    return enemies;
}

bool MouseController::isEnemyTarget(const cs2_detection::Detection& detection) {
    if (player_team_ == Team::NONE) {
        return true; // Target all if no team set
    }
    
    // Class IDs: 0=CT, 1=CT+Helmet, 2=T, 3=T+Helmet
    if (player_team_ == Team::CT) {
        return detection.class_id == 2 || detection.class_id == 3; // Target T and T+Helmet
    } else if (player_team_ == Team::T) {
        return detection.class_id == 0 || detection.class_id == 1; // Target CT and CT+Helmet
    }
    
    return false;
}

bool MouseController::hasHelmet(const cs2_detection::Detection& detection) {
    return detection.class_id == 1 || detection.class_id == 3; // CT+Helmet or T+Helmet
}

float MouseController::calculateTargetPriority(const cs2_detection::Detection& detection, cv::Point2f screen_center) {
    float priority = detection.confidence;
    
    // Helmet targets get significant priority boost
    if (hasHelmet(detection)) {
        priority += 0.3f; // 30% boost for helmet targets
    }
    
    // Distance factor (closer to center gets slight boost)
    cv::Point2f det_center(detection.bbox.x + detection.bbox.width / 2.0f,
                          detection.bbox.y + detection.bbox.height / 2.0f);
    float distance = static_cast<float>(cv::norm(det_center - screen_center));
    float normalized_distance = distance / (screen_center.x + screen_center.y); // Rough normalization
    priority += (1.0f - normalized_distance) * 0.1f; // Small distance boost
    
    return priority;
}

void MouseController::setDetectionWindow(HWND window) {
    detection_window = window;
    if (debug_mode) {
        std::cout << "Detection window handle stored: " << window << std::endl;
    }
}

cv::Point2f MouseController::getCurrentMousePosition() {
    POINT current_pos;
    if (GetCursorPos(&current_pos)) {
        return cv::Point2f(static_cast<float>(current_pos.x), static_cast<float>(current_pos.y));
    }
    return current_position_; // Return cached position if GetCursorPos fails
}

bool MouseController::isInGameplay() {
    if (!cs2_window) return false;
    
    // Cache window title checks to avoid frequent WinAPI calls
    static std::string cached_title;
    static auto last_title_check = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    
    if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_title_check).count() > 1000) {
        char title[256];
        if (GetWindowTextA(cs2_window, title, sizeof(title))) {
            cached_title = title;
            last_title_check = now;
        }
    }
    
    if (!cached_title.empty()) {
        // Menu indicators (use absolute positioning)
        if (cached_title.find("Menu") != std::string::npos ||
            cached_title.find("Main Menu") != std::string::npos ||
            cached_title.find("Loading") != std::string::npos) {
            return false;
        }
        
        // Gameplay indicators (use raw input)
        if (cached_title.find("de_") != std::string::npos ||  // Map names
            cached_title.find("cs_") != std::string::npos ||
            cached_title.find("Competitive") != std::string::npos ||
            cached_title.find("Casual") != std::string::npos ||
            cached_title.find("Deathmatch") != std::string::npos) {
            return true;
        }
    }
    
    // Default assumption: if CS2 is running and we can't tell, assume gameplay
    return true;
}

void MouseController::calibrateMouseSensitivity() {
    // Start with a more aggressive default for faster movement
    mouse_sensitivity_scale = 3.0f;
    
    if (debug_mode) {
        std::cout << "Mouse sensitivity scale: " << mouse_sensitivity_scale << std::endl;
        std::cout << "Use +/- keys to adjust if movement feels too fast/slow" << std::endl;
        std::cout << "Recommended range: 1.5-6.0 depending on your CS2 sensitivity" << std::endl;
    }
}

void MouseController::setSensitivity(float scale) {
    mouse_sensitivity_scale = scale;
    if (debug_mode) {
        std::cout << "Mouse sensitivity scale updated to: " << scale << std::endl;
    }
}

void MouseController::setInputMethod(InputMethod method) {
    switch (method) {
        case InputMethod::RAW_INPUT_ONLY:
            use_raw_input = true;
            force_absolute_mode = false;
            if (debug_mode) std::cout << "Input method: Raw input only (gameplay mode)" << std::endl;
            break;
        case InputMethod::ABSOLUTE_ONLY:
            use_raw_input = false;
            force_absolute_mode = true;
            if (debug_mode) std::cout << "Input method: Absolute positioning only (menu mode)" << std::endl;
            break;
        case InputMethod::AUTO_DETECT:
            force_absolute_mode = false;
            if (debug_mode) std::cout << "Input method: Auto-detect based on game state" << std::endl;
            break;
        case InputMethod::HYBRID:
            if (debug_mode) std::cout << "Input method: Hybrid (try both methods)" << std::endl;
            break;
    }
}

bool MouseController::moveMouseRawInput(cv::Point2f target) {
    if (!cs2_window) return false;
    
    // Fast path - skip expensive operations in critical path
    POINT current_pos;
    if (!GetCursorPos(&current_pos)) {
        return false;
    }
    
    cv::Point2f current(static_cast<float>(current_pos.x), static_cast<float>(current_pos.y));
    cv::Point2f movement = target - current;
    
    // Apply sensitivity scaling
    movement.x *= mouse_sensitivity_scale;
    movement.y *= mouse_sensitivity_scale;
    
    float distance = static_cast<float>(cv::norm(movement));
    
    if (debug_mode) {
        std::cout << "\n=== RAW INPUT MOVEMENT ===" << std::endl;
        std::cout << "Current: " << current.x << "," << current.y << std::endl;
        std::cout << "Target: " << target.x << "," << target.y << std::endl;
        std::cout << "Raw movement: " << movement.x << "," << movement.y << std::endl;
        std::cout << "Distance: " << distance << " pixels" << std::endl;
    }
    
    if (distance < 2.0f) {
        return true;  // Already close enough
    }
    
    // Optimized movement - fewer steps, no delays for maximum speed
    int steps = std::min(movement_steps, std::max(1, static_cast<int>(distance / 20.0f)));
    cv::Point2f step_movement = movement / static_cast<float>(steps);
    
    for (int i = 0; i < steps; i++) {
        INPUT input = {};
        input.type = INPUT_MOUSE;
        input.mi.dwFlags = MOUSEEVENTF_MOVE;
        input.mi.dx = static_cast<LONG>(step_movement.x);
        input.mi.dy = static_cast<LONG>(step_movement.y);
        input.mi.dwExtraInfo = 0;
        input.mi.mouseData = 0;
        input.mi.time = 0;
        
        UINT result = SendInput(1, &input, sizeof(INPUT));
        
        if (result != 1) {
            if (debug_mode) {
                DWORD error = GetLastError();
                std::cout << "SendInput failed at step " << i << ", error: " << error << std::endl;
            }
            return false;
        }
        
        // Remove sleep for maximum responsiveness
        // Only add minimal delay if explicitly configured
        if (step_delay_ms > 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(step_delay_ms * 100));
        }
    }
    
    if (debug_mode) {
        // Verify final position
        POINT final_pos;
        GetCursorPos(&final_pos);
        float final_distance = static_cast<float>(cv::norm(cv::Point2f(static_cast<float>(final_pos.x), 
                                                   static_cast<float>(final_pos.y)) - target));
        std::cout << "Final position: " << final_pos.x << "," << final_pos.y << std::endl;
        std::cout << "Final distance from target: " << final_distance << " pixels" << std::endl;
        std::cout << "Raw input movement completed" << std::endl;
    }
    
    return true;
}

bool MouseController::moveMouseAbsolute(cv::Point2f target) {
    if (!cs2_window) return false;
    
    // Fast validation - cache screen metrics
    static int screen_width = GetSystemMetrics(SM_CXSCREEN);
    static int screen_height = GetSystemMetrics(SM_CYSCREEN);
    
    if (target.x < 0 || target.y < 0 || target.x > screen_width || target.y > screen_height) {
        if (debug_mode) {
            std::cout << "Invalid target coordinates: " << target.x << "," << target.y << std::endl;
        }
        return false;
    }
    
    POINT current_pos;
    GetCursorPos(&current_pos);
    
    cv::Point2f start(static_cast<float>(current_pos.x), static_cast<float>(current_pos.y));
    cv::Point2f diff = target - start;
    float distance = static_cast<float>(cv::norm(diff));
    
    if (debug_mode) {
        std::cout << "\n=== ABSOLUTE MOVEMENT ===" << std::endl;
        std::cout << "Start: " << start.x << "," << start.y << std::endl;
        std::cout << "Target: " << target.x << "," << target.y << std::endl;
        std::cout << "Distance: " << distance << " pixels" << std::endl;
    }
    
    if (distance < 3.0f) {
        return true;
    }
    
    // Use fewer steps for faster menu interactions
    int actual_steps = std::min(3, std::max(1, static_cast<int>(distance / 30.0f)));
    
    for (int i = 1; i <= actual_steps; i++) {
        cv::Point2f intermediate = start + diff * (static_cast<float>(i) / actual_steps);
        
        bool result = SetCursorPos(static_cast<int>(intermediate.x), static_cast<int>(intermediate.y));
        
        if (!result) {
            if (debug_mode) {
                DWORD error = GetLastError();
                std::cout << "SetCursorPos failed at step " << i << ", error: " << error << std::endl;
            }
            return false;
        }
        
        // Minimal delay only if configured
        if (step_delay_ms > 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(step_delay_ms * 500));
        }
    }
    
    if (debug_mode) {
        GetCursorPos(&current_pos);
        float final_distance = static_cast<float>(cv::norm(cv::Point2f(static_cast<float>(current_pos.x), 
                                                   static_cast<float>(current_pos.y)) - target));
        std::cout << "Final distance from target: " << final_distance << " pixels" << std::endl;
        std::cout << "Absolute movement completed" << std::endl;
    }
    
    return true;
}

bool MouseController::moveMouseHybrid(cv::Point2f target) {
    if (debug_mode) {
        std::cout << "\n=== HYBRID MOVEMENT ATTEMPT ===" << std::endl;
    }
    
    // Try raw input first (for gameplay)
    if (moveMouseRawInput(target)) {
        return true;
    }
    
    if (debug_mode) {
        std::cout << "Raw input failed, trying absolute positioning..." << std::endl;
    }
    
    // Fallback to absolute positioning (for menus)
    return moveMouseAbsolute(target);
}

void MouseController::smoothMoveTo(cv::Point2f target) {
    if (!is_active || !cs2_window) {
        return;
    }
    
    // Fast window validation - only check periodically
    auto now = std::chrono::steady_clock::now();
    if (!window_validated || 
        std::chrono::duration_cast<std::chrono::milliseconds>(now - last_window_check).count() > 1000) {
        
        if (!IsWindowVisible(cs2_window) || IsIconic(cs2_window)) {
            if (debug_mode) {
                std::cout << "CS2 window not visible or minimized" << std::endl;
            }
            window_validated = false;
            return;
        }
        window_validated = true;
        last_window_check = now;
    }
    
    // Get current mouse position
    current_position_ = getCurrentMousePosition();
    
    // Check if this is a new target (different from last target position)
    float target_distance_threshold = 50.0f; // pixels
    bool is_new_target = (cv::norm(target - last_target_position_) > target_distance_threshold);
    
    if (is_new_target) {
        // Reset for new target
        direct_move_done_ = false;
        last_target_position_ = target;
        pid_controller_.reset();
        pid_initialized_ = false;
        
        if (debug_mode) {
            std::cout << "New target detected, will do direct move first" << std::endl;
        }
    }
    
    // Phase 1: Direct movement (only once per target)
    if (!direct_move_done_) {
        cv::Point2f error = target - current_position_;
        float error_magnitude = cv::norm(error);
        
        if (error_magnitude > 5.0f) { // Only do direct move if we're far enough
            // Calculate direct movement - move a percentage of the way toward target
            float direct_move_factor = 0.6f; // Move 60% of the way toward target
            cv::Point2f direct_target = current_position_ + error * direct_move_factor;
            
            if (debug_mode) {
                std::cout << "Direct move: " << error_magnitude << " pixels to target" << std::endl;
            }
            
            // Choose movement method based on game state and settings
            bool success = false;
            
            if (force_absolute_mode) {
                success = moveMouseAbsolute(direct_target);
            } else if (use_raw_input && isInGameplay()) {
                success = moveMouseRawInput(direct_target);
            } else if (!isInGameplay()) {
                success = moveMouseAbsolute(direct_target);
            } else {
                success = moveMouseHybrid(direct_target);
            }
            
            if (!success && debug_mode) {
                std::cout << "Direct mouse movement failed" << std::endl;
            }
        }
        
        direct_move_done_ = true;
        if (debug_mode) {
            std::cout << "Direct move completed, switching to PID control" << std::endl;
        }
        return; // Exit after direct move
    }
    
    // Phase 2: PID control for fine adjustment
    cv::Point2f error = target - current_position_;
    float error_magnitude = cv::norm(error);
    
    // Check for auto-fire when we're close to target
    checkAndFire(target);
    
    // Skip if already close enough (reduces oscillation)
    if (error_magnitude < 2.0f) {
        return;
    }
    
    // Calculate time delta for PID
    auto current_time = std::chrono::high_resolution_clock::now();
    float dt = 0.016f; // Default to ~60fps if not initialized
    
    if (pid_initialized_) {
        auto time_diff = std::chrono::duration_cast<std::chrono::microseconds>(current_time - last_update_time_);
        dt = time_diff.count() / 1000000.0f; // Convert to seconds
        dt = std::max(0.001f, std::min(dt, 0.1f)); // Clamp dt to reasonable range
    } else {
        pid_initialized_ = true;
        pid_controller_.reset();
    }
    last_update_time_ = current_time;
    
    // Get PID output
    cv::Point2f pid_output = pid_controller_.update(error, dt);
    
    // Apply sensitivity scaling to PID output
    pid_output *= mouse_sensitivity_scale;
    
    // Limit maximum movement per frame to prevent jumps
    float max_movement = 30.0f; // Reduced for finer control during PID phase
    if (cv::norm(pid_output) > max_movement) {
        pid_output = pid_output * (max_movement / cv::norm(pid_output));
    }
    
    // Calculate new target position based on PID output
    cv::Point2f new_target = current_position_ + pid_output;
    
    if (debug_mode) {
        std::cout << "PID Control - Error: " << error_magnitude 
                  << ", Output: " << cv::norm(pid_output) << std::endl;
    }
    
    // Choose movement method based on game state and settings
    bool success = false;
    
    if (force_absolute_mode) {
        success = moveMouseAbsolute(new_target);
    } else if (use_raw_input && isInGameplay()) {
        success = moveMouseRawInput(new_target);
    } else if (!isInGameplay()) {
        success = moveMouseAbsolute(new_target);
    } else {
        success = moveMouseHybrid(new_target);
    }
    
    if (!success && debug_mode) {
        std::cout << "PID mouse movement failed with current method" << std::endl;
    }
    
    // Store last target for relative movement calculations
    last_target = target;
}

void MouseController::adjustPIDGains(char adjustment) {
    float kp, ki, kd;
    pid_controller_.getGains(kp, ki, kd);
    
    float step = 0.05f;
    
    switch (adjustment) {
        case 't': kp += step * 0.2f; break;
        case 'y': kp = std::max(0.0f, kp - step * 0.2f); break;
        case 'u': ki += step * 0.2f; break; // Smaller steps for integral
        case 'i': ki = std::max(0.0f, ki - step * 0.2f); break;
        case 'o': kd += step * 0.2f; break;
        case 'l': kd = std::max(0.0f, kd - step * 0.2f); break;
    }
    
    pid_controller_.setGains(kp, ki, kd);
    printPIDStatus();
}

void MouseController::resetPID() {
    pid_controller_.reset();
    pid_initialized_ = false;
    direct_move_done_ = false; // Reset direct move flag too
    last_target_position_ = cv::Point2f(-1, -1);
    std::cout << "PID controller and direct move state reset" << std::endl;
}

void MouseController::printPIDStatus() {
    float kp, ki, kd;
    pid_controller_.getGains(kp, ki, kd);
    std::cout << "PID Gains - P: " << std::fixed << std::setprecision(3) << kp 
              << ", I: " << ki << ", D: " << kd << std::endl;
}

void MouseController::testInputMethods() {
    std::cout << "\n=== INPUT METHOD TESTING ===" << std::endl;
    std::cout << "This will test different mouse input methods." << std::endl;
    std::cout << "Make sure CS2 is in the foreground and start the test in 5 seconds..." << std::endl;
    
    std::this_thread::sleep_for(std::chrono::seconds(5));
    
    POINT original;
    GetCursorPos(&original);
    cv::Point2f original_pos(static_cast<float>(original.x), static_cast<float>(original.y));
    
    // Test positions around the original cursor position
    std::vector<cv::Point2f> test_positions = {
        cv::Point2f(original_pos.x + 100, original_pos.y),
        cv::Point2f(original_pos.x + 100, original_pos.y + 100),
        cv::Point2f(original_pos.x, original_pos.y + 100),
        cv::Point2f(original_pos.x - 100, original_pos.y + 100),
        cv::Point2f(original_pos.x - 100, original_pos.y),
        cv::Point2f(original_pos.x - 100, original_pos.y - 100),
        cv::Point2f(original_pos.x, original_pos.y - 100),
        cv::Point2f(original_pos.x + 100, original_pos.y - 100),
        original_pos  // Return to start
    };
    
    // Test raw input method
    std::cout << "\n--- Testing RAW INPUT method ---" << std::endl;
    for (size_t i = 0; i < test_positions.size(); i++) {
        std::cout << "Moving to position " << (i + 1) << "..." << std::endl;
        moveMouseRawInput(test_positions[i]);
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    // Test absolute method
    std::cout << "\n--- Testing ABSOLUTE POSITIONING method ---" << std::endl;
    for (size_t i = 0; i < test_positions.size(); i++) {
        std::cout << "Moving to position " << (i + 1) << "..." << std::endl;
        moveMouseAbsolute(test_positions[i]);
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    
    std::cout << "\nInput method testing complete!" << std::endl;
    std::cout << "Observe which method worked better in CS2:" << std::endl;
    std::cout << "- If the second test (absolute) worked better, you're likely in a menu" << std::endl;
    std::cout << "- If the first test (raw input) worked better, you're likely in gameplay" << std::endl;
    std::cout << "- If neither worked well, there might be a permission or anti-cheat issue" << std::endl;
}

bool MouseController::findCS2Window() {
    if (debug_mode) {
        std::cout << "\n=== ENHANCED CS2 WINDOW DETECTION ===" << std::endl;
    }
    
    // Reset previous state
    cs2_window = nullptr;
    cs2_process_id = 0;
    cs2_window_title.clear();
    window_validated = false;
    
    struct WindowSearchData {
        HWND best_window;
        DWORD best_process_id;
        std::string best_title;
        std::vector<std::tuple<HWND, std::string, DWORD, RECT>> candidates;
        HWND detection_window_handle;
    } search_data = { nullptr, 0, "", {}, detection_window };
    
    // Enhanced window enumeration
    EnumWindows([](HWND hwnd, LPARAM lParam) -> BOOL {
        auto* data = reinterpret_cast<WindowSearchData*>(lParam);
        
        // Skip our own detection window
        if (hwnd == data->detection_window_handle) {
            return TRUE;
        }
        
        if (IsWindowVisible(hwnd) && !IsIconic(hwnd)) {
            char title[256] = {0};
            char class_name[256] = {0};
            GetWindowTextA(hwnd, title, sizeof(title));
            GetClassNameA(hwnd, class_name, sizeof(class_name));
            
            DWORD process_id = 0;
            GetWindowThreadProcessId(hwnd, &process_id);
            
            RECT rect;
            GetWindowRect(hwnd, &rect);
            int width = rect.right - rect.left;
            int height = rect.bottom - rect.top;
            
            std::string window_title(title);
            std::string window_class(class_name);
            
            // Minimum size filter for game windows
            if (width > 800 && height > 600) {
                data->candidates.emplace_back(hwnd, window_title, process_id, rect);
                
                // Score-based CS2 detection
                int score = 0;
                
                // Title-based scoring (highest priority)
                if (window_title.find("Counter-Strike 2") != std::string::npos) score += 100;
                else if (window_title.find("Counter-Strike") != std::string::npos) score += 80;
                else if (window_title == "cs2") score += 70;
                else if (window_title.find("cs2") != std::string::npos) score += 50;
                
                // Class name scoring
                if (window_class.find("Valve") != std::string::npos) score += 30;
                if (window_class.find("SDL") != std::string::npos) score += 20;
                
                // Size scoring (common CS2 resolutions)
                if ((width == 1920 && height == 1080) || 
                    (width == 1280 && height == 720) ||
                    (width == 1440 && height == 900) ||
                    (width == 1600 && height == 900)) score += 10;
                
                // Prefer larger windows (likely fullscreen)
                if (width >= 1920 && height >= 1080) score += 15;
                
                if (score > 50) { // Threshold for CS2 candidates
                    if (data->best_window == nullptr || score > 80) {
                        data->best_window = hwnd;
                        data->best_process_id = process_id;
                        data->best_title = window_title;
                    }
                }
            }
        }
        return TRUE;
    }, reinterpret_cast<LPARAM>(&search_data));
    
    if (!search_data.best_window) {
        if (debug_mode) {
            std::cout << "No CS2 window found automatically!" << std::endl;
        }
        return false;
    }
    
    cs2_window = search_data.best_window;
    cs2_process_id = search_data.best_process_id;
    cs2_window_title = search_data.best_title;
    
    // Validate and get detailed info
    if (!validateCS2Window()) {
        cs2_window = nullptr;
        return false;
    }
    
    if (debug_mode) {
        std::cout << "CS2 Window Selected:" << std::endl;
        std::cout << "  Title: '" << cs2_window_title << "'" << std::endl;
        std::cout << "  HWND: " << cs2_window << std::endl;
        std::cout << "  Process ID: " << cs2_process_id << std::endl;
        std::cout << "  Gameplay detected: " << (isInGameplay() ? "Yes" : "No") << std::endl;
    }
    
    updateCaptureRegion();
    window_validated = true;
    return true;
}

bool MouseController::validateCS2Window() {
    if (!cs2_window || !IsWindow(cs2_window)) {
        return false;
    }
    
    char current_title[256];
    if (!GetWindowTextA(cs2_window, current_title, sizeof(current_title))) {
        return false;
    }
    
    cs2_window_title = current_title;
    
    RECT window_rect, client_rect;
    if (!GetWindowRect(cs2_window, &window_rect) || 
        !GetClientRect(cs2_window, &client_rect)) {
        return false;
    }
    
    int client_width = client_rect.right - client_rect.left;
    int client_height = client_rect.bottom - client_rect.top;
    
    return client_width > 400 && client_height > 300;
}

void MouseController::updateCaptureRegion() {
    if (!cs2_window) return;
    
    RECT client_rect;
    if (GetClientRect(cs2_window, &client_rect)) {
        capture_region = cv::Rect(0, 0, 
                                static_cast<int>(client_rect.right - client_rect.left),
                                static_cast<int>(client_rect.bottom - client_rect.top));
    }
}

cv::Point2f MouseController::detectionToScreen(const cs2_detection::Detection& detection) {
    if (!cs2_window) {
        if (debug_mode) {
            std::cout << "ERROR: No CS2 window found!" << std::endl;
        }
        return cv::Point2f(0, 0);
    }
    
    // Fast path - skip expensive validation in critical path
    if (!window_validated) {
        if (!validateCS2Window()) {
            if (debug_mode) {
                std::cout << "ERROR: CS2 window became invalid, attempting to find again..." << std::endl;
            }
            if (!findCS2Window()) {
                return cv::Point2f(0, 0);
            }
        }
        window_validated = true;
    }
    
    RECT window_rect, client_rect;
    if (!GetWindowRect(cs2_window, &window_rect) || 
        !GetClientRect(cs2_window, &client_rect)) {
        if (debug_mode) {
            std::cout << "ERROR: Failed to get window rectangles!" << std::endl;
        }
        window_validated = false;
        return cv::Point2f(0, 0);
    }
    
    float det_center_x = detection.bbox.x + detection.bbox.width * (0.5f + target_offset_x);
    float det_center_y = detection.bbox.y + detection.bbox.height * (0.5f + target_offset_y);
    
    POINT client_point = { static_cast<LONG>(det_center_x), static_cast<LONG>(det_center_y) };
    
    if (!ClientToScreen(cs2_window, &client_point)) {
        if (debug_mode) {
            std::cout << "ERROR: ClientToScreen conversion failed!" << std::endl;
        }
        window_validated = false;
        return cv::Point2f(0, 0);
    }
    
    return cv::Point2f(static_cast<float>(client_point.x), static_cast<float>(client_point.y));
}

bool MouseController::recheckCS2Window() {
    if (!cs2_window) {
        return findCS2Window();
    }
    
    if (!IsWindow(cs2_window)) {
        if (debug_mode) {
            std::cout << "CS2 window handle became invalid, searching again..." << std::endl;
        }
        window_validated = false;
        return findCS2Window();
    }
    
    char current_title[256];
    if (GetWindowTextA(cs2_window, current_title, sizeof(current_title))) {
        std::string new_title(current_title);
        if (new_title != cs2_window_title) {
            if (debug_mode) {
                std::cout << "CS2 window title changed: '" << cs2_window_title 
                          << "' -> '" << new_title << "'" << std::endl;
            }
            cs2_window_title = new_title;
            updateCaptureRegion();
        }
    }
    
    return validateCS2Window();
}

void MouseController::setActive(bool active) {
    is_active = active;
    std::cout << "\n=== MOUSE CONTROL " << (active ? "ACTIVATED" : "DEACTIVATED") << " ===" << std::endl;
    if (active) {
        findCS2Window();
        resetPID(); // Reset PID when activating
    }
}

cs2_detection::Detection MouseController::selectTarget(const std::vector<cs2_detection::Detection>& detections, 
                                    TargetPriority priority) {
    // Filter enemy targets first
    std::vector<cs2_detection::Detection> enemy_targets = filterEnemyTargets(detections);
    
    if (enemy_targets.empty()) {
        cs2_detection::Detection empty_target;
        empty_target.confidence = 0.0f;
        return empty_target;
    }
    
    cv::Point2f screen_center(static_cast<float>(capture_region.width) / 2.0f, 
                            static_cast<float>(capture_region.height) / 2.0f);
    
    // Find the best target based on priority calculation
    float best_priority = -1.0f;
    size_t best_idx = 0;
    
    for (size_t i = 0; i < enemy_targets.size(); i++) {
        float target_priority = calculateTargetPriority(enemy_targets[i], screen_center);
        
        // Apply target switching threshold only if we have a current target
        bool should_consider = true;
        if (current_target_.confidence > 0.0f) {
            float current_priority = calculateTargetPriority(current_target_, screen_center);
            // Only consider switching if new target is significantly better
            should_consider = (target_priority > current_priority + target_switch_threshold_);
        }
        
        if (should_consider && target_priority > best_priority) {
            best_priority = target_priority;
            best_idx = i;
        }
    }
    
    // If we found a valid target, update current target
    if (best_priority > -1.0f) {
        current_target_ = enemy_targets[best_idx];
        
        if (debug_mode) {
            std::string target_type = hasHelmet(current_target_) ? " (HELMET)" : "";
            std::cout << "Selected target: " << current_target_.class_name << target_type 
                      << " (conf: " << current_target_.confidence << ", priority: " << best_priority << ")" << std::endl;
        }
        
        return current_target_;
    }
    
    // If no new target found but we have a current target, keep using it
    if (current_target_.confidence > 0.0f) {
        // Check if current target is still in the detection list
        for (const auto& detection : enemy_targets) {
            cv::Point2f current_center(current_target_.bbox.x + current_target_.bbox.width / 2.0f,
                                      current_target_.bbox.y + current_target_.bbox.height / 2.0f);
            cv::Point2f det_center(detection.bbox.x + detection.bbox.width / 2.0f,
                                  detection.bbox.y + detection.bbox.height / 2.0f);
            float distance = cv::norm(current_center - det_center);
            
            // If we find the same target, update it with fresh data
            if (distance < 100.0f && detection.class_id == current_target_.class_id) {
                current_target_ = detection;
                return current_target_;
            }
        }
        
        // Current target not found in new detections, clear it
        current_target_.confidence = 0.0f;
    }
    
    // Fallback: if no smart target found, use original simple logic (closest to center)
    float min_distance = std::numeric_limits<float>::max();
    best_idx = 0;
    
    for (size_t i = 0; i < enemy_targets.size(); i++) {
        cv::Point2f det_center(enemy_targets[i].bbox.x + enemy_targets[i].bbox.width / 2.0f,
                             enemy_targets[i].bbox.y + enemy_targets[i].bbox.height / 2.0f);
        float distance = static_cast<float>(cv::norm(det_center - screen_center));
        
        if (distance < min_distance) {
            min_distance = distance;
            best_idx = i;
        }
    }
    
    current_target_ = enemy_targets[best_idx];
    return current_target_;
}

void MouseController::aimAtTarget(const std::vector<cs2_detection::Detection>& detections, 
                TargetPriority priority) {
    if (!is_active || detections.empty() || !cs2_window) {
        return;
    }
    
    auto target = selectTarget(detections, priority);
    if (target.confidence > 0) {
        cv::Point2f screen_target = detectionToScreen(target);
        if (screen_target.x > 0 && screen_target.y > 0) {
            smoothMoveTo(screen_target);
        }
    }
}

void MouseController::setSmoothness(int steps, int delay_ms) {
    movement_steps = std::max(1, std::min(steps, 5));  // Cap at 5 steps for speed
    step_delay_ms = std::max(0, std::min(delay_ms, 2)); // Cap at 2ms for responsiveness
    if (debug_mode) {
        std::cout << "Mouse smoothness: " << movement_steps << " steps, " << step_delay_ms << "ms delay" << std::endl;
    }
}

void MouseController::setTargetOffset(float offset_x, float offset_y) {
    target_offset_x = offset_x;
    target_offset_y = offset_y;
    if (debug_mode) {
        std::cout << "Target offset: " << offset_x << "," << offset_y << std::endl;
    }
}

void MouseController::toggleDebugMode() {
    debug_mode = !debug_mode;
    std::cout << "Debug mode: " << (debug_mode ? "ON" : "OFF") << std::endl;
}

void MouseController::manualWindowSelection() {
    std::cout << "Manual window selection not implemented in this version" << std::endl;
}

void MouseController::testBasicMovement() {
    std::cout << "Test basic movement not implemented in this version" << std::endl;
}

void MouseController::testScreenCenter() {
    std::cout << "Test screen center not implemented in this version" << std::endl;
}

// CS2MouseIntegration implementation
CS2MouseIntegration::CS2MouseIntegration() : mouse_enabled(false) {
}

void CS2MouseIntegration::toggleMouseControl() {
    mouse_enabled = !mouse_enabled;
    mouse_controller.setActive(mouse_enabled);
}

void CS2MouseIntegration::processDetections(const std::vector<cs2_detection::Detection>& detections) {
    if (mouse_enabled && !detections.empty()) {
        mouse_controller.aimAtTarget(detections, MouseController::TargetPriority::CLOSEST_TO_CENTER);
    }
    previous_detections = detections;
}

void CS2MouseIntegration::setupMouseControl() {
    // Optimized settings for maximum responsiveness
    mouse_controller.setSmoothness(2, 0);  // Minimal steps, no delay
    mouse_controller.setTargetOffset(0.0f, -0.2f);
    mouse_controller.setInputMethod(MouseController::InputMethod::AUTO_DETECT);
    
    std::cout << "\n=== ENHANCED MOUSE CONTROL WITH PID ===" << std::endl;
    std::cout << "Features:" << std::endl;
    std::cout << "  PID controller for smooth, stable aiming" << std::endl;
    std::cout << "  Reduced oscillation and overshoot" << std::endl;
    std::cout << "  Raw input support for CS2 gameplay" << std::endl;
    std::cout << "  Absolute positioning for menus" << std::endl;
    std::cout << "  Auto-detection of game state" << std::endl;
    std::cout << "  Team-based targeting system" << std::endl;
    std::cout << "  Helmet target prioritization" << std::endl;
    std::cout << "\nControls:" << std::endl;
    std::cout << "  M = Toggle mouse control on/off" << std::endl;
    std::cout << "  V = Toggle debug mode" << std::endl;
    std::cout << "  R = Reset PID controller" << std::endl;
    std::cout << "  t/y = Increase/Decrease PID P-gain" << std::endl;
    std::cout << "  u/i = Increase/Decrease PID I-gain" << std::endl;
    std::cout << "  o/l = Increase/Decrease PID D-gain" << std::endl;
    std::cout << "  + = Increase sensitivity (currently 3.0x)" << std::endl;
    std::cout << "  - = Decrease sensitivity" << std::endl;
    std::cout << "  4/5/6 = Fast/Medium/Slow presets" << std::endl;
    std::cout << "  Z = Set team to CT" << std::endl;
    std::cout << "  X = Set team to T" << std::endl;
    std::cout << "  B = Toggle team" << std::endl;
    std::cout << "  F = Toggle auto-fire on/off" << std::endl;
    std::cout << "\nPID Tips:" << std::endl;
    std::cout << "  P-gain: Higher = faster response, too high = oscillation" << std::endl;
    std::cout << "  I-gain: Eliminates steady-state error, too high = instability" << std::endl;
    std::cout << "  D-gain: Reduces overshoot, too high = noise sensitivity" << std::endl;
    std::cout << "\nTeam Targeting:" << std::endl;
    std::cout << "  Set your team first (Z for CT, X for T)" << std::endl;
    std::cout << "  Only enemy targets will be aimed at" << std::endl;
    std::cout << "  Helmet targets get priority boost" << std::endl;
    std::cout << "  5% threshold prevents target switching spam" << std::endl;
    std::cout << "\nAuto-Fire:" << std::endl;
    std::cout << "  Press F to enable/disable auto-fire" << std::endl;
    std::cout << "  Fires when within 15 pixels of target" << std::endl;
    std::cout << "  0.8 second cooldown between shots" << std::endl;
    
    mouse_controller.printPIDStatus();
}

void CS2MouseIntegration::setDetectionWindowHandle(HWND window_handle) {
    mouse_controller.setDetectionWindow(window_handle);
}

bool CS2MouseIntegration::recheckCS2Window() {
    return mouse_controller.recheckCS2Window();
}

void CS2MouseIntegration::refreshWindowDetection() {
    std::cout << "\n=== REFRESHING WINDOW DETECTION ===" << std::endl;
    mouse_controller.findCS2Window();
}

void CS2MouseIntegration::manualWindowSelection() {
    mouse_controller.manualWindowSelection();
}

void CS2MouseIntegration::testInputMethods() {
    mouse_controller.testInputMethods();
}

void CS2MouseIntegration::setInputMode(int mode) {
    switch (mode) {
        case 1:
            mouse_controller.setInputMethod(MouseController::InputMethod::RAW_INPUT_ONLY);
            break;
        case 2:
            mouse_controller.setInputMethod(MouseController::InputMethod::ABSOLUTE_ONLY);
            break;
        case 3:
            mouse_controller.setInputMethod(MouseController::InputMethod::AUTO_DETECT);
            break;
        case 4:
            setSensitivityPreset(6.0f);
            break;
        case 5:
            setSensitivityPreset(3.0f);
            break;
        case 6:
            setSensitivityPreset(1.5f);
            break;
    }
}

void CS2MouseIntegration::adjustSensitivity(bool increase) {
    static float current_sensitivity = 3.0f;  // Start with the new default
    if (increase) {
        current_sensitivity += 0.1f;  // Larger increments for faster adjustment
        current_sensitivity = std::min(10.0f, current_sensitivity);  // Cap at 10x
    } else {
        current_sensitivity -= 0.1f;
        current_sensitivity = std::max(0.2f, current_sensitivity);  // Min 0.2x
    }
    mouse_controller.setSensitivity(current_sensitivity);
    std::cout << "Sensitivity: " << current_sensitivity << "x" << std::endl;
}

void CS2MouseIntegration::setSensitivityPreset(float sensitivity) {
    mouse_controller.setSensitivity(sensitivity);
    std::cout << "Sensitivity preset: " << sensitivity << "x" << std::endl;
}

void CS2MouseIntegration::toggleDebugMode() {
    mouse_controller.toggleDebugMode();
}

void CS2MouseIntegration::adjustPIDGains(char adjustment) {
    mouse_controller.adjustPIDGains(adjustment);
}

void CS2MouseIntegration::resetPID() {
    mouse_controller.resetPID();
}

void CS2MouseIntegration::setPlayerTeam(Team team) {
    mouse_controller.setPlayerTeam(team);
}

void MouseController::setAutoFire(bool enabled) {
    auto_fire_enabled_ = enabled;
    std::cout << "Auto-fire: " << (enabled ? "ENABLED" : "DISABLED") << std::endl;
    if (enabled) {
        std::cout << "Will fire when within " << fire_distance_threshold_ << " pixels of target" << std::endl;
        std::cout << "Fire cooldown: " << fire_cooldown_ms_ << "ms" << std::endl;
    }
}

void MouseController::toggleAutoFire() {
    setAutoFire(!auto_fire_enabled_);
}

void MouseController::performSingleShot() {
    if (!cs2_window) return;
    
    // Send mouse left click down
    INPUT input_down = {};
    input_down.type = INPUT_MOUSE;
    input_down.mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
    input_down.mi.dx = 0;
    input_down.mi.dy = 0;
    input_down.mi.dwExtraInfo = 0;
    input_down.mi.mouseData = 0;
    input_down.mi.time = 0;
    
    // Send mouse left click up
    INPUT input_up = {};
    input_up.type = INPUT_MOUSE;
    input_up.mi.dwFlags = MOUSEEVENTF_LEFTUP;
    input_up.mi.dx = 0;
    input_up.mi.dy = 0;
    input_up.mi.dwExtraInfo = 0;
    input_up.mi.mouseData = 0;
    input_up.mi.time = 0;
    
    // Send both inputs as a single shot
    INPUT inputs[2] = {input_down, input_up};
    UINT result = SendInput(2, inputs, sizeof(INPUT));
    
    if (debug_mode) {
        if (result == 2) {
            std::cout << "Auto-fire: Shot fired!" << std::endl;
        } else {
            std::cout << "Auto-fire: Failed to send click input" << std::endl;
        }
    }
    
    // Update last fire time
    last_fire_time_ = std::chrono::high_resolution_clock::now();
}

void MouseController::checkAndFire(cv::Point2f target_position) {
    if (!auto_fire_enabled_ || !cs2_window) {
        return;
    }
    
    // Check if we're close enough to target
    cv::Point2f current_pos = getCurrentMousePosition();
    float distance_to_target = cv::norm(target_position - current_pos);
    
    if (distance_to_target > fire_distance_threshold_) {
        return; // Too far from target
    }
    
    // Check cooldown
    auto now = std::chrono::high_resolution_clock::now();
    auto time_since_last_fire = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_fire_time_);
    
    if (time_since_last_fire.count() < fire_cooldown_ms_) {
        return; // Still in cooldown
    }
    
    // Fire!
    performSingleShot();
    
    if (debug_mode) {
        std::cout << "Auto-fire triggered at distance: " << distance_to_target << " pixels" << std::endl;
    }
}

void CS2MouseIntegration::toggleTeam() {
    mouse_controller.toggleTeam();
}

void CS2MouseIntegration::toggleAutoFire() {
    mouse_controller.toggleAutoFire();
}

} // namespace cs2_control

#endif // _WIN32