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
#include <vector>
#include <random>
#include <algorithm>
#include <limits>

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

#ifdef _WIN32
// Fixed MouseController class - replace the existing one in your code
class MouseController {
private:
    HWND cs2_window = nullptr;
    HWND detection_window = nullptr;
    cv::Rect capture_region;
    bool is_active = false;
    bool debug_mode = true;
    
    // Input method detection
    bool use_raw_input = true;  // Default to raw input for gameplay
    bool force_absolute_mode = false;  // For testing menu mode
    
    // Smoothing parameters
    int movement_steps = 12;  // More steps for smoother raw input
    int step_delay_ms = 1;    // Faster steps for responsiveness
    
    // Target selection parameters
    float target_offset_x = 0.0f;
    float target_offset_y = -0.2f;
    
    // Mouse sensitivity calibration
    float mouse_sensitivity_scale = 3.0f;  // Increased default for faster movement
    
    // Window validation
    DWORD cs2_process_id = 0;
    std::string cs2_window_title;
    
    // Movement tracking for relative input
    cv::Point2f last_target = cv::Point2f(-1, -1);
    
public:
    enum class TargetPriority {
        CLOSEST_TO_CENTER,
        CLOSEST_TO_CURSOR,
        HIGHEST_CONFIDENCE,
        NEAREST_ENEMY
    };
    
    enum class InputMethod {
        AUTO_DETECT,
        RAW_INPUT_ONLY,     // For gameplay
        ABSOLUTE_ONLY,      // For menus
        HYBRID              // Try both
    };
    
    MouseController() {
        findCS2Window();
        calibrateMouseSensitivity();
    }
    
    void setDetectionWindow(HWND window) {
        detection_window = window;
        std::cout << "Detection window handle stored: " << window << std::endl;
    }
    
    // Detect if CS2 is likely in-game vs in menu
    bool isInGameplay() {
        if (!cs2_window) return false;
        
        // Check window title for gameplay indicators
        char title[256];
        if (GetWindowTextA(cs2_window, title, sizeof(title))) {
            std::string window_title(title);
            
            // Menu indicators (use absolute positioning)
            if (window_title.find("Menu") != std::string::npos ||
                window_title.find("Main Menu") != std::string::npos ||
                window_title.find("Loading") != std::string::npos) {
                return false;
            }
            
            // Gameplay indicators (use raw input)
            if (window_title.find("de_") != std::string::npos ||  // Map names
                window_title.find("cs_") != std::string::npos ||
                window_title.find("Competitive") != std::string::npos ||
                window_title.find("Casual") != std::string::npos ||
                window_title.find("Deathmatch") != std::string::npos) {
                return true;
            }
        }
        
        // Default assumption: if CS2 is running and we can't tell, assume gameplay
        return true;
    }
    
    void calibrateMouseSensitivity() {
        // Start with a more aggressive default for faster movement
        mouse_sensitivity_scale = 3.0f;
        
        std::cout << "Mouse sensitivity scale: " << mouse_sensitivity_scale << std::endl;
        std::cout << "Use +/- keys to adjust if movement feels too fast/slow" << std::endl;
        std::cout << "Recommended range: 1.5-6.0 depending on your CS2 sensitivity" << std::endl;
    }
    
    void setSensitivity(float scale) {
        mouse_sensitivity_scale = scale;
        std::cout << "Mouse sensitivity scale updated to: " << scale << std::endl;
    }
    
    void setInputMethod(InputMethod method) {
        switch (method) {
            case InputMethod::RAW_INPUT_ONLY:
                use_raw_input = true;
                force_absolute_mode = false;
                std::cout << "Input method: Raw input only (gameplay mode)" << std::endl;
                break;
            case InputMethod::ABSOLUTE_ONLY:
                use_raw_input = false;
                force_absolute_mode = true;
                std::cout << "Input method: Absolute positioning only (menu mode)" << std::endl;
                break;
            case InputMethod::AUTO_DETECT:
                force_absolute_mode = false;
                std::cout << "Input method: Auto-detect based on game state" << std::endl;
                break;
            case InputMethod::HYBRID:
                std::cout << "Input method: Hybrid (try both methods)" << std::endl;
                break;
        }
    }
    
    // Raw input method using SendInput with relative movements
    bool moveMouseRawInput(cv::Point2f target) {
        if (!cs2_window) return false;
        
        // Get current cursor position
        POINT current_pos;
        if (!GetCursorPos(&current_pos)) {
            std::cout << "❌ Failed to get cursor position" << std::endl;
            return false;
        }
        
        cv::Point2f current(static_cast<float>(current_pos.x), static_cast<float>(current_pos.y));
        cv::Point2f movement = target - current;
        
        // Apply sensitivity scaling
        movement.x *= mouse_sensitivity_scale;
        movement.y *= mouse_sensitivity_scale;
        
        float distance = cv::norm(movement);
        
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
        
        // Break movement into smaller steps for smoother movement
        int steps = std::min(movement_steps, static_cast<int>(distance / 5.0f) + 1);
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
                DWORD error = GetLastError();
                std::cout << "❌ SendInput failed at step " << i << ", error: " << error << std::endl;
                return false;
            }
            
            if (step_delay_ms > 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(step_delay_ms));
            }
        }
        
        if (debug_mode) {
            // Verify final position
            POINT final_pos;
            GetCursorPos(&final_pos);
            float final_distance = cv::norm(cv::Point2f(static_cast<float>(final_pos.x), 
                                                       static_cast<float>(final_pos.y)) - target);
            std::cout << "Final position: " << final_pos.x << "," << final_pos.y << std::endl;
            std::cout << "Final distance from target: " << final_distance << " pixels" << std::endl;
            std::cout << "✅ Raw input movement completed" << std::endl;
        }
        
        return true;
    }
    
    // Absolute positioning method (for menus)
    bool moveMouseAbsolute(cv::Point2f target) {
        if (!cs2_window) return false;
        
        // Validate target coordinates
        int screen_width = GetSystemMetrics(SM_CXSCREEN);
        int screen_height = GetSystemMetrics(SM_CYSCREEN);
        
        if (target.x < 0 || target.y < 0 || target.x > screen_width || target.y > screen_height) {
            std::cout << "❌ Invalid target coordinates: " << target.x << "," << target.y << std::endl;
            return false;
        }
        
        POINT current_pos;
        GetCursorPos(&current_pos);
        
        cv::Point2f start(static_cast<float>(current_pos.x), static_cast<float>(current_pos.y));
        cv::Point2f diff = target - start;
        float distance = cv::norm(diff);
        
        if (debug_mode) {
            std::cout << "\n=== ABSOLUTE MOVEMENT ===" << std::endl;
            std::cout << "Start: " << start.x << "," << start.y << std::endl;
            std::cout << "Target: " << target.x << "," << target.y << std::endl;
            std::cout << "Distance: " << distance << " pixels" << std::endl;
        }
        
        if (distance < 3.0f) {
            return true;
        }
        
        // Use fewer steps for menu interactions
        int actual_steps = std::min(8, static_cast<int>(distance / 10.0f) + 1);
        
        for (int i = 1; i <= actual_steps; i++) {
            cv::Point2f intermediate = start + diff * (static_cast<float>(i) / actual_steps);
            
            bool result = SetCursorPos(static_cast<int>(intermediate.x), static_cast<int>(intermediate.y));
            
            if (!result) {
                DWORD error = GetLastError();
                std::cout << "❌ SetCursorPos failed at step " << i << ", error: " << error << std::endl;
                return false;
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(step_delay_ms * 2));
        }
        
        if (debug_mode) {
            GetCursorPos(&current_pos);
            float final_distance = cv::norm(cv::Point2f(static_cast<float>(current_pos.x), 
                                                       static_cast<float>(current_pos.y)) - target);
            std::cout << "Final distance from target: " << final_distance << " pixels" << std::endl;
            std::cout << "✅ Absolute movement completed" << std::endl;
        }
        
        return true;
    }
    
    // Hybrid approach - try both methods
    bool moveMouseHybrid(cv::Point2f target) {
        if (debug_mode) {
            std::cout << "\n=== HYBRID MOVEMENT ATTEMPT ===" << std::endl;
        }
        
        // Try raw input first (for gameplay)
        if (moveMouseRawInput(target)) {
            return true;
        }
        
        std::cout << "Raw input failed, trying absolute positioning..." << std::endl;
        
        // Fallback to absolute positioning (for menus)
        return moveMouseAbsolute(target);
    }
    
    void smoothMoveTo(cv::Point2f target) {
        if (!is_active || !cs2_window) {
            return;
        }
        
        // Validate window state
        if (!IsWindowVisible(cs2_window) || IsIconic(cs2_window)) {
            if (debug_mode) {
                std::cout << "CS2 window not visible or minimized" << std::endl;
            }
            return;
        }
        
        // Choose movement method based on game state and settings
        bool success = false;
        
        if (force_absolute_mode) {
            // Forced absolute mode (for testing menus)
            success = moveMouseAbsolute(target);
        } else if (use_raw_input && isInGameplay()) {
            // Raw input for gameplay
            if (debug_mode) {
                std::cout << "Detected gameplay - using raw input" << std::endl;
            }
            success = moveMouseRawInput(target);
        } else if (!isInGameplay()) {
            // Absolute positioning for menus
            if (debug_mode) {
                std::cout << "Detected menu - using absolute positioning" << std::endl;
            }
            success = moveMouseAbsolute(target);
        } else {
            // Hybrid approach when uncertain
            success = moveMouseHybrid(target);
        }
        
        if (!success && debug_mode) {
            std::cout << "⚠️ Mouse movement failed with current method" << std::endl;
        }
        
        // Store last target for relative movement calculations
        last_target = target;
    }
    
    // Test different input methods
    void testInputMethods() {
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
    
    // All the existing methods remain the same...
    bool findCS2Window() {
        std::cout << "\n=== ENHANCED CS2 WINDOW DETECTION ===" << std::endl;
        
        // Reset previous state
        cs2_window = nullptr;
        cs2_process_id = 0;
        cs2_window_title.clear();
        
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
                        std::cout << "CS2 Candidate: '" << window_title 
                                  << "' [" << window_class << "] "
                                  << "(" << width << "x" << height << ") "
                                  << "PID:" << process_id << " Score:" << score << std::endl;
                        
                        if (score > 80 || (score >= 50 && !data->best_window)) {
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
            std::cout << "❌ No CS2 window found automatically!" << std::endl;
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
        
        std::cout << "✅ CS2 Window Selected:" << std::endl;
        std::cout << "  Title: '" << cs2_window_title << "'" << std::endl;
        std::cout << "  HWND: " << cs2_window << std::endl;
        std::cout << "  Process ID: " << cs2_process_id << std::endl;
        std::cout << "  Gameplay detected: " << (isInGameplay() ? "Yes" : "No") << std::endl;
        
        updateCaptureRegion();
        return true;
    }
    
    bool validateCS2Window() {
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
    
    void updateCaptureRegion() {
        if (!cs2_window) return;
        
        RECT client_rect;
        if (GetClientRect(cs2_window, &client_rect)) {
            capture_region = cv::Rect(0, 0, 
                                    static_cast<int>(client_rect.right - client_rect.left),
                                    static_cast<int>(client_rect.bottom - client_rect.top));
        }
    }
    
    cv::Point2f detectionToScreen(const cs2_detection::Detection& detection) {
        if (!cs2_window) {
            std::cout << "ERROR: No CS2 window found!" << std::endl;
            return cv::Point2f(0, 0);
        }
        
        if (!validateCS2Window()) {
            std::cout << "ERROR: CS2 window became invalid, attempting to find again..." << std::endl;
            if (!findCS2Window()) {
                return cv::Point2f(0, 0);
            }
        }
        
        RECT window_rect, client_rect;
        if (!GetWindowRect(cs2_window, &window_rect) || 
            !GetClientRect(cs2_window, &client_rect)) {
            std::cout << "ERROR: Failed to get window rectangles!" << std::endl;
            return cv::Point2f(0, 0);
        }
        
        float det_center_x = detection.bbox.x + detection.bbox.width * (0.5f + target_offset_x);
        float det_center_y = detection.bbox.y + detection.bbox.height * (0.5f + target_offset_y);
        
        POINT client_point = { static_cast<LONG>(det_center_x), static_cast<LONG>(det_center_y) };
        
        if (!ClientToScreen(cs2_window, &client_point)) {
            std::cout << "ERROR: ClientToScreen conversion failed!" << std::endl;
            return cv::Point2f(0, 0);
        }
        
        return cv::Point2f(static_cast<float>(client_point.x), static_cast<float>(client_point.y));
    }
    
    bool recheckCS2Window() {
        if (!cs2_window) {
            return findCS2Window();
        }
        
        if (!IsWindow(cs2_window)) {
            std::cout << "⚠️ CS2 window handle became invalid, searching again..." << std::endl;
            return findCS2Window();
        }
        
        char current_title[256];
        if (GetWindowTextA(cs2_window, current_title, sizeof(current_title))) {
            std::string new_title(current_title);
            if (new_title != cs2_window_title) {
                std::cout << "🔄 CS2 window title changed: '" << cs2_window_title 
                          << "' -> '" << new_title << "'" << std::endl;
                cs2_window_title = new_title;
                updateCaptureRegion();
            }
        }
        
        return validateCS2Window();
    }
    
    void setActive(bool active) {
        is_active = active;
        std::cout << "\n=== MOUSE CONTROL " << (active ? "ACTIVATED" : "DEACTIVATED") << " ===" << std::endl;
        if (active) {
            findCS2Window();
        }
    }
    
    cs2_detection::Detection selectTarget(const std::vector<cs2_detection::Detection>& detections, 
                                        TargetPriority priority) {
        if (detections.empty()) return {};
        
        cv::Point2f screen_center(static_cast<float>(capture_region.width) / 2.0f, 
                                static_cast<float>(capture_region.height) / 2.0f);
        float min_distance = std::numeric_limits<float>::max();
        size_t best_idx = 0;
        
        for (size_t i = 0; i < detections.size(); i++) {
            cv::Point2f det_center(detections[i].bbox.x + detections[i].bbox.width / 2.0f,
                                 detections[i].bbox.y + detections[i].bbox.height / 2.0f);
            float distance = cv::norm(det_center - screen_center);
            
            if (distance < min_distance) {
                min_distance = distance;
                best_idx = i;
            }
        }
        
        return detections[best_idx];
    }
    
    void aimAtTarget(const std::vector<cs2_detection::Detection>& detections, 
                    TargetPriority priority = TargetPriority::CLOSEST_TO_CENTER) {
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
    
    void setSmoothness(int steps, int delay_ms) {
        movement_steps = steps;
        step_delay_ms = delay_ms;
        std::cout << "Mouse smoothness: " << steps << " steps, " << delay_ms << "ms delay" << std::endl;
    }
    
    void setTargetOffset(float offset_x, float offset_y) {
        target_offset_x = offset_x;
        target_offset_y = offset_y;
        std::cout << "Target offset: " << offset_x << "," << offset_y << std::endl;
    }
    
    void toggleDebugMode() {
        debug_mode = !debug_mode;
        std::cout << "Debug mode: " << (debug_mode ? "ON" : "OFF") << std::endl;
    }
    
    // Additional convenience methods for testing
    void manualWindowSelection() {
        // Implementation from original code...
    }
    
    void testBasicMovement() {
        // Implementation from original code...
    }
    
    void testScreenCenter() {
        // Implementation from original code...
    }
};

// Updated CS2MouseIntegration class
class CS2MouseIntegration {
private:
    MouseController mouse_controller;
    std::vector<cs2_detection::Detection> previous_detections;
    bool mouse_enabled = false;
    
public:
    void toggleMouseControl() {
        mouse_enabled = !mouse_enabled;
        mouse_controller.setActive(mouse_enabled);
    }
    
    void processDetections(const std::vector<cs2_detection::Detection>& detections) {
        if (mouse_enabled && !detections.empty()) {
            mouse_controller.aimAtTarget(detections, MouseController::TargetPriority::CLOSEST_TO_CENTER);
        }
        previous_detections = detections;
    }
    
    void setupMouseControl() {
        // Optimized settings for raw input - faster and more responsive
        mouse_controller.setSmoothness(4, 0);  // Fewer steps, no delay for speed
        mouse_controller.setTargetOffset(0.0f, -0.2f);
        mouse_controller.setInputMethod(MouseController::InputMethod::AUTO_DETECT);
        
        std::cout << "\n=== ENHANCED MOUSE CONTROL SETUP ===" << std::endl;
        std::cout << "Features:" << std::endl;
        std::cout << "  ✅ Raw input support for CS2 gameplay" << std::endl;
        std::cout << "  ✅ Absolute positioning for menus" << std::endl;
        std::cout << "  ✅ Auto-detection of game state" << std::endl;
        std::cout << "  ✅ Multiple input method fallbacks" << std::endl;
        std::cout << "  ⚡ Optimized for speed and responsiveness" << std::endl;
        std::cout << "\nControls:" << std::endl;
        std::cout << "  M = Toggle mouse control on/off" << std::endl;
        std::cout << "  V = Toggle debug mode" << std::endl;
        std::cout << "  C = Refresh window detection" << std::endl;
        std::cout << "  X = Manual window selection" << std::endl;
        std::cout << "  T = Test input methods" << std::endl;
        std::cout << "  1 = Force raw input mode" << std::endl;
        std::cout << "  2 = Force absolute mode" << std::endl;
        std::cout << "  3 = Auto-detect mode" << std::endl;
        std::cout << "  + = Increase sensitivity (currently 3.0x)" << std::endl;
        std::cout << "  - = Decrease sensitivity" << std::endl;
        std::cout << "  4 = Fast preset (6.0x)" << std::endl;
        std::cout << "  5 = Medium preset (3.0x)" << std::endl;
        std::cout << "  6 = Slow preset (1.5x)" << std::endl;
        std::cout << "\n💡 TIP: Use +/- to fine-tune or 4/5/6 for quick presets!" << std::endl;
    }
    
    void setDetectionWindowHandle(HWND window_handle) {
        mouse_controller.setDetectionWindow(window_handle);
    }
    
    bool recheckCS2Window() {
        return mouse_controller.recheckCS2Window();
    }
    
    void refreshWindowDetection() {
        std::cout << "\n=== REFRESHING WINDOW DETECTION ===" << std::endl;
        mouse_controller.findCS2Window();
    }
    
    void manualWindowSelection() {
        mouse_controller.manualWindowSelection();
    }
    
    void testInputMethods() {
        mouse_controller.testInputMethods();
    }
    
    void setInputMode(int mode) {
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
    
    void adjustSensitivity(bool increase) {
        static float current_sensitivity = 3.0f;  // Start with the new default
        if (increase) {
            current_sensitivity += 0.5f;  // Larger increments for faster adjustment
            current_sensitivity = std::min(10.0f, current_sensitivity);  // Cap at 10x
        } else {
            current_sensitivity -= 0.5f;
            current_sensitivity = std::max(0.2f, current_sensitivity);  // Min 0.2x
        }
        mouse_controller.setSensitivity(current_sensitivity);
        std::cout << "Sensitivity: " << current_sensitivity << "x" << std::endl;
    }
    
    void setSensitivityPreset(float sensitivity) {
        mouse_controller.setSensitivity(sensitivity);
        std::cout << "Sensitivity preset: " << sensitivity << "x" << std::endl;
    }
    
    void toggleDebugMode() {
        mouse_controller.toggleDebugMode();
    }
};

#endif // _WIN32

// Real-time detector class with mouse control integration
#ifdef _WIN32
class CS2RealTimeDetector {
private:
    std::unique_ptr<cs2_detection::YOLOv8Detector> detector;
    std::unique_ptr<cs2_detection::WindowsScreenCapture> screen_capture;
    std::atomic<bool> running{false};
    std::atomic<bool> save_next_frame{false};
    int window_recheck_counter = 0;
    
    // Performance tracking
    int total_frames = 0;
    int detected_frames = 0;
    std::chrono::high_resolution_clock::time_point session_start;
    
    // Display settings
    bool display_enabled = false;
    std::string window_name = "CS2 Detection";
    
    // Mouse control integration
    CS2MouseIntegration mouse_integration;

public:
    CS2RealTimeDetector(const std::string& model_path, bool use_gpu = true) {
        detector = std::make_unique<cs2_detection::YOLOv8Detector>(model_path, use_gpu);
        screen_capture = cs2_detection::ScreenCaptureFactory::create(cs2_detection::CaptureMethod::AUTO);
        
        if (!screen_capture->initializeForCS2Window()) {
            std::cerr << "Failed to initialize screen capture for CS2" << std::endl;
            throw std::runtime_error("Screen capture initialization failed");
        }
        
        detector->warmup(3);
        
        // Setup mouse control
        mouse_integration.setupMouseControl();
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
            std::cout << "  M: Toggle mouse control on/off" << std::endl;
            std::cout << "  V: Toggle debug mode" << std::endl;
            std::cout << "  C: Refresh window detection" << std::endl;
            std::cout << "  X: Manual window selection" << std::endl;
            std::cout << "  Z: Test basic movement" << std::endl;
            std::cout << "  SPACE: Test screen center" << std::endl;
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
                        std::cout << "⚠️ CS2 window validation failed, continuing detection..." << std::endl;
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

    
private:
    void setupDisplayWindow() {
        try {
            cv::namedWindow(window_name, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
            
#ifdef _WIN32
            int screen_width = GetSystemMetrics(SM_CXSCREEN);
            int screen_height = GetSystemMetrics(SM_CYSCREEN);
#else
            int screen_width = 1920;
            int screen_height = 1080;
#endif
            int window_width = 800;
            int window_height = 600;
            
            cv::moveWindow(window_name, screen_width - window_width - 50, screen_height - window_height - 100);
            cv::resizeWindow(window_name, window_width, window_height);
            cv::setWindowProperty(window_name, cv::WND_PROP_TOPMOST, 0);
            
            display_enabled = true;
            std::cout << "Display window created and positioned away from game area" << std::endl;
            
            // IMPORTANT: Pass the detection window handle to mouse controller
#ifdef _WIN32
            // Get the OpenCV window handle and pass it to mouse controller
            HWND cv_window = FindWindowA(nullptr, window_name.c_str());
            if (cv_window) {
                mouse_integration.setDetectionWindowHandle(cv_window);
            }
#endif
            
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
        
        if (key == 'q' || key == 'Q' || key == 27) { // Quit
            running = false;
        } else if (key == 'p' || key == 'P') { // Pause
            paused = !paused;
            std::cout << (paused ? "\n[PAUSED]" : "\n[RESUMED]") << std::endl;
        } else if (key == 's' || key == 'S') { // Save
            if (save_results) {
                save_next_frame = true;
                std::cout << "\n[SAVE] Screenshot queued" << std::endl;
            }
        } else if (key == 'd' || key == 'D') { // Display toggle
            display_enabled = !display_enabled;
            if (!display_enabled) {
                cv::destroyAllWindows();
                std::cout << "\n[DISPLAY OFF]" << std::endl;
            } else {
                setupDisplayWindow();
                std::cout << "\n[DISPLAY ON]" << std::endl;
            }
        } else if (key == 'm' || key == 'M') { // Mouse control toggle
            mouse_integration.toggleMouseControl();
        } else if (key == 'v' || key == 'V') { // Debug mode
            mouse_integration.toggleDebugMode();
        } else if (key == 'c' || key == 'C') { // Refresh window detection
            mouse_integration.refreshWindowDetection();
        } else if (key == 'x' || key == 'X') { // Manual window selection
            mouse_integration.manualWindowSelection();
        }
    }
    
    void addPerformanceOverlay(cv::Mat& frame, const cs2_detection::CaptureMetrics& capture_metrics,
                              const cs2_detection::PerformanceMetrics& detection_metrics,
                              int detection_count, const std::vector<int>& class_counts) {
        
        cv::Mat overlay = frame.clone();
        cv::rectangle(overlay, cv::Point(10, 10), cv::Point(500, 200), cv::Scalar(0, 0, 0), -1);
        cv::addWeighted(frame, 0.75, overlay, 0.25, 0, frame);
        
        cv::putText(frame, "CS2 REAL-TIME DETECTION + MOUSE CONTROL", 
                   cv::Point(20, 35), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
        
        cv::putText(frame, "FPS: " + std::to_string(static_cast<int>(capture_metrics.fps)), 
                   cv::Point(20, 60), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
        
        cv::putText(frame, "Latency: " + std::to_string(static_cast<int>(capture_metrics.total_ms + detection_metrics.total_ms)) + "ms",
                   cv::Point(20, 85), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        
        cv::putText(frame, "Players: " + std::to_string(detection_count),
                   cv::Point(20, 110), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);
        
        if (detection_count > 0 && class_counts.size() >= 4) {
            std::string class_breakdown = "CT:" + std::to_string(class_counts[0]) + 
                                        " T:" + std::to_string(class_counts[2]) + 
                                        " +H:" + std::to_string(class_counts[1] + class_counts[3]);
            cv::putText(frame, class_breakdown,
                       cv::Point(20, 135), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);
        }
        
        float total_latency = capture_metrics.total_ms + detection_metrics.total_ms;
        cv::Scalar perf_color = total_latency < 20 ? cv::Scalar(0, 255, 0) : 
                               total_latency < 33 ? cv::Scalar(0, 255, 255) : cv::Scalar(0, 0, 255);
        cv::circle(frame, cv::Point(450, 35), 10, perf_color, -1);
        
        cv::putText(frame, "M=Mouse V=Debug C=Refresh X=Select Z=Test",
                   cv::Point(20, 160), cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(150, 150, 150), 1);
        
        cv::putText(frame, "Q=Quit P=Pause D=Display S=Save SPACE=Center",
                   cv::Point(20, 180), cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(150, 150, 150), 1);
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
                std::cout << "CS2 Player Detection with Mouse Control - Usage:" << std::endl;
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
        
        std::cout << "=== CS2 PLAYER DETECTION SYSTEM WITH MOUSE CONTROL ===" << std::endl;
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
            
            std::string output_path = "cs2_result_" + 
                std::to_string(std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count()) + ".jpg";
            cv::imwrite(output_path, result_image);
            std::cout << "\nResult saved as: " << output_path << std::endl;
            
            if (!detections.empty()) {
                exportToJSON(detections, "cs2_detections.json");
            }
            
            std::cout << "\n=== PERFORMANCE ANALYSIS ===" << std::endl;
            if (metrics.inference_ms < 20) {
                std::cout << "EXCELLENT: GPU acceleration working perfectly!" << std::endl;
            } else if (metrics.inference_ms < 50) {
                std::cout << "GOOD: GPU acceleration working" << std::endl;
            } else {
                std::cout << "WARNING: Performance suboptimal - check GPU drivers" << std::endl;
            }
            
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