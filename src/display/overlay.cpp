#include "display/overlay.hpp"
#include <iostream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <atomic>
#include <mutex>

#ifdef _WIN32
#include <windows.h>
#endif

namespace cs2_display {

// PerformanceOverlay implementation
void PerformanceOverlay::addPerformanceOverlay(cv::Mat& frame, 
                                              const cs2_detection::CaptureMetrics& capture_metrics,
                                              const cs2_detection::PerformanceMetrics& detection_metrics,
                                              int detection_count, 
                                              const std::vector<int>& class_counts) {
    
    // Fast exit if frame empty
    if (frame.empty()) {
        return;
    }
    
    try {
        // Optimized overlay rendering - pre-calculate positions and colors
        static const cv::Scalar bg_color(0, 0, 0);
        static const cv::Scalar title_color(0, 255, 255);
        static const cv::Scalar fps_color(0, 255, 0);
        static const cv::Scalar text_color(255, 255, 255);
        static const cv::Scalar detection_color(255, 255, 0);
        static const cv::Scalar hint_color(150, 150, 150);
        
        // Create semi-transparent overlay background (optimized)
        cv::Mat overlay = cv::Mat::zeros(200, 500, frame.type());
        cv::rectangle(overlay, cv::Point(0, 0), cv::Point(500, 200), bg_color, -1);
        
        // Blend overlay with frame efficiently
        cv::Rect overlay_roi(10, 10, std::min(490, frame.cols - 10), std::min(190, frame.rows - 10));
        if (overlay_roi.width > 0 && overlay_roi.height > 0) {
            cv::Mat frame_roi = frame(overlay_roi);
            cv::Mat overlay_resized = overlay(cv::Rect(0, 0, overlay_roi.width, overlay_roi.height));
            cv::addWeighted(frame_roi, 0.75, overlay_resized, 0.25, 0, frame_roi);
        }
        
        // Text rendering with optimized positioning
        int y_pos = 35;
        const int line_height = 25;
        const int x_pos = 20;
        
        // Main title
        cv::putText(frame, "CS2 MULTI-THREADED DETECTION + MOUSE CONTROL", 
                   cv::Point(x_pos, y_pos), cv::FONT_HERSHEY_SIMPLEX, 0.6, title_color, 2);
        y_pos += line_height;
        
        // FPS display (most important metric)
        std::string fps_text = "FPS: " + std::to_string(static_cast<int>(capture_metrics.fps));
        cv::putText(frame, fps_text, cv::Point(x_pos, y_pos), cv::FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2);
        y_pos += line_height;
        
        // Latency display (critical for responsiveness)
        float total_latency = capture_metrics.total_ms + detection_metrics.total_ms;
        std::string latency_text = "Latency: " + std::to_string(static_cast<int>(total_latency)) + "ms";
        cv::putText(frame, latency_text, cv::Point(x_pos, y_pos), cv::FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1);
        y_pos += line_height;
        
        // Detection count
        std::string detection_text = "Players: " + std::to_string(detection_count);
        cv::putText(frame, detection_text, cv::Point(x_pos, y_pos), cv::FONT_HERSHEY_SIMPLEX, 0.6, detection_color, 2);
        y_pos += line_height;
        
        // Class breakdown (if we have detections and valid class counts)
        if (detection_count > 0 && class_counts.size() >= 4) {
            std::string class_breakdown = "CT:" + std::to_string(class_counts[0]) + 
                                        " T:" + std::to_string(class_counts[2]) + 
                                        " +H:" + std::to_string(class_counts[1] + class_counts[3]);
            cv::putText(frame, class_breakdown, cv::Point(x_pos, y_pos), cv::FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1);
            y_pos += line_height;
        }
        
        // Performance indicator circle (green = good, yellow = ok, red = bad)
        cv::Scalar perf_color = total_latency < 16 ? cv::Scalar(0, 255, 0) :     // < 16ms = excellent
                               total_latency < 33 ? cv::Scalar(0, 255, 255) :    // < 33ms = good  
                               cv::Scalar(0, 0, 255);                            // >= 33ms = needs optimization
        cv::circle(frame, cv::Point(450, 35), 10, perf_color, -1);
        
        // Threading status indicator
        cv::putText(frame, "[MULTI-THREADED]", cv::Point(350, 60), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(100, 255, 100), 1);
        
        // Control hints (compact)
        cv::putText(frame, "M=Mouse V=Debug C=Refresh Q=Quit P=Pause",
                   cv::Point(x_pos, 160), cv::FONT_HERSHEY_SIMPLEX, 0.35, hint_color, 1);
        
        cv::putText(frame, "1/2/3=Input Mode +/-=Sensitivity 4/5/6=Presets",
                   cv::Point(x_pos, 180), cv::FONT_HERSHEY_SIMPLEX, 0.35, hint_color, 1);
                   
    } catch (const cv::Exception& e) {
        // Handle OpenCV errors gracefully to prevent crashes
        static int error_count = 0;
        if (++error_count < 5) { // Limit error spam
            std::cout << "Overlay rendering error: " << e.what() << std::endl;
        }
        if (error_count == 5) {
            std::cout << "Too many overlay errors, suppressing further messages..." << std::endl;
        }
    }
}

// DisplayWindow implementation with thread safety improvements
DisplayWindow::DisplayWindow(const std::string& window_name) 
    : window_name(window_name), display_enabled(false), window_mutex() {
}

DisplayWindow::~DisplayWindow() {
    std::lock_guard<std::mutex> lock(window_mutex);
    if (display_enabled) {
        try {
            cv::destroyWindow(window_name);
        } catch (const cv::Exception& e) {
            // Suppress errors during destruction
        }
    }
}

bool DisplayWindow::setupWindow() {
    std::lock_guard<std::mutex> lock(window_mutex);
    
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
        
        // Position window away from game area (bottom-right corner)
        cv::moveWindow(window_name, screen_width - window_width - 50, screen_height - window_height - 100);
        cv::resizeWindow(window_name, window_width, window_height);
        cv::setWindowProperty(window_name, cv::WND_PROP_TOPMOST, 0);
        
        display_enabled = true;
        std::cout << "Display window created and positioned away from game area" << std::endl;
        
        return true;
        
    } catch (const cv::Exception& e) {
        std::cout << "Could not create display window: " << e.what() << std::endl;
        display_enabled = false;
        return false;
    }
}

void DisplayWindow::showFrame(const cv::Mat& frame) {
    // Early exit if display disabled or frame empty
    if (!display_enabled || frame.empty()) {
        return;
    }
    
    // Thread-safe frame display
    std::lock_guard<std::mutex> lock(window_mutex);
    
    try {
        if (display_enabled) { // Double-check after acquiring lock
            cv::imshow(window_name, frame);
        }
    } catch (const cv::Exception& e) {
        std::cout << "Display error: " << e.what() << std::endl;
        display_enabled = false;
    }
}

char DisplayWindow::handleInput(bool& paused, bool save_results, bool update_display) {
    if (!display_enabled) {
        return 0;
    }
    
    // Non-blocking input handling for better responsiveness
    char key = 0;
    try {
        key = cv::waitKey(update_display ? 1 : 50) & 0xFF;
    } catch (const cv::Exception& e) {
        // Handle OpenCV window errors
        std::cout << "Input handling error: " << e.what() << std::endl;
        return 0;
    }
    
    // Handle basic window controls first
    if (key == 'q' || key == 'Q' || key == 27) { // Quit
        return 'q';
    } else if (key == 'p' || key == 'P') { // Pause
        paused = !paused;
        std::cout << (paused ? "\n[PAUSED] All threads suspended" : "\n[RESUMED] All threads active") << std::endl;
        return 'p';
    } else if (key == 's' || key == 'S') { // Save
        if (save_results) {
            std::cout << "\n[SAVE] Screenshot queued for background save" << std::endl;
        }
        return 's';
    } else if (key == 'd' || key == 'D') { // Display toggle
        toggleDisplay();
        return 'd';
    }
    
    // Return the key for other modules to handle
    return key;
}

void DisplayWindow::toggleDisplay() {
    std::lock_guard<std::mutex> lock(window_mutex);
    
    display_enabled = !display_enabled;
    if (!display_enabled) {
        try {
            cv::destroyWindow(window_name);
        } catch (const cv::Exception& e) {
            // Suppress errors
        }
        std::cout << "\n[DISPLAY OFF] - Performance optimized for mouse control only" << std::endl;
    } else {
        try {
            cv::namedWindow(window_name, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
        } catch (const cv::Exception& e) {
            std::cout << "Failed to re-enable display: " << e.what() << std::endl;
            display_enabled = false;
            return;
        }
        std::cout << "\n[DISPLAY ON] - Visual feedback enabled" << std::endl;
    }
}

#ifdef _WIN32
HWND DisplayWindow::getWindowHandle() const {
    std::lock_guard<std::mutex> lock(window_mutex);
    
    if (!display_enabled) {
        return nullptr;
    }
    
    try {
        // Find the OpenCV window by its title
        HWND cv_window = FindWindowA(nullptr, window_name.c_str());
        return cv_window;
    } catch (...) {
        return nullptr;
    }
}
#endif

} // namespace cs2_display