#include "display/overlay.hpp"
#include <iostream>
#include <iomanip>
#include <opencv2/opencv.hpp>

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
    
    // Create semi-transparent overlay background
    cv::Mat overlay = frame.clone();
    cv::rectangle(overlay, cv::Point(10, 10), cv::Point(500, 200), cv::Scalar(0, 0, 0), -1);
    cv::addWeighted(frame, 0.75, overlay, 0.25, 0, frame);
    
    // Main title
    cv::putText(frame, "CS2 REAL-TIME DETECTION + MOUSE CONTROL", 
               cv::Point(20, 35), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
    
    // FPS display
    cv::putText(frame, "FPS: " + std::to_string(static_cast<int>(capture_metrics.fps)), 
               cv::Point(20, 60), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    
    // Latency display
    float total_latency = capture_metrics.total_ms + detection_metrics.total_ms;
    cv::putText(frame, "Latency: " + std::to_string(static_cast<int>(total_latency)) + "ms",
               cv::Point(20, 85), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    
    // Detection count
    cv::putText(frame, "Players: " + std::to_string(detection_count),
               cv::Point(20, 110), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);
    
    // Class breakdown (if we have detections and valid class counts)
    if (detection_count > 0 && class_counts.size() >= 4) {
        std::string class_breakdown = "CT:" + std::to_string(class_counts[0]) + 
                                    " T:" + std::to_string(class_counts[2]) + 
                                    " +H:" + std::to_string(class_counts[1] + class_counts[3]);
        cv::putText(frame, class_breakdown,
                   cv::Point(20, 135), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);
    }
    
    // Performance indicator circle (green = good, yellow = ok, red = bad)
    cv::Scalar perf_color = total_latency < 20 ? cv::Scalar(0, 255, 0) : 
                           total_latency < 33 ? cv::Scalar(0, 255, 255) : cv::Scalar(0, 0, 255);
    cv::circle(frame, cv::Point(450, 35), 10, perf_color, -1);
    
    // Control hints
    cv::putText(frame, "M=Mouse V=Debug C=Refresh X=Select T=Test",
               cv::Point(20, 160), cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(150, 150, 150), 1);
    
    cv::putText(frame, "Q=Quit P=Pause D=Display S=Save SPACE=Center",
               cv::Point(20, 180), cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(150, 150, 150), 1);
}

// DisplayWindow implementation
DisplayWindow::DisplayWindow(const std::string& window_name) 
    : window_name(window_name), display_enabled(false) {
}

DisplayWindow::~DisplayWindow() {
    if (display_enabled) {
        cv::destroyWindow(window_name);
    }
}

bool DisplayWindow::setupWindow() {
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
        
    } catch (const std::exception& e) {
        std::cout << "Could not create display window: " << e.what() << std::endl;
        display_enabled = false;
        return false;
    }
}

void DisplayWindow::showFrame(const cv::Mat& frame) {
    if (!display_enabled) {
        return;
    }
    
    try {
        cv::imshow(window_name, frame);
    } catch (const std::exception& e) {
        std::cout << "Display error: " << e.what() << std::endl;
        display_enabled = false;
    }
}

char DisplayWindow::handleInput(bool& paused, bool save_results, bool update_display) {
    char key = cv::waitKey(update_display ? 1 : 50) & 0xFF;
    
    // Handle basic window controls first
    if (key == 'q' || key == 'Q' || key == 27) { // Quit
        return 'q';
    } else if (key == 'p' || key == 'P') { // Pause
        paused = !paused;
        std::cout << (paused ? "\n[PAUSED]" : "\n[RESUMED]") << std::endl;
        return 'p';
    } else if (key == 's' || key == 'S') { // Save
        if (save_results) {
            std::cout << "\n[SAVE] Screenshot queued" << std::endl;
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
    display_enabled = !display_enabled;
    if (!display_enabled) {
        cv::destroyWindow(window_name);
        std::cout << "\n[DISPLAY OFF]" << std::endl;
    } else {
        setupWindow();
        std::cout << "\n[DISPLAY ON]" << std::endl;
    }
}

#ifdef _WIN32
HWND DisplayWindow::getWindowHandle() const {
    if (!display_enabled) {
        return nullptr;
    }
    
    // Find the OpenCV window by its title
    HWND cv_window = FindWindowA(nullptr, window_name.c_str());
    return cv_window;
}
#endif

} // namespace cs2_display