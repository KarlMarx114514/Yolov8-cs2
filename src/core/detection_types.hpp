#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>

namespace cs2_detection {

struct Detection {
    cv::Rect bbox;
    float confidence;
    int class_id;
    std::string class_name;
};

struct PerformanceMetrics {
    float preprocessing_ms = 0.0f;
    float inference_ms = 0.0f;
    float postprocessing_ms = 0.0f;
    float total_ms = 0.0f;
    float fps = 0.0f;
    
    void print() const {
        std::cout << "\n=== PERFORMANCE METRICS ===" << std::endl;
        std::cout << "Preprocessing:   " << preprocessing_ms << " ms" << std::endl;
        std::cout << "Inference:       " << inference_ms << " ms" << std::endl;
        std::cout << "Postprocessing:  " << postprocessing_ms << " ms" << std::endl;
        std::cout << "Total:           " << total_ms << " ms" << std::endl;
        std::cout << "FPS:             " << fps << std::endl;
        std::cout << "==========================" << std::endl;
    }
};

using DetectionResult = std::pair<std::vector<Detection>, PerformanceMetrics>;

} // namespace cs2_detection