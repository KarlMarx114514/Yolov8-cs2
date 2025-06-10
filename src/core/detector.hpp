#pragma once
#include "detection_types.hpp"
#include <onnxruntime_cxx_api.h>
#include <memory>

namespace cs2_detection {

class YOLOv8Detector {
public:
    explicit YOLOv8Detector(const std::string& model_path, bool use_gpu = true);
    ~YOLOv8Detector() = default;
    
    // Core detection methods
    std::vector<Detection> detect(const cv::Mat& image, 
                                 float conf_threshold = 0.2f, 
                                 float nms_threshold = 0.45f);
    
    DetectionResult detectWithMetrics(const cv::Mat& image, 
                                    float conf_threshold = 0.2f, 
                                    float nms_threshold = 0.45f);
    
    // Benchmarking
    PerformanceMetrics benchmark(const cv::Mat& image, int num_runs = 10,
                               float conf_threshold = 0.2f, float nms_threshold = 0.45f);
    
    // Utility methods
    void warmup(int warmup_runs = 3);
    cv::Mat preprocess(const cv::Mat& image);
    void drawResults(cv::Mat& image, const std::vector<Detection>& detections);
    
private:
    // ONNX Runtime components
    Ort::Session session{nullptr};
    Ort::MemoryInfo memory_info{nullptr};
    std::vector<Ort::AllocatedStringPtr> input_names_ptrs;
    std::vector<Ort::AllocatedStringPtr> output_names_ptrs;
    std::vector<const char*> input_names_cstr;
    std::vector<const char*> output_names_cstr;
    
    // Model input dimensions
    int input_width = 640;
    int input_height = 640;
    
    // CS2 class names and colors
    std::vector<std::string> class_names = {"CT", "CT+Helmet", "T", "T+Helmet"};
    std::vector<cv::Scalar> class_colors = {
        cv::Scalar(255, 100, 100),   // CT - Light Blue
        cv::Scalar(255, 150, 100),   // CT+Helmet - Lighter Blue
        cv::Scalar(100, 100, 255),   // T - Light Red
        cv::Scalar(100, 150, 255)    // T+Helmet - Orange
    };
    
    bool gpu_acceleration_enabled = false;
    
    // Helper methods
    float calculateIoU(const cv::Rect& box1, const cv::Rect& box2);
    std::vector<Detection> applyNMS(std::vector<Detection>& detections, float nms_threshold = 0.45f);
    void printProviderInfo();
    std::vector<Detection> postprocess(std::vector<Ort::Value>& outputs, 
                                     cv::Size original_size, float conf_threshold, float nms_threshold);
};

} // namespace cs2_detection