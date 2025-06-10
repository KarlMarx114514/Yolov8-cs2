#include "detector.hpp"
#include <iostream>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <onnxruntime_c_api.h>

#ifdef _WIN32
#include <windows.h>
#endif

namespace cs2_detection {

YOLOv8Detector::YOLOv8Detector(const std::string& model_path, bool use_gpu) {
    std::cout << "Initializing CS2 Player Detection..." << std::endl;
    
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YOLOv8CS2");
    
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetInterOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    
    if (use_gpu) {
        try {
            std::cout << "Attempting CUDA..." << std::endl;
            
            OrtCUDAProviderOptions cuda_options{};
            cuda_options.device_id = 0;
            cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
            cuda_options.gpu_mem_limit = SIZE_MAX;
            cuda_options.arena_extend_strategy = 1;
            cuda_options.do_copy_in_default_stream = 1;
            cuda_options.has_user_compute_stream = 0;
            cuda_options.user_compute_stream = nullptr;
            cuda_options.default_memory_arena_cfg = nullptr;
            
            session_options.AppendExecutionProvider_CUDA(cuda_options);
            gpu_acceleration_enabled = true;
            std::cout << "CUDA provider added successfully!" << std::endl;
            
        } catch (const Ort::Exception& e) {
            std::cout << "CUDA failed: " << e.what() << std::endl;
            try {
                std::cout << "Attempting DirectML..." << std::endl;
                session_options.AppendExecutionProvider("DML");
                gpu_acceleration_enabled = true;
                std::cout << "DirectML provider added!" << std::endl;
            } catch (const Ort::Exception& e2) {
                std::cout << "DirectML failed: " << e2.what() << std::endl;
                gpu_acceleration_enabled = false;
            }
        } catch (const std::exception&) {
            std::cout << "GPU setup failed" << std::endl;
            gpu_acceleration_enabled = false;
        }
    }
    
    std::cout << "Loading CS2 model from: " << model_path << std::endl;
    
    try {
#ifdef _WIN32
        std::wstring model_path_w(model_path.begin(), model_path.end());
        session = Ort::Session(env, model_path_w.c_str(), session_options);
#else
        session = Ort::Session(env, model_path.c_str(), session_options);
#endif
        std::cout << "CS2 model loaded successfully!" << std::endl;
    } catch (const Ort::Exception& e) {
        std::cerr << "Failed to load model: " << e.what() << std::endl;
        throw;
    }
    
    printProviderInfo();
    
    memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    
    size_t num_input_nodes = session.GetInputCount();
    for (size_t i = 0; i < num_input_nodes; i++) {
        auto input_name = session.GetInputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
        input_names_ptrs.push_back(std::move(input_name));
        input_names_cstr.push_back(input_names_ptrs.back().get());
    }
    
    size_t num_output_nodes = session.GetOutputCount();
    for (size_t i = 0; i < num_output_nodes; i++) {
        auto output_name = session.GetOutputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
        output_names_ptrs.push_back(std::move(output_name));
        output_names_cstr.push_back(output_names_ptrs.back().get());
    }
    
    try {
        auto input_info = session.GetInputTypeInfo(0);
        auto tensor_info = input_info.GetTensorTypeAndShapeInfo();
        auto input_shape = tensor_info.GetShape();
        
        if (input_shape.size() >= 4) {
            if (input_shape[2] > 0 && input_shape[3] > 0) {
                input_height = static_cast<int>(input_shape[2]);
                input_width = static_cast<int>(input_shape[3]);
            }
        }
        std::cout << "Model input size: " << input_width << "x" << input_height << std::endl;
    } catch (const std::exception&) {
        std::cout << "Using default input size: " << input_width << "x" << input_height << std::endl;
    }
    
    std::cout << "CS2 Player Detector ready!" << std::endl;
}

float YOLOv8Detector::calculateIoU(const cv::Rect& box1, const cv::Rect& box2) {
    int x1 = std::max(box1.x, box2.x);
    int y1 = std::max(box1.y, box2.y);
    int x2 = std::min(box1.x + box1.width, box2.x + box2.width);
    int y2 = std::min(box1.y + box1.height, box2.y + box2.height);
    
    if (x2 <= x1 || y2 <= y1) return 0.0f;
    
    float intersection = static_cast<float>((x2 - x1) * (y2 - y1));
    float area1 = static_cast<float>(box1.width * box1.height);
    float area2 = static_cast<float>(box2.width * box2.height);
    float union_area = area1 + area2 - intersection;
    
    return intersection / union_area;
}

std::vector<Detection> YOLOv8Detector::applyNMS(std::vector<Detection>& detections, float nms_threshold) {
    if (detections.empty()) return detections;
    
    std::sort(detections.begin(), detections.end(), 
              [](const Detection& a, const Detection& b) {
                  return a.confidence > b.confidence;
              });
    
    std::vector<Detection> filtered_detections;
    std::vector<bool> suppressed(detections.size(), false);
    
    for (size_t i = 0; i < detections.size(); i++) {
        if (suppressed[i]) continue;
        
        filtered_detections.push_back(detections[i]);
        
        for (size_t j = i + 1; j < detections.size(); j++) {
            if (suppressed[j]) continue;
            
            float iou = calculateIoU(detections[i].bbox, detections[j].bbox);
            if (iou > nms_threshold) {
                suppressed[j] = true;
            }
        }
    }
    
    return filtered_detections;
}

void YOLOv8Detector::printProviderInfo() {
    std::cout << "\n=== EXECUTION PROVIDER DIAGNOSTICS ===" << std::endl;
    
    auto available_providers = Ort::GetAvailableProviders();
    std::cout << "Available providers: ";
    for (const auto& provider : available_providers) {
        std::cout << provider << " ";
    }
    std::cout << std::endl;
    
    if (gpu_acceleration_enabled) {
        std::cout << "GPU acceleration requested and configured" << std::endl;
    } else {
        std::cout << "Using CPU execution" << std::endl;
    }
    std::cout << "=======================================" << std::endl;
}

std::vector<Detection> YOLOv8Detector::postprocess(std::vector<Ort::Value>& outputs, 
                                 cv::Size original_size, float conf_threshold, float nms_threshold) {
    std::vector<Detection> detections;
    
    if (outputs.empty()) {
        return detections;
    }
    
    float* output_data = outputs[0].GetTensorMutableData<float>();
    auto output_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    
    int num_detections = static_cast<int>(output_shape[2]);
    int num_attributes = static_cast<int>(output_shape[1]);
    
    float scale_x = static_cast<float>(original_size.width) / input_width;
    float scale_y = static_cast<float>(original_size.height) / input_height;
    
    for (int i = 0; i < num_detections; i++) {
        float max_confidence = 0.0f;
        int best_class = -1;
        
        for (int class_idx = 0; class_idx < 4; class_idx++) {
            float class_conf = output_data[(4 + class_idx) * num_detections + i];
            if (class_conf > max_confidence) {
                max_confidence = class_conf;
                best_class = class_idx;
            }
        }
        
        if (max_confidence > conf_threshold && best_class >= 0) {
            float x_center = output_data[0 * num_detections + i];
            float y_center = output_data[1 * num_detections + i];
            float width = output_data[2 * num_detections + i];
            float height = output_data[3 * num_detections + i];
            
            float x_min = (x_center - width / 2.0f) * scale_x;
            float y_min = (y_center - height / 2.0f) * scale_y;
            float bbox_width = width * scale_x;
            float bbox_height = height * scale_y;
            
            x_min = std::max(0.0f, std::min(x_min, static_cast<float>(original_size.width)));
            y_min = std::max(0.0f, std::min(y_min, static_cast<float>(original_size.height)));
            bbox_width = std::max(1.0f, std::min(bbox_width, static_cast<float>(original_size.width) - x_min));
            bbox_height = std::max(1.0f, std::min(bbox_height, static_cast<float>(original_size.height) - y_min));
            
            Detection det;
            det.confidence = max_confidence;
            det.class_id = best_class;
            det.class_name = class_names[best_class];
            det.bbox = cv::Rect(static_cast<int>(x_min), static_cast<int>(y_min), 
                               static_cast<int>(bbox_width), static_cast<int>(bbox_height));
            
            detections.push_back(det);
        }
    }
    
    return applyNMS(detections, nms_threshold);
}

void YOLOv8Detector::warmup(int warmup_runs) {
    std::cout << "\n=== GPU WARMUP ===" << std::endl;
    cv::Mat dummy_image = cv::Mat::zeros(640, 640, CV_8UC3);
    
    for (int i = 0; i < warmup_runs; i++) {
        std::cout << "Warmup run " << (i + 1) << "/" << warmup_runs << "..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        detect(dummy_image, 0.2f, 0.45f);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "  Time: " << duration.count() << "ms" << std::endl;
    }
    std::cout << "Warmup complete!" << std::endl;
}

cv::Mat YOLOv8Detector::preprocess(const cv::Mat& image) {
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(input_width, input_height));
    
    cv::Mat blob;
    resized.convertTo(blob, CV_32F, 1.0/255.0);
    cv::cvtColor(blob, blob, cv::COLOR_BGR2RGB);
    
    return blob;
}

std::vector<Detection> YOLOv8Detector::detect(const cv::Mat& image, float conf_threshold, float nms_threshold) {
    try {
        cv::Mat input_blob = preprocess(image);
        
        std::vector<int64_t> input_shape = {1, 3, input_height, input_width};
        size_t input_tensor_size = 3 * input_height * input_width;
        std::vector<float> input_tensor_values(input_tensor_size);
        
        std::vector<cv::Mat> channels;
        cv::split(input_blob, channels);
        for (int c = 0; c < 3; ++c) {
            std::memcpy(input_tensor_values.data() + c * input_height * input_width,
                       channels[c].data, input_height * input_width * sizeof(float));
        }
        
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_tensor_values.data(), input_tensor_size,
            input_shape.data(), input_shape.size());
        
        auto output_tensors = session.Run(Ort::RunOptions{nullptr},
                                        input_names_cstr.data(), &input_tensor, 1,
                                        output_names_cstr.data(), output_names_cstr.size());
        
        return postprocess(output_tensors, image.size(), conf_threshold, nms_threshold);
        
    } catch (const std::exception& e) {
        std::cerr << "Error during detection: " << e.what() << std::endl;
        return std::vector<Detection>();
    }
}

DetectionResult YOLOv8Detector::detectWithMetrics(const cv::Mat& image, float conf_threshold, float nms_threshold) {
    PerformanceMetrics metrics;
    
    try {
        auto total_start = std::chrono::high_resolution_clock::now();
        
        auto prep_start = std::chrono::high_resolution_clock::now();
        cv::Mat input_blob = preprocess(image);
        
        std::vector<int64_t> input_shape = {1, 3, input_height, input_width};
        size_t input_tensor_size = 3 * input_height * input_width;
        std::vector<float> input_tensor_values(input_tensor_size);
        
        std::vector<cv::Mat> channels;
        cv::split(input_blob, channels);
        for (int c = 0; c < 3; ++c) {
            std::memcpy(input_tensor_values.data() + c * input_height * input_width,
                       channels[c].data, input_height * input_width * sizeof(float));
        }
        
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_tensor_values.data(), input_tensor_size,
            input_shape.data(), input_shape.size());
            
        auto prep_end = std::chrono::high_resolution_clock::now();
        metrics.preprocessing_ms = std::chrono::duration_cast<std::chrono::microseconds>(prep_end - prep_start).count() / 1000.0f;
        
        auto inference_start = std::chrono::high_resolution_clock::now();
        auto output_tensors = session.Run(Ort::RunOptions{nullptr},
                                        input_names_cstr.data(), &input_tensor, 1,
                                        output_names_cstr.data(), output_names_cstr.size());
        auto inference_end = std::chrono::high_resolution_clock::now();
        metrics.inference_ms = std::chrono::duration_cast<std::chrono::microseconds>(inference_end - inference_start).count() / 1000.0f;
        
        auto post_start = std::chrono::high_resolution_clock::now();
        auto detections = postprocess(output_tensors, image.size(), conf_threshold, nms_threshold);
        auto post_end = std::chrono::high_resolution_clock::now();
        metrics.postprocessing_ms = std::chrono::duration_cast<std::chrono::microseconds>(post_end - post_start).count() / 1000.0f;
        
        auto total_end = std::chrono::high_resolution_clock::now();
        metrics.total_ms = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start).count() / 1000.0f;
        metrics.fps = 1000.0f / metrics.total_ms;
        
        return {detections, metrics};
        
    } catch (const std::exception& e) {
        std::cerr << "Error during detection: " << e.what() << std::endl;
        return {std::vector<Detection>(), metrics};
    }
}

PerformanceMetrics YOLOv8Detector::benchmark(const cv::Mat& image, int num_runs, float conf_threshold, float nms_threshold) {
    std::cout << "\n=== BENCHMARKING (" << num_runs << " runs) ===" << std::endl;
    
    std::vector<PerformanceMetrics> all_metrics;
    
    std::cout << "Warmup run..." << std::endl;
    detectWithMetrics(image, conf_threshold, nms_threshold);
    
    for (int i = 0; i < num_runs; i++) {
        std::cout << "Run " << (i + 1) << "/" << num_runs << "..." << std::endl;
        auto result = detectWithMetrics(image, conf_threshold, nms_threshold);
        all_metrics.push_back(result.second);
    }
    
    PerformanceMetrics avg_metrics;
    for (const auto& m : all_metrics) {
        avg_metrics.preprocessing_ms += m.preprocessing_ms;
        avg_metrics.inference_ms += m.inference_ms;
        avg_metrics.postprocessing_ms += m.postprocessing_ms;
        avg_metrics.total_ms += m.total_ms;
    }
    
    avg_metrics.preprocessing_ms /= num_runs;
    avg_metrics.inference_ms /= num_runs;
    avg_metrics.postprocessing_ms /= num_runs;
    avg_metrics.total_ms /= num_runs;
    avg_metrics.fps = 1000.0f / avg_metrics.total_ms;
    
    std::cout << "\n=== AVERAGE PERFORMANCE ===" << std::endl;
    avg_metrics.print();
    
    float variance = 0.0f;
    for (const auto& m : all_metrics) {
        variance += (m.total_ms - avg_metrics.total_ms) * (m.total_ms - avg_metrics.total_ms);
    }
    float std_dev = std::sqrt(variance / num_runs);
    std::cout << "Standard deviation: " << std_dev << " ms" << std::endl;
    
    return avg_metrics;
}

void YOLOv8Detector::drawResults(cv::Mat& image, const std::vector<Detection>& detections) {
    for (const auto& det : detections) {
        cv::Scalar color = class_colors[det.class_id];
        
        // Draw thick bounding box
        cv::rectangle(image, det.bbox, color, 3);
        
        // Draw label with confidence
        std::string label = det.class_name + ": " + std::to_string(det.confidence).substr(0, 4);
        
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseline);
        
        // Draw background rectangle for text
        cv::Point text_origin(det.bbox.x, det.bbox.y - text_size.height - 8);
        cv::rectangle(image, 
                     cv::Point(text_origin.x - 2, text_origin.y - baseline - 2),
                     cv::Point(text_origin.x + text_size.width + 4, text_origin.y + text_size.height + 4),
                     color, -1);
        
        // Draw text
        cv::putText(image, label, text_origin, cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    }
}

} // namespace cs2_detection