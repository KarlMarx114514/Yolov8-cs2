#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <thread>
#include <chrono>
#include <onnxruntime_c_api.h>
#include <cmath>

class YOLOv8CS2 {
public:
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

private:
    Ort::Session session{nullptr};
    Ort::MemoryInfo memory_info{nullptr};
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::vector<Ort::AllocatedStringPtr> input_names_ptrs;
    std::vector<Ort::AllocatedStringPtr> output_names_ptrs;
    std::vector<const char*> input_names_cstr;
    std::vector<const char*> output_names_cstr;
    
    // Model input dimensions - Updated for better detection
    int input_width = 640;
    int input_height = 640;
    
    // CS2 class names
    std::vector<std::string> class_names = {"CT", "CT+Helmet", "T", "T+Helmet"};
    std::vector<cv::Scalar> class_colors = {
        cv::Scalar(255, 0, 0),     // CT - Blue
        cv::Scalar(255, 100, 0),   // CT+Helmet - Light Blue
        cv::Scalar(0, 0, 255),     // T - Red
        cv::Scalar(0, 100, 255)    // T+Helmet - Orange
    };
    
    // Track if GPU is enabled
    bool gpu_acceleration_enabled = false;

    // NMS helper function to calculate IoU
    float calculateIoU(const cv::Rect& box1, const cv::Rect& box2) {
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

    // Non-Maximum Suppression
    std::vector<Detection> applyNMS(std::vector<Detection>& detections, float nms_threshold = 0.45f) {
        if (detections.empty()) return detections;
        
        // Sort by confidence (highest first)
        std::sort(detections.begin(), detections.end(), 
                  [](const Detection& a, const Detection& b) {
                      return a.confidence > b.confidence;
                  });
        
        std::vector<Detection> filtered_detections;
        std::vector<bool> suppressed(detections.size(), false);
        
        for (size_t i = 0; i < detections.size(); i++) {
            if (suppressed[i]) continue;
            
            filtered_detections.push_back(detections[i]);
            
            // Suppress overlapping detections
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

    void printProviderInfo() {
        std::cout << "\n=== EXECUTION PROVIDER DIAGNOSTICS ===" << std::endl;
        
        // Get available providers
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

    std::vector<Detection> postprocess(std::vector<Ort::Value>& outputs, 
                                     cv::Size original_size, float conf_threshold, float nms_threshold) {
        std::vector<Detection> detections;
        
        if (outputs.empty()) {
            return detections;
        }
        
        float* output_data = outputs[0].GetTensorMutableData<float>();
        auto output_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        
        int num_detections = static_cast<int>(output_shape[2]);
        int num_attributes = static_cast<int>(output_shape[1]);
        
        std::cout << "Output shape: [" << output_shape[0] << ", " << output_shape[1] << ", " << output_shape[2] << "]" << std::endl;
        
        float scale_x = static_cast<float>(original_size.width) / input_width;
        float scale_y = static_cast<float>(original_size.height) / input_height;
        
        int detection_count = 0;
        
        // Process detections - CS2 format: [x, y, w, h, class0_conf, class1_conf, class2_conf, class3_conf]
        for (int i = 0; i < num_detections; i++) {
            // Get class with highest confidence
            float max_confidence = 0.0f;
            int best_class = -1;
            
            // Check all 4 classes (starting from index 4)
            for (int class_idx = 0; class_idx < 4; class_idx++) {
                float class_conf = output_data[(4 + class_idx) * num_detections + i];
                if (class_conf > max_confidence) {
                    max_confidence = class_conf;
                    best_class = class_idx;
                }
            }
            
            if (max_confidence > conf_threshold && best_class >= 0) {
                detection_count++;
                
                // Extract bounding box coordinates
                float x_center = output_data[0 * num_detections + i];
                float y_center = output_data[1 * num_detections + i];
                float width = output_data[2 * num_detections + i];
                float height = output_data[3 * num_detections + i];
                
                // Convert to original image coordinates
                float x_min = (x_center - width / 2.0f) * scale_x;
                float y_min = (y_center - height / 2.0f) * scale_y;
                float bbox_width = width * scale_x;
                float bbox_height = height * scale_y;
                
                // Clamp to image bounds
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
        
        std::cout << "Found " << detection_count << " players before NMS" << std::endl;
        
        // Apply NMS
        auto filtered_detections = applyNMS(detections, nms_threshold);
        
        // Count by class
        std::vector<int> class_counts(4, 0);
        for (const auto& det : filtered_detections) {
            class_counts[det.class_id]++;
        }
        
        std::cout << "After NMS: " << filtered_detections.size() << " players total" << std::endl;
        std::cout << "  CT: " << class_counts[0] << ", CT+Helmet: " << class_counts[1] 
                  << ", T: " << class_counts[2] << ", T+Helmet: " << class_counts[3] << std::endl;
        
        return filtered_detections;
    }

public:
    YOLOv8CS2(const std::string& model_path, bool use_gpu = true) {
        std::cout << "Initializing CS2 Player Detection..." << std::endl;
        
        // Initialize ONNX Runtime
        static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YOLOv8CS2");
        
        // Session options
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetInterOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        
        // GPU setup
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
        
        // Create session
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
        
        // Get input/output info
        memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        
        // Get input names
        size_t num_input_nodes = session.GetInputCount();
        for (size_t i = 0; i < num_input_nodes; i++) {
            auto input_name = session.GetInputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
            input_names_ptrs.push_back(std::move(input_name));
            input_names_cstr.push_back(input_names_ptrs.back().get());
        }
        
        // Get output names
        size_t num_output_nodes = session.GetOutputCount();
        for (size_t i = 0; i < num_output_nodes; i++) {
            auto output_name = session.GetOutputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
            output_names_ptrs.push_back(std::move(output_name));
            output_names_cstr.push_back(output_names_ptrs.back().get());
        }
        
        // Get input shape
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

    void warmup(int warmup_runs = 3) {
        std::cout << "\n=== GPU WARMUP ===" << std::endl;
        cv::Mat dummy_image = cv::Mat::zeros(640, 640, CV_8UC3); // Updated size
        
        for (int i = 0; i < warmup_runs; i++) {
            std::cout << "Warmup run " << (i + 1) << "/" << warmup_runs << "..." << std::endl;
            auto start = std::chrono::high_resolution_clock::now();
            detect(dummy_image, 0.2f, 0.45f); // Lower confidence threshold
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "  Time: " << duration.count() << "ms" << std::endl;
        }
        std::cout << "Warmup complete!" << std::endl;
    }
    
    cv::Mat preprocess(const cv::Mat& image) {
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(input_width, input_height));
        
        cv::Mat blob;
        resized.convertTo(blob, CV_32F, 1.0/255.0);
        cv::cvtColor(blob, blob, cv::COLOR_BGR2RGB);
        
        return blob;
    }
    
    std::vector<Detection> detect(const cv::Mat& image, float conf_threshold = 0.2f, float nms_threshold = 0.45f) { // Lowered default confidence
        try {
            cv::Mat input_blob = preprocess(image);
            
            // Create input tensor
            std::vector<int64_t> input_shape = {1, 3, input_height, input_width};
            size_t input_tensor_size = 3 * input_height * input_width;
            std::vector<float> input_tensor_values(input_tensor_size);
            
            // HWC to CHW conversion
            std::vector<cv::Mat> channels;
            cv::split(input_blob, channels);
            for (int c = 0; c < 3; ++c) {
                std::memcpy(input_tensor_values.data() + c * input_height * input_width,
                           channels[c].data, input_height * input_width * sizeof(float));
            }
            
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info, input_tensor_values.data(), input_tensor_size,
                input_shape.data(), input_shape.size());
            
            // Run inference
            auto start_time = std::chrono::high_resolution_clock::now();
            auto output_tensors = session.Run(Ort::RunOptions{nullptr},
                                            input_names_cstr.data(), &input_tensor, 1,
                                            output_names_cstr.data(), output_names_cstr.size());
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            
            std::cout << "Inference time: " << duration.count() << " ms" << std::endl;
            
            return postprocess(output_tensors, image.size(), conf_threshold, nms_threshold);
            
        } catch (const std::exception& e) {
            std::cerr << "Error during detection: " << e.what() << std::endl;
            return std::vector<Detection>();
        }
    }

    std::pair<std::vector<Detection>, PerformanceMetrics> detectWithMetrics(const cv::Mat& image, float conf_threshold = 0.2f, float nms_threshold = 0.45f) { // Lowered default confidence
        PerformanceMetrics metrics;
        
        try {
            auto total_start = std::chrono::high_resolution_clock::now();
            
            // Preprocessing
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
            
            // Inference
            auto inference_start = std::chrono::high_resolution_clock::now();
            auto output_tensors = session.Run(Ort::RunOptions{nullptr},
                                            input_names_cstr.data(), &input_tensor, 1,
                                            output_names_cstr.data(), output_names_cstr.size());
            auto inference_end = std::chrono::high_resolution_clock::now();
            metrics.inference_ms = std::chrono::duration_cast<std::chrono::microseconds>(inference_end - inference_start).count() / 1000.0f;
            
            // Postprocessing
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

    // Benchmark function for multiple runs
    PerformanceMetrics benchmark(const cv::Mat& image, int num_runs = 10, float conf_threshold = 0.2f, float nms_threshold = 0.45f) { // Lowered default confidence
        std::cout << "\n=== BENCHMARKING (" << num_runs << " runs) ===" << std::endl;
        
        std::vector<PerformanceMetrics> all_metrics;
        
        // Warmup run
        std::cout << "Warmup run..." << std::endl;
        detectWithMetrics(image, conf_threshold, nms_threshold);
        
        // Actual benchmark runs
        for (int i = 0; i < num_runs; i++) {
            std::cout << "Run " << (i + 1) << "/" << num_runs << "..." << std::endl;
            auto result = detectWithMetrics(image, conf_threshold, nms_threshold);
            all_metrics.push_back(result.second);
        }
        
        // Calculate averages
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
        
        // Calculate standard deviation for inference time
        float variance = 0.0f;
        for (const auto& m : all_metrics) {
            variance += (m.total_ms - avg_metrics.total_ms) * (m.total_ms - avg_metrics.total_ms);
        }
        float std_dev = std::sqrt(variance / num_runs);
        std::cout << "Standard deviation: " << std_dev << " ms" << std::endl;
        
        return avg_metrics;
    }
    
    void drawResults(cv::Mat& image, const std::vector<Detection>& detections) {
        for (const auto& det : detections) {
            // Draw bounding box with class-specific color
            cv::Scalar color = class_colors[det.class_id];
            cv::rectangle(image, det.bbox, color, 2);
            
            // Draw label with confidence
            std::string label = det.class_name + ": " + std::to_string(det.confidence).substr(0, 4);
            
            // Calculate text size for background
            int baseline = 0;
            cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseline);
            
            // Draw background rectangle for text
            cv::Point text_origin(det.bbox.x, det.bbox.y - text_size.height - 5);
            cv::rectangle(image, 
                         cv::Point(text_origin.x, text_origin.y - baseline),
                         cv::Point(text_origin.x + text_size.width, text_origin.y + text_size.height),
                         color, -1);
            
            // Draw text
            cv::putText(image, label, text_origin, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
        }
    }
};

// Export to JSON function for CS2 detections
void exportToJSON(const std::vector<YOLOv8CS2::Detection>& detections, const std::string& filename) {
    std::ofstream json_file(filename);
    if (!json_file.is_open()) {
        std::cerr << "Error: Cannot create JSON file " << filename << std::endl;
        return;
    }
    
    json_file << "{\n";
    json_file << "  \"detections\": [\n";
    
    for (size_t i = 0; i < detections.size(); i++) {
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
        if (i < detections.size() - 1) json_file << ",";
        json_file << "\n";
    }
    
    json_file << "  ]\n";
    json_file << "}\n";
    
    json_file.close();
    std::cout << "CS2 detection data exported to: " << filename << std::endl;
}

int main(int argc, char* argv[]) {
    try {
        std::string image_path = "cs2_screenshot.jpg";
        bool run_benchmark = false;
        int benchmark_runs = 10;
        
        // Parse command line arguments
        for (int i = 1; i < argc; i++) {
            if (std::string(argv[i]) == "--benchmark") {
                run_benchmark = true;
                if (i + 1 < argc && std::string(argv[i + 1])[0] != '-') {
                    benchmark_runs = std::atoi(argv[i + 1]);
                    i++;
                }
            } else if (std::string(argv[i]) == "--help" || std::string(argv[i]) == "-h") {
                std::cout << "CS2 Player Detection - Usage:" << std::endl;
                std::cout << argv[0] << " [image_path] [--benchmark [runs]]" << std::endl;
                std::cout << "Classes: CT, CT+Helmet, T, T+Helmet" << std::endl;
                return 0;
            } else {
                image_path = argv[i];
            }
        }
        
        std::cout << "=== CS2 Player Detection ===" << std::endl;
        std::cout << "Loading image: " << image_path << std::endl;
        
        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            std::cerr << "Error: Cannot load image " << image_path << std::endl;
            return -1;
        }
        
        std::cout << "Image loaded! Size: " << image.cols << "x" << image.rows << std::endl;
        
        YOLOv8CS2 detector("yolov8s_cs2.onnx", true);
        detector.warmup(3);
        
        if (run_benchmark) {
            std::cout << "\n=== RUNNING BENCHMARK ===" << std::endl;
            auto avg_metrics = detector.benchmark(image, benchmark_runs, 0.2f, 0.45f); // Lower confidence
            
            // Performance analysis
            std::cout << "\n=== PERFORMANCE ANALYSIS ===" << std::endl;
            if (avg_metrics.inference_ms < 20) { // Adjusted for 640x640
                std::cout << "EXCELLENT: GPU acceleration working perfectly!" << std::endl;
            } else if (avg_metrics.inference_ms < 50) {
                std::cout << "GOOD: GPU acceleration working" << std::endl;
            } else {
                std::cout << "WARNING: Performance suboptimal" << std::endl;
            }
        } else {
            std::cout << "\n=== DETECTING CS2 PLAYERS ===" << std::endl;
            auto result = detector.detectWithMetrics(image, 0.2f, 0.45f); // Lower confidence
            auto detections = result.first;
            auto metrics = result.second;
            
            metrics.print();
            
            // Performance analysis
            std::cout << "\n=== PERFORMANCE ANALYSIS ===" << std::endl;
            if (metrics.inference_ms < 20) { // Adjusted for 640x640
                std::cout << "EXCELLENT: GPU acceleration working perfectly!" << std::endl;
            } else if (metrics.inference_ms < 50) {
                std::cout << "GOOD: GPU acceleration working" << std::endl;
            } else {
                std::cout << "WARNING: Performance suboptimal" << std::endl;
            }
            
            cv::Mat result_image = image.clone();
            if (!detections.empty()) {
                detector.drawResults(result_image, detections);
                std::cout << "\nDetected " << detections.size() << " CS2 players!" << std::endl;
            } else {
                std::cout << "\nNo CS2 players detected" << std::endl;
            }
            
            // Save results
            std::string output_path = "cs2_result_" + image_path;
            cv::imwrite(output_path, result_image);
            std::cout << "Result saved as: " << output_path << std::endl;
            
            if (!detections.empty()) {
                exportToJSON(detections, "cs2_detections.json");
            }
            
            // Display
            try {
                cv::imshow("CS2 Player Detection", result_image);
                std::cout << "\nPress any key to exit..." << std::endl;
                cv::waitKey(0);
                cv::destroyAllWindows();
            } catch (const std::exception&) {
                std::cout << "Display not available (headless mode)" << std::endl;
            }
        }
        
        std::cout << "\nCS2 Detection Complete! " << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}