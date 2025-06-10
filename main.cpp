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
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>

// Windows-specific includes for screen capture
#ifdef _WIN32
#include <windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <wrl/client.h>
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "winmm.lib")
using Microsoft::WRL::ComPtr;
#endif

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
    
    // Model input dimensions
    int input_width = 640;
    int input_height = 640;
    
    // CS2 class names
    std::vector<std::string> class_names = {"CT", "CT+Helmet", "T", "T+Helmet"};
    std::vector<cv::Scalar> class_colors = {
        cv::Scalar(255, 100, 100),   // CT - Light Blue
        cv::Scalar(255, 150, 100),   // CT+Helmet - Lighter Blue
        cv::Scalar(100, 100, 255),   // T - Light Red
        cv::Scalar(100, 150, 255)    // T+Helmet - Orange
    };
    
    bool gpu_acceleration_enabled = false;

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

    std::vector<Detection> applyNMS(std::vector<Detection>& detections, float nms_threshold = 0.45f) {
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

    void printProviderInfo() {
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

public:
    YOLOv8CS2(const std::string& model_path, bool use_gpu = true) {
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

    void warmup(int warmup_runs = 3) {
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
    
    cv::Mat preprocess(const cv::Mat& image) {
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(input_width, input_height));
        
        cv::Mat blob;
        resized.convertTo(blob, CV_32F, 1.0/255.0);
        cv::cvtColor(blob, blob, cv::COLOR_BGR2RGB);
        
        return blob;
    }
    
    std::vector<Detection> detect(const cv::Mat& image, float conf_threshold = 0.2f, float nms_threshold = 0.45f) {
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

    std::pair<std::vector<Detection>, PerformanceMetrics> detectWithMetrics(const cv::Mat& image, float conf_threshold = 0.2f, float nms_threshold = 0.45f) {
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

    PerformanceMetrics benchmark(const cv::Mat& image, int num_runs = 10, float conf_threshold = 0.2f, float nms_threshold = 0.45f) {
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
    
    void drawResults(cv::Mat& image, const std::vector<Detection>& detections) {
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
};

#ifdef _WIN32
class ScreenCapture {
public:
    struct CaptureMetrics {
        float capture_ms = 0.0f;
        float copy_ms = 0.0f;
        float total_ms = 0.0f;
        int fps = 0;
        
        void print() const {
            std::cout << "Capture: " << capture_ms << "ms, Copy: " << copy_ms 
                      << "ms, Total: " << total_ms << "ms, FPS: " << fps << std::endl;
        }
    };

private:
    ComPtr<ID3D11Device> d3d_device;
    ComPtr<ID3D11DeviceContext> d3d_context;
    ComPtr<IDXGIOutputDuplication> desktop_duplication;
    ComPtr<ID3D11Texture2D> staging_texture;
    
    HDC screen_dc = nullptr;
    HDC memory_dc = nullptr;
    HBITMAP bitmap = nullptr;
    BITMAPINFO bitmap_info;
    
    int capture_width = 0;
    int capture_height = 0;
    int capture_x = 0;
    int capture_y = 0;
    
    bool dx_initialized = false;
    bool gdi_initialized = false;
    
    std::chrono::high_resolution_clock::time_point last_fps_time;
    int frame_count = 0;
    int current_fps = 0;
    
    // Store window information to avoid repeated calculations
    HWND cs2_window = nullptr;
    RECT last_window_rect = {0};
    bool window_info_valid = false;

public:
    ScreenCapture() : last_fps_time(std::chrono::high_resolution_clock::now()) {}
    
    ~ScreenCapture() {
        cleanup();
    }
    
    bool initializeDesktopDuplication(int x = 0, int y = 0, int width = 0, int height = 0) {
        try {
            if (width == 0 || height == 0) {
                capture_x = 0;
                capture_y = 0;
                capture_width = GetSystemMetrics(SM_CXSCREEN);
                capture_height = GetSystemMetrics(SM_CYSCREEN);
            } else {
                capture_x = x;
                capture_y = y;
                capture_width = width;
                capture_height = height;
            }
            
            D3D_FEATURE_LEVEL feature_level;
            HRESULT hr = D3D11CreateDevice(
                nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
                0, nullptr, 0, D3D11_SDK_VERSION,
                &d3d_device, &feature_level, &d3d_context);
                
            if (FAILED(hr)) {
                std::cerr << "Failed to create D3D11 device" << std::endl;
                return false;
            }
            
            ComPtr<IDXGIDevice> dxgi_device;
            hr = d3d_device.As(&dxgi_device);
            if (FAILED(hr)) return false;
            
            ComPtr<IDXGIAdapter> dxgi_adapter;
            hr = dxgi_device->GetAdapter(&dxgi_adapter);
            if (FAILED(hr)) return false;
            
            ComPtr<IDXGIOutput> dxgi_output;
            hr = dxgi_adapter->EnumOutputs(0, &dxgi_output);
            if (FAILED(hr)) return false;
            
            ComPtr<IDXGIOutput1> dxgi_output1;
            hr = dxgi_output.As(&dxgi_output1);
            if (FAILED(hr)) return false;
            
            hr = dxgi_output1->DuplicateOutput(d3d_device.Get(), &desktop_duplication);
            if (FAILED(hr)) {
                std::cerr << "Failed to create desktop duplication. Error: 0x" << std::hex << hr << std::endl;
                return false;
            }
            
            D3D11_TEXTURE2D_DESC staging_desc = {};
            staging_desc.Width = capture_width;
            staging_desc.Height = capture_height;
            staging_desc.MipLevels = 1;
            staging_desc.ArraySize = 1;
            staging_desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
            staging_desc.SampleDesc.Count = 1;
            staging_desc.Usage = D3D11_USAGE_STAGING;
            staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
            
            hr = d3d_device->CreateTexture2D(&staging_desc, nullptr, &staging_texture);
            if (FAILED(hr)) {
                std::cerr << "Failed to create staging texture" << std::endl;
                return false;
            }
            
            dx_initialized = true;
            std::cout << "Desktop Duplication initialized successfully" << std::endl;
            std::cout << "Capture area: " << capture_width << "x" << capture_height 
                      << " at (" << capture_x << "," << capture_y << ")" << std::endl;
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "Exception in Desktop Duplication init: " << e.what() << std::endl;
            return false;
        }
    }
    
    bool initializeGDI(int x = 0, int y = 0, int width = 0, int height = 0) {
        if (width == 0 || height == 0) {
            capture_x = 0;
            capture_y = 0;
            capture_width = GetSystemMetrics(SM_CXSCREEN);
            capture_height = GetSystemMetrics(SM_CYSCREEN);
        } else {
            capture_x = x;
            capture_y = y;
            capture_width = width;
            capture_height = height;
        }
        
        screen_dc = GetDC(nullptr);
        if (!screen_dc) return false;
        
        memory_dc = CreateCompatibleDC(screen_dc);
        if (!memory_dc) return false;
        
        bitmap = CreateCompatibleBitmap(screen_dc, capture_width, capture_height);
        if (!bitmap) return false;
        
        ZeroMemory(&bitmap_info, sizeof(BITMAPINFO));
        bitmap_info.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
        bitmap_info.bmiHeader.biWidth = capture_width;
        bitmap_info.bmiHeader.biHeight = -capture_height;
        bitmap_info.bmiHeader.biPlanes = 1;
        bitmap_info.bmiHeader.biBitCount = 32;
        bitmap_info.bmiHeader.biCompression = BI_RGB;
        
        gdi_initialized = true;
        std::cout << "GDI capture initialized" << std::endl;
        return true;
    }
    
    std::pair<cv::Mat, CaptureMetrics> captureDesktopDuplication() {
        CaptureMetrics metrics;
        cv::Mat frame;
        
        if (!dx_initialized) {
            std::cerr << "Desktop Duplication not initialized" << std::endl;
            return {frame, metrics};
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        try {
            DXGI_OUTDUPL_FRAME_INFO frame_info;
            ComPtr<IDXGIResource> desktop_resource;
            
            auto capture_start = std::chrono::high_resolution_clock::now();
            HRESULT hr = desktop_duplication->AcquireNextFrame(0, &frame_info, &desktop_resource);
            
            if (hr == DXGI_ERROR_WAIT_TIMEOUT) {
                return {frame, metrics};
            } else if (FAILED(hr)) {
                return {frame, metrics};
            }
            
            auto capture_end = std::chrono::high_resolution_clock::now();
            metrics.capture_ms = std::chrono::duration_cast<std::chrono::microseconds>(capture_end - capture_start).count() / 1000.0f;
            
            ComPtr<ID3D11Texture2D> acquired_texture;
            hr = desktop_resource.As(&acquired_texture);
            if (FAILED(hr)) {
                desktop_duplication->ReleaseFrame();
                return {frame, metrics};
            }
            
            auto copy_start = std::chrono::high_resolution_clock::now();
            
            if (capture_x == 0 && capture_y == 0) {
                d3d_context->CopyResource(staging_texture.Get(), acquired_texture.Get());
            } else {
                D3D11_BOX source_box;
                source_box.left = capture_x;
                source_box.top = capture_y;
                source_box.right = capture_x + capture_width;
                source_box.bottom = capture_y + capture_height;
                source_box.front = 0;
                source_box.back = 1;
                
                d3d_context->CopySubresourceRegion(
                    staging_texture.Get(), 0, 0, 0, 0,
                    acquired_texture.Get(), 0, &source_box);
            }
            
            D3D11_MAPPED_SUBRESOURCE mapped_resource;
            hr = d3d_context->Map(staging_texture.Get(), 0, D3D11_MAP_READ, 0, &mapped_resource);
            if (FAILED(hr)) {
                desktop_duplication->ReleaseFrame();
                return {frame, metrics};
            }
            
            frame = cv::Mat(capture_height, capture_width, CV_8UC4, mapped_resource.pData, mapped_resource.RowPitch);
            
            cv::Mat bgr_frame;
            cv::cvtColor(frame, bgr_frame, cv::COLOR_BGRA2BGR);
            frame = bgr_frame.clone();
            
            d3d_context->Unmap(staging_texture.Get(), 0);
            desktop_duplication->ReleaseFrame();
            
            auto copy_end = std::chrono::high_resolution_clock::now();
            metrics.copy_ms = std::chrono::duration_cast<std::chrono::microseconds>(copy_end - copy_start).count() / 1000.0f;
            
        } catch (const std::exception& e) {
            std::cerr << "Exception in Desktop Duplication capture: " << e.what() << std::endl;
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        metrics.total_ms = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000.0f;
        
        updateFPS(metrics);
        return {frame, metrics};
    }
    
    std::pair<cv::Mat, CaptureMetrics> captureGDI() {
        CaptureMetrics metrics;
        cv::Mat frame;
        
        if (!gdi_initialized) {
            std::cerr << "GDI not initialized" << std::endl;
            return {frame, metrics};
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        HGDIOBJ old_bitmap = SelectObject(memory_dc, bitmap);
        
        auto capture_start = std::chrono::high_resolution_clock::now();
        BOOL result = BitBlt(memory_dc, 0, 0, capture_width, capture_height,
                           screen_dc, capture_x, capture_y, SRCCOPY);
        auto capture_end = std::chrono::high_resolution_clock::now();
        metrics.capture_ms = std::chrono::duration_cast<std::chrono::microseconds>(capture_end - capture_start).count() / 1000.0f;
        
        if (result) {
            auto copy_start = std::chrono::high_resolution_clock::now();
            
            std::vector<uint8_t> buffer(capture_width * capture_height * 4);
            int lines = GetDIBits(memory_dc, bitmap, 0, capture_height,
                                buffer.data(), &bitmap_info, DIB_RGB_COLORS);
            
            if (lines > 0) {
                cv::Mat temp_frame(capture_height, capture_width, CV_8UC4, buffer.data());
                cv::cvtColor(temp_frame, frame, cv::COLOR_BGRA2BGR);
            }
            
            auto copy_end = std::chrono::high_resolution_clock::now();
            metrics.copy_ms = std::chrono::duration_cast<std::chrono::microseconds>(copy_end - copy_start).count() / 1000.0f;
        }
        
        SelectObject(memory_dc, old_bitmap);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        metrics.total_ms = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000.0f;
        
        updateFPS(metrics);
        return {frame, metrics};
    }
    
    std::pair<cv::Mat, CaptureMetrics> capture() {
        if (dx_initialized) {
            return captureDesktopDuplication();
        } else if (gdi_initialized) {
            return captureGDI();
        } else {
            CaptureMetrics metrics;
            return {cv::Mat(), metrics};
        }
    }
    
    // FIXED: Proper client area calculation to exclude window decorations
    bool initializeForCS2Window() {
        // Find CS2 window with multiple possible names
        std::vector<std::string> possible_names = {
            "Counter-Strike 2",
            "cs2",
            "CS2"
        };
        
        for (const auto& name : possible_names) {
            cs2_window = FindWindowA(nullptr, name.c_str());
            if (cs2_window) {
                std::cout << "Found CS2 window: " << name << std::endl;
                break;
            }
        }
        
        if (!cs2_window) {
            std::cerr << "CS2 window not found. Make sure CS2 is running." << std::endl;
            return false;
        }
        
        // Get the actual client area (game content only, no decorations)
        RECT client_rect;
        if (!GetClientRect(cs2_window, &client_rect)) {
            std::cerr << "Failed to get CS2 client rect" << std::endl;
            return false;
        }
        
        // Convert client coordinates to screen coordinates
        POINT top_left = {client_rect.left, client_rect.top};
        POINT bottom_right = {client_rect.right, client_rect.bottom};
        
        if (!ClientToScreen(cs2_window, &top_left) || !ClientToScreen(cs2_window, &bottom_right)) {
            std::cerr << "Failed to convert client coordinates to screen coordinates" << std::endl;
            return false;
        }
        
        // Calculate actual client dimensions
        int client_width = bottom_right.x - top_left.x;
        int client_height = bottom_right.y - top_left.y;
        
        std::cout << "CS2 client area: " << client_width << "x" << client_height 
                  << " at screen position (" << top_left.x << "," << top_left.y << ")" << std::endl;
        
        // Store window info for validation
        last_window_rect = {top_left.x, top_left.y, bottom_right.x, bottom_right.y};
        window_info_valid = true;
        
        // Check if window is minimized or has invalid dimensions
        if (client_width <= 0 || client_height <= 0) {
            std::cerr << "CS2 window appears to be minimized or has invalid dimensions" << std::endl;
            return false;
        }
        
        // Try Desktop Duplication first
        if (initializeDesktopDuplication(top_left.x, top_left.y, client_width, client_height)) {
            return true;
        } else {
            std::cout << "Desktop Duplication failed, trying GDI..." << std::endl;
            return initializeGDI(top_left.x, top_left.y, client_width, client_height);
        }
    }
    
    // Helper function to validate if CS2 window is still valid
    bool validateCS2Window() {
        if (!cs2_window || !window_info_valid) {
            return false;
        }
        
        // Check if window still exists
        if (!IsWindow(cs2_window)) {
            std::cout << "CS2 window no longer exists" << std::endl;
            return false;
        }
        
        // Check if window is minimized
        if (IsIconic(cs2_window)) {
            std::cout << "CS2 window is minimized" << std::endl;
            return false;
        }
        
        // Optionally check if window position/size changed significantly
        RECT current_rect;
        if (GetClientRect(cs2_window, &current_rect)) {
            POINT top_left = {current_rect.left, current_rect.top};
            POINT bottom_right = {current_rect.right, current_rect.bottom};
            ClientToScreen(cs2_window, &top_left);
            ClientToScreen(cs2_window, &bottom_right);
            
            // Check if position changed by more than 50 pixels (window moved)
            if (abs(top_left.x - last_window_rect.left) > 50 || 
                abs(top_left.y - last_window_rect.top) > 50) {
                std::cout << "CS2 window position changed significantly, reinitializing..." << std::endl;
                return false;
            }
        }
        
        return true;
    }
    
    void cleanup() {
        if (desktop_duplication) desktop_duplication.Reset();
        if (staging_texture) staging_texture.Reset();
        if (d3d_context) d3d_context.Reset();
        if (d3d_device) d3d_device.Reset();
        
        if (bitmap) DeleteObject(bitmap);
        if (memory_dc) DeleteDC(memory_dc);
        if (screen_dc) ReleaseDC(nullptr, screen_dc);
        
        dx_initialized = false;
        gdi_initialized = false;
        window_info_valid = false;
        cs2_window = nullptr;
    }
    
private:
    void updateFPS(CaptureMetrics& metrics) {
        frame_count++;
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_fps_time);
        
        if (elapsed.count() >= 1000) {
            current_fps = frame_count;
            frame_count = 0;
            last_fps_time = now;
        }
        
        metrics.fps = current_fps;
    }
};

// FIXED: Display management to avoid window conflicts
class CS2RealTimeDetector {
    bool save_next_frame = false;
private:
    YOLOv8CS2 detector;
    ScreenCapture screen_capture;
    bool running = false;
    
    // Performance tracking
    int total_frames = 0;
    int detected_frames = 0;
    std::chrono::high_resolution_clock::time_point session_start;
    
    // Display settings
    bool display_enabled = false;
    std::string window_name = "CS2 Detection";
    
public:
    CS2RealTimeDetector(const std::string& model_path, bool use_gpu = true) 
        : detector(model_path, use_gpu) {
        
        if (!screen_capture.initializeForCS2Window()) {
            std::cerr << "Failed to initialize screen capture for CS2" << std::endl;
            throw std::runtime_error("Screen capture initialization failed");
        }
        
        detector.warmup(3);
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
        }

        bool paused = false;
        auto last_stats_time = std::chrono::high_resolution_clock::now();
        int window_check_counter = 0;

        // Initialize display window if needed
        if (show_display) {
            setupDisplayWindow();
        }

        while (running) {
            if (!paused) {
                // Periodically check if CS2 window is still valid
                if (++window_check_counter % 100 == 0) {
                    if (!screen_capture.validateCS2Window()) {
                        std::cout << "CS2 window validation failed, attempting to reinitialize..." << std::endl;
                        if (!screen_capture.initializeForCS2Window()) {
                            std::cout << "Failed to reinitialize screen capture" << std::endl;
                            std::this_thread::sleep_for(std::chrono::seconds(1));
                            continue;
                        }
                    }
                }

                // Capture screen
                auto capture_result = screen_capture.capture();
                cv::Mat frame = capture_result.first;
                auto capture_metrics = capture_result.second;

                if (frame.empty()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    continue;
                }

                total_frames++;

                // Run detection
                auto detection_result = detector.detectWithMetrics(frame, conf_threshold, nms_threshold);
                auto detections = detection_result.first;
                auto detection_metrics = detection_result.second;

                // Count detections by class
                std::vector<int> class_counts(4, 0);
                for (const auto& det : detections) {
                    if (det.class_id >= 0 && det.class_id < 4) {
                        class_counts[det.class_id]++;
                    }
                }

                if (!detections.empty()) {
                    detected_frames++;
                    detector.drawResults(frame, detections);
                }

                // Save screenshot if requested
                if (save_next_frame) {
                    auto now = std::chrono::system_clock::now();
                    auto timestamp = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
                    std::string screenshot_filename = "cs2_screenshot_" + std::to_string(timestamp) + ".jpg";

                    cv::imwrite(screenshot_filename, frame);
                    std::cout << "\n[SAVED] Screenshot saved as: " << screenshot_filename << std::endl;

                    // Also save detection data if there are any detections
                    if (!detections.empty()) {
                        std::string json_filename = "cs2_detections_" + std::to_string(timestamp) + ".json";
                        exportToJSON(detections, json_filename);
                    }

                    save_next_frame = false;
                }

                // Add performance overlay
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
                    handleInput(paused, save_results, save_results);
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
    
    void stop() {
        running = false;
    }
    
private:
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
    void setupDisplayWindow() {
        try {
            cv::namedWindow(window_name, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
            
            // Position window away from game area (bottom-right corner)
            int screen_width = GetSystemMetrics(SM_CXSCREEN);
            int screen_height = GetSystemMetrics(SM_CYSCREEN);
            int window_width = 800;
            int window_height = 600;
            
            cv::moveWindow(window_name, screen_width - window_width - 50, screen_height - window_height - 100);
            cv::resizeWindow(window_name, window_width, window_height);
            
            // Set window to not be always on top to avoid interference
            cv::setWindowProperty(window_name, cv::WND_PROP_TOPMOST, 0);
            
            display_enabled = true;
            std::cout << "Display window created and positioned away from game area" << std::endl;
            
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
        
        if (key == 'q' || key == 'Q' || key == 27) { // 'q', 'Q' or ESC
            running = false;
        } else if (key == 'p' || key == 'P') {
            paused = !paused;
            std::cout << (paused ? "\n[PAUSED] - Press P to resume" : "\n[RESUMED]") << std::endl;
        } else if (key == 's' || key == 'S') {
            if (save_results) {
                save_next_frame = true;
                std::cout << "\n[SAVE] - Screenshot will be saved on next frame" << std::endl;
            }
        } else if (key == 'd' || key == 'D') {
            display_enabled = !display_enabled;
            if (!display_enabled) {
                cv::destroyAllWindows();
                std::cout << "\n[DISPLAY OFF] - Press D to turn back on" << std::endl;
            } else {
                setupDisplayWindow();
                std::cout << "\n[DISPLAY ON]" << std::endl;
            }
        }
    }
    
    void addPerformanceOverlay(cv::Mat& frame, const ScreenCapture::CaptureMetrics& capture_metrics,
                              const YOLOv8CS2::PerformanceMetrics& detection_metrics,
                              int detection_count, const std::vector<int>& class_counts) {
        
        // Create semi-transparent overlay
        cv::Mat overlay = frame.clone();
        cv::rectangle(overlay, cv::Point(10, 10), cv::Point(450, 180), cv::Scalar(0, 0, 0), -1);
        cv::addWeighted(frame, 0.75, overlay, 0.25, 0, frame);
        
        // Performance metrics
        cv::putText(frame, "CS2 REAL-TIME DETECTION", 
                   cv::Point(20, 35), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
        
        cv::putText(frame, "FPS: " + std::to_string(capture_metrics.fps), 
                   cv::Point(20, 60), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
        
        cv::putText(frame, "Latency: " + std::to_string(capture_metrics.total_ms + detection_metrics.total_ms).substr(0, 4) + "ms",
                   cv::Point(20, 85), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        
        cv::putText(frame, "Players: " + std::to_string(detection_count),
                   cv::Point(20, 110), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);
        
        // Class breakdown (compact)
        if (detection_count > 0) {
            std::string class_breakdown = "CT:" + std::to_string(class_counts[0]) + 
                                        " T:" + std::to_string(class_counts[2]) + 
                                        " +H:" + std::to_string(class_counts[1] + class_counts[3]);
            cv::putText(frame, class_breakdown,
                       cv::Point(20, 135), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);
        }
        
        // Performance indicator
        float total_latency = capture_metrics.total_ms + detection_metrics.total_ms;
        cv::Scalar perf_color = total_latency < 20 ? cv::Scalar(0, 255, 0) : 
                               total_latency < 33 ? cv::Scalar(0, 255, 255) : cv::Scalar(0, 0, 255);
        cv::circle(frame, cv::Point(400, 35), 10, perf_color, -1);
        
        // Controls hint (smaller)
        cv::putText(frame, "Q=Quit P=Pause D=Display S=Save",
                   cv::Point(20, 160), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(150, 150, 150), 1);
    }
    
    void printLiveStats(const ScreenCapture::CaptureMetrics& capture_metrics,
                       const YOLOv8CS2::PerformanceMetrics& detection_metrics,
                       int current_detections) {
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - session_start);
        
        float detection_rate = total_frames > 0 ? (detected_frames * 100.0f / total_frames) : 0.0f;
        float avg_fps = elapsed.count() > 0 ? (total_frames / static_cast<float>(elapsed.count())) : 0.0f;
        
        std::cout << "\n=== LIVE STATS (T+" << elapsed.count() << "s) ===" << std::endl;
        std::cout << "Current FPS: " << capture_metrics.fps << " | Avg FPS: " << std::fixed << std::setprecision(1) << avg_fps << std::endl;
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
#endif

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
                std::cout << "CS2 Player Detection - Usage:" << std::endl;
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
        
        std::cout << "=== CS2 PLAYER DETECTION SYSTEM ===" << std::endl;
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
            
            YOLOv8CS2 detector(model_path, true);
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
            
            YOLOv8CS2 detector(model_path, true);
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
                
                // Count by class
                std::vector<int> class_counts(4, 0);
                for (const auto& det : detections) {
                    class_counts[det.class_id]++;
                    std::cout << "  " << det.class_name << " (conf: " 
                              << std::fixed << std::setprecision(3) << det.confidence 
                              << ") at [" << det.bbox.x << "," << det.bbox.y 
                              << "," << det.bbox.width << "," << det.bbox.height << "]" << std::endl;
                }
                
                std::cout << "\nClass breakdown:" << std::endl;
                std::vector<std::string> class_names = {"CT", "CT+Helmet", "T", "T+Helmet"};
                for (int i = 0; i < 4; i++) {
                    if (class_counts[i] > 0) {
                        std::cout << "  " << class_names[i] << ": " << class_counts[i] << std::endl;
                    }
                }
                
            } else {
                std::cout << "\nNo CS2 players detected" << std::endl;
                std::cout << "Try lowering the confidence threshold with --conf 0.1" << std::endl;
            }
            
            // Save results
            std::string output_path = "cs2_result_" + 
                std::to_string(std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count()) + ".jpg";
            cv::imwrite(output_path, result_image);
            std::cout << "\nResult saved as: " << output_path << std::endl;
            
            if (!detections.empty()) {
                exportToJSON(detections, "cs2_detections.json");
            }
            
            // Performance analysis
            std::cout << "\n=== PERFORMANCE ANALYSIS ===" << std::endl;
            if (metrics.inference_ms < 20) {
                std::cout << "EXCELLENT: GPU acceleration working perfectly!" << std::endl;
            } else if (metrics.inference_ms < 50) {
                std::cout << "GOOD: GPU acceleration working" << std::endl;
            } else {
                std::cout << "WARNING: Performance suboptimal - check GPU drivers" << std::endl;
            }
            
            // Display
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