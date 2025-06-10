#pragma once
#include <opencv2/opencv.hpp>
#include <memory>
#include <chrono>
#include <windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <wrl/client.h>

using Microsoft::WRL::ComPtr;

namespace cs2_detection {

struct CaptureMetrics {
    float capture_ms = 0.0f;
    float copy_ms = 0.0f;
    float total_ms = 0.0f;
    int fps = 0;
    
    void print() const;
};

using CaptureResult = std::pair<cv::Mat, CaptureMetrics>;

// Base class for Windows screen capture
class WindowsScreenCapture {
public:
    virtual ~WindowsScreenCapture() = default;
    
    virtual bool initialize(int x = 0, int y = 0, int width = 0, int height = 0) = 0;
    virtual bool initializeForCS2Window() = 0;
    virtual CaptureResult capture() = 0;
    virtual bool validateCS2Window() = 0;
    virtual void cleanup() = 0;
    
protected:
    int capture_width = 0;
    int capture_height = 0;
    int capture_x = 0;
    int capture_y = 0;
    HWND cs2_window = nullptr;
    RECT last_window_rect = {0};
    bool window_info_valid = false;
    
    // FPS tracking
    std::chrono::high_resolution_clock::time_point last_fps_time;
    int frame_count = 0;
    int current_fps = 0;
    
    void updateFPS(CaptureMetrics& metrics);
};

// DirectX Desktop Duplication implementation
class DXScreenCapture : public WindowsScreenCapture {
public:
    DXScreenCapture();
    ~DXScreenCapture();
    
    bool initialize(int x = 0, int y = 0, int width = 0, int height = 0) override;
    bool initializeForCS2Window() override;
    CaptureResult capture() override;
    bool validateCS2Window() override;
    void cleanup() override;
    
private:
    ComPtr<ID3D11Device> d3d_device;
    ComPtr<ID3D11DeviceContext> d3d_context;
    ComPtr<IDXGIOutputDuplication> desktop_duplication;
    ComPtr<ID3D11Texture2D> staging_texture;
    
    bool dx_initialized = false;
};

// GDI implementation
class GDIScreenCapture : public WindowsScreenCapture {
public:
    GDIScreenCapture();
    ~GDIScreenCapture();
    
    bool initialize(int x = 0, int y = 0, int width = 0, int height = 0) override;
    bool initializeForCS2Window() override;
    CaptureResult capture() override;
    bool validateCS2Window() override;
    void cleanup() override;
    
private:
    HDC screen_dc = nullptr;
    HDC memory_dc = nullptr;
    HBITMAP bitmap = nullptr;
    BITMAPINFO bitmap_info;
    
    bool gdi_initialized = false;
};

enum class CaptureMethod {
    AUTO,
    DESKTOP_DUPLICATION,  // DirectX 11
    GDI                   // Windows GDI
};

class ScreenCaptureFactory {
public:
    static std::unique_ptr<WindowsScreenCapture> create(CaptureMethod method = CaptureMethod::AUTO);
    static std::vector<CaptureMethod> getAvailableMethods();
    static const char* methodToString(CaptureMethod method);
};

} // namespace cs2_detection