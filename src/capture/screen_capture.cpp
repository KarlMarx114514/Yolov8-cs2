#include "screen_capture.hpp"
#include <iostream>
#include <vector>
#include <string>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "winmm.lib")

namespace cs2_detection {

void CaptureMetrics::print() const {
    std::cout << "Capture: " << capture_ms << "ms, Copy: " << copy_ms 
              << "ms, Total: " << total_ms << "ms, FPS: " << fps << std::endl;
}

void WindowsScreenCapture::updateFPS(CaptureMetrics& metrics) {
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

// DXScreenCapture Implementation
DXScreenCapture::DXScreenCapture() : WindowsScreenCapture() {
    last_fps_time = std::chrono::high_resolution_clock::now();
}

DXScreenCapture::~DXScreenCapture() {
    cleanup();
}

bool DXScreenCapture::initialize(int x, int y, int width, int height) {
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

bool DXScreenCapture::initializeForCS2Window() {
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
    
    return initialize(top_left.x, top_left.y, client_width, client_height);
}

CaptureResult DXScreenCapture::capture() {
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

bool DXScreenCapture::validateCS2Window() {
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
    
    return true;
}

void DXScreenCapture::cleanup() {
    if (desktop_duplication) desktop_duplication.Reset();
    if (staging_texture) staging_texture.Reset();
    if (d3d_context) d3d_context.Reset();
    if (d3d_device) d3d_device.Reset();
    
    dx_initialized = false;
    window_info_valid = false;
    cs2_window = nullptr;
}

// GDIScreenCapture Implementation
GDIScreenCapture::GDIScreenCapture() : WindowsScreenCapture() {
    last_fps_time = std::chrono::high_resolution_clock::now();
}

GDIScreenCapture::~GDIScreenCapture() {
    cleanup();
}

bool GDIScreenCapture::initialize(int x, int y, int width, int height) {
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

bool GDIScreenCapture::initializeForCS2Window() {
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
    
    return initialize(top_left.x, top_left.y, client_width, client_height);
}

CaptureResult GDIScreenCapture::capture() {
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

bool GDIScreenCapture::validateCS2Window() {
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
    
    return true;
}

void GDIScreenCapture::cleanup() {
    if (bitmap) DeleteObject(bitmap);
    if (memory_dc) DeleteDC(memory_dc);
    if (screen_dc) ReleaseDC(nullptr, screen_dc);
    
    gdi_initialized = false;
    window_info_valid = false;
    cs2_window = nullptr;
}

// ScreenCaptureFactory Implementation
std::unique_ptr<WindowsScreenCapture> ScreenCaptureFactory::create(CaptureMethod method) {
    if (method == CaptureMethod::AUTO) {
        // Try DirectX first, fall back to GDI
        auto dx_capture = std::make_unique<DXScreenCapture>();
        if (dx_capture) {
            std::cout << "Auto-selected DirectX Desktop Duplication" << std::endl;
            return std::move(dx_capture);
        } else {
            std::cout << "DirectX failed, falling back to GDI" << std::endl;
            return std::make_unique<GDIScreenCapture>();
        }
    } else if (method == CaptureMethod::DESKTOP_DUPLICATION) {
        return std::make_unique<DXScreenCapture>();
    } else if (method == CaptureMethod::GDI) {
        return std::make_unique<GDIScreenCapture>();
    }
    
    return nullptr;
}

std::vector<CaptureMethod> ScreenCaptureFactory::getAvailableMethods() {
    std::vector<CaptureMethod> methods;
    
    // DirectX is always available on Windows 8+
    methods.push_back(CaptureMethod::DESKTOP_DUPLICATION);
    
    // GDI is always available on Windows
    methods.push_back(CaptureMethod::GDI);
    
    return methods;
}

const char* ScreenCaptureFactory::methodToString(CaptureMethod method) {
    switch (method) {
        case CaptureMethod::AUTO: return "Auto";
        case CaptureMethod::DESKTOP_DUPLICATION: return "DirectX Desktop Duplication";
        case CaptureMethod::GDI: return "GDI";
        default: return "Unknown";
    }
}

} // namespace cs2_detection