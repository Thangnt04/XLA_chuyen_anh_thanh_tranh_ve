# app/processing.py

import numpy as np

def convert_to_gray(rgb_image):
    """
    Chuyển ảnh RGB (H, W, 3) sang ảnh xám (H, W).
    Đây là triển khai THAY THẾ cho skimage.color.rgb2gray
    
    Công thức: Gray = 0.299*R + 0.587*G + 0.114*B
    """
    if rgb_image.ndim == 2:
        return rgb_image  # Ảnh đã là ảnh xám

    if rgb_image.shape[2] == 4:
        # Xử lý ảnh PNG có kênh Alpha (bỏ kênh A)
        rgb_image = rgb_image[..., :3]

    # Dùng phép nhân ma trận (dot product) của NumPy để tính toán nhanh
    weights = np.array([0.299, 0.587, 0.114])
    gray_image = np.dot(rgb_image[..., :3], weights)
    
    return gray_image

def convolution_2d(image, kernel):
    """
    Hàm TÍCH CHẬP (convolution) 2D cơ bản.
    Đây là hàm CỐT LÕI thay thế cho mọi hàm filter() có sẵn.
    """
    k_height, k_width = kernel.shape
    i_height, i_width = image.shape
    
    # Tính toán padding (số pixel cần thêm vào viền)
    pad_h = k_height // 2
    pad_w = k_width // 2
    
    # Tạo ảnh mới (toàn số 0) với padding ở viền
    padded_image = np.zeros((i_height + 2 * pad_h, i_width + 2 * pad_w))
    padded_image[pad_h:pad_h + i_height, pad_w:pad_w + i_width] = image
    
    # Tạo ảnh đầu ra (toàn số 0) với kích thước GỐC
    output_image = np.zeros_like(image)
    
    # Di chuyển kernel qua TỪNG pixel của ảnh GỐC
    for y in range(i_height):
        for x in range(i_width):
            # Lấy vùng ảnh (Region of Interest) từ ảnh đã padding
            roi = padded_image[y : y + k_height, x : x + k_width]
            
            # Tính toán tích chập: nhân từng phần tử rồi tính tổng
            output_pixel = np.sum(roi * kernel)
            
            # Gán giá trị vào ảnh đầu ra
            output_image[y, x] = output_pixel
            
    return output_image

def _create_gaussian_kernel(size=5, sigma=1.0):
    """(Hàm phụ trợ) Tạo một kernel Gaussian (bộ lọc làm mịn)"""
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return kernel / np.sum(kernel) # Chuẩn hóa

def gaussian_blur(image, size=5, sigma=1.0):
    """
    Áp dụng bộ lọc Gaussian (làm mịn) bằng hàm tích chập tự viết.
    Đây là "kỹ thuật làm mịn" mà đề bài yêu cầu.
    """
    kernel = _create_gaussian_kernel(size, sigma)
    return convolution_2d(image, kernel)

def sobel_edge_detection(gray_image):
    """
    Phát hiện biên Sobel bằng hàm tích chập tự viết.
    Thay thế cho: filters.sobel_h và filters.sobel_v
    """
    # 1. Định nghĩa kernel Sobel
    kernel_x = np.array([[-1, 0, 1], 
                         [-2, 0, 2], 
                         [-1, 0, 1]], dtype=np.float64)
    
    kernel_y = np.array([[-1, -2, -1], 
                         [ 0,  0,  0], 
                         [ 1,  2,  1]], dtype=np.float64)
    
    # 2. Áp dụng tích chập (dùng hàm tự viết)
    g_x = convolution_2d(gray_image, kernel_x)
    g_y = convolution_2d(gray_image, kernel_y)
    
    # 3. Tính độ lớn gradient: G = sqrt(Gx^2 + Gy^2)
    g_magnitude = np.hypot(g_x, g_y)
    
    # 4. Chuẩn hóa kết quả về [0, 1]
    if g_magnitude.max() > 0:
        g_magnitude = g_magnitude / g_magnitude.max()
        
    return g_magnitude

def apply_laplacian_filter(gray_image):
    """
    Áp dụng bộ lọc Laplacian (làm sắc nét).
    Thay thế cho: filters.laplace
    """
    # Kernel Laplacian 4 lân cận
    lap_kernel = np.array([[ 0,  1,  0],
                           [ 1, -4,  1],
                           [ 0,  1,  0]], dtype=np.float64)
    
    lap_image = convolution_2d(gray_image, lap_kernel)
    
    # Cộng ảnh gốc và ảnh sau khi áp dụng Laplacian
    sketch = np.clip(gray_image + lap_image, 0, 1)
    return sketch

def log_edge_detection(gray_image, sigma=1.0):
    """
    Áp dụng bộ lọc Laplacian of Gaussian (LoG).
    Thay thế cho: ndimage.gaussian_laplace
    """
    # 1. Làm mịn ảnh bằng Gaussian (TỰ VIẾT)
    blurred_image = gaussian_blur(gray_image, sigma=sigma)
    
    # 2. Áp dụng Laplacian (TỰ VIẾT) lên ảnh đã làm mịn
    lap_kernel = np.array([[ 0,  1,  0],
                           [ 1, -4,  1],
                           [ 0,  1,  0]], dtype=np.float64)
    
    log_image = convolution_2d(blurred_image, lap_kernel)
    
    # Chuẩn hóa về [0, 1]
    if log_image.max() > 0:
        log_image = log_image / log_image.max()

    return np.clip(log_image, 0, 1)

# --- CÁC BỘ LỌC NÂNG CAO ---
# Đây là các hàm giữ chỗ. Tự viết các bộ lọc này rất phức tạp.
# Chúng ta sẽ chỉ in cảnh báo và trả về ảnh gốc để không làm hỏng chương trình.

def bilateral_filter(image):
    """Hàm giữ chỗ cho Bilateral Filter."""
    print("CẢNH BÁO: Tự implement Bilateral Filter RẤT PHỨC TẠP và chưa được làm.")
    print("Yêu cầu của đề bài là TỰ VIẾT, không phải gọi 'cv2.bilateralFilter()'.")
    print("Tạm thời trả về ảnh gốc.")
    return image 

def edge_preserving_filter(image):
    """Hàm giữ chỗ cho Edge Preserving Filter."""
    print("CẢNH BÁO: Tự implement Edge Preserving Filter RẤT PHỨC TẠP và chưa được làm.")
    print("Tạm thời trả về ảnh gốc.")
    return image