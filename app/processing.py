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
    Áp dụng bộ lọc Gaussian (làm mịn) bằng hàm tích chập.
    Đây là "kỹ thuật làm mịn" mà đề bài yêu cầu.
    """
    kernel = _create_gaussian_kernel(size, sigma)
    return convolution_2d(image, kernel)

def sobel_edge_detection(gray_image):
    """
    Phát hiện biên Sobel bằng hàm tích chập.
    """
    # 1. Định nghĩa kernel Sobel
    kernel_x = np.array([[-1, 0, 1], 
                         [-2, 0, 2], 
                         [-1, 0, 1]], dtype=np.float64)
    
    kernel_y = np.array([[-1, -2, -1], 
                         [ 0,  0,  0], 
                         [ 1,  2,  1]], dtype=np.float64)
    # 2. Áp dụng tích chập 
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
    """
    # 1. Làm mịn ảnh bằng Gaussian 
    blurred_image = gaussian_blur(gray_image, sigma=sigma)
    
    # 2. Áp dụng Laplacian lên ảnh đã làm mịn
    lap_kernel = np.array([[ 0,  1,  0],
                           [ 1, -4,  1],
                           [ 0,  1,  0]], dtype=np.float64)
    
    log_image = convolution_2d(blurred_image, lap_kernel)
    
    # Chuẩn hóa về [0, 1]
    if log_image.max() > 0:
        log_image = log_image / log_image.max()

    return np.clip(log_image, 0, 1)

# --- CÁC BỘ LỌC NÂNG CAO ---

def bilateral_filter(image, diameter=5, sigma_color=0.1, sigma_space=1.5):
    """Apply a bilateral filter implemented purely with NumPy."""
    if diameter < 1:
        raise ValueError("diameter must be >= 1")
    if sigma_color <= 0 or sigma_space <= 0:
        raise ValueError("sigma_color and sigma_space must be > 0")
    if diameter % 2 == 0:
        diameter += 1  # ensure odd kernel size
    radius = diameter // 2
    image = image.astype(float)
    if image.ndim == 2:
        return _bilateral_single_channel(image, radius, sigma_color, sigma_space)
    if image.ndim == 3:
        channels = [
            _bilateral_single_channel(image[..., ch], radius, sigma_color, sigma_space)
            for ch in range(image.shape[2])
        ]
        return np.stack(channels, axis=-1)
    raise ValueError("Bilateral filter expects a 2D or 3D array")


def _bilateral_single_channel(channel, radius, sigma_color, sigma_space):
    """Bilateral filter helper for a single channel."""
    diameter = 2 * radius + 1
    padded = np.pad(channel, radius, mode='reflect')
    ax = np.arange(-radius, radius + 1, dtype=float)
    xx, yy = np.meshgrid(ax, ax)
    spatial_kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma_space ** 2))
    output = np.zeros_like(channel)
    for y in range(channel.shape[0]):
        for x in range(channel.shape[1]):
            patch = padded[y : y + diameter, x : x + diameter]
            center_val = channel[y, x]
            range_kernel = np.exp(-((patch - center_val) ** 2) / (2.0 * sigma_color ** 2))
            weights = spatial_kernel * range_kernel
            weights_sum = np.sum(weights)
            if weights_sum == 0:
                output[y, x] = center_val
            else:
                output[y, x] = np.sum(patch * weights) / weights_sum
    return output


def edge_preserving_filter(image, num_iterations=10, kappa=30.0, gamma=0.2, option=1):
    """Edge-preserving smoothing via anisotropic diffusion."""
    if not (0 < gamma <= 0.25):
        raise ValueError("gamma must lie in (0, 0.25]")
    if num_iterations < 1:
        raise ValueError("num_iterations must be >= 1")
    image = image.astype(float)
    if image.ndim == 2:
        return _anisotropic_diffusion(image, num_iterations, kappa, gamma, option)
    if image.ndim == 3:
        channels = [
            _anisotropic_diffusion(image[..., ch], num_iterations, kappa, gamma, option)
            for ch in range(image.shape[2])
        ]
        return np.stack(channels, axis=-1)
    raise ValueError("Edge preserving filter expects a 2D or 3D array")


def _anisotropic_diffusion(channel, num_iterations, kappa, gamma, option):
    """Perona-Malik anisotropic diffusion core implementation."""
    diffused = channel.copy()
    for _ in range(num_iterations):
        padded = np.pad(diffused, 1, mode='edge')
        center = padded[1:-1, 1:-1]
        north = padded[:-2, 1:-1] - center
        south = padded[2:, 1:-1] - center
        east = padded[1:-1, 2:] - center
        west = padded[1:-1, :-2] - center
        if option == 1:
            c_n = np.exp(-(north / kappa) ** 2)
            c_s = np.exp(-(south / kappa) ** 2)
            c_e = np.exp(-(east / kappa) ** 2)
            c_w = np.exp(-(west / kappa) ** 2)
        elif option == 2:
            c_n = 1.0 / (1.0 + (north / kappa) ** 2)
            c_s = 1.0 / (1.0 + (south / kappa) ** 2)
            c_e = 1.0 / (1.0 + (east / kappa) ** 2)
            c_w = 1.0 / (1.0 + (west / kappa) ** 2)
        else:
            raise ValueError("option must be 1 or 2")
        diffused += gamma * (c_n * north + c_s * south + c_e * east + c_w * west)
    return np.clip(diffused, 0.0, 1.0)