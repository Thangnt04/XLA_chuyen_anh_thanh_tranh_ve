# app/utils.py

import numpy as np
from skimage.io import imread, imsave
from skimage import img_as_float # Dùng để chuyển ảnh 0-255 về 0-1
from PIL import Image, ImageTk, ImageOps

def load_image_file(image_path):
    """Đọc file ảnh bằng skimage và chuyển về dạng float [0, 1]"""
    try:
        # Đọc ảnh, img_as_float sẽ tự động chuẩn hóa về [0, 1]
        img = img_as_float(imread(image_path))
        return img
    except Exception as e:
        print(f"Lỗi khi đọc ảnh: {e}")
        return None

def save_image_file(file_path, np_image):
    """Lưu ảnh (dạng mảng numpy) ra file"""
    try:
        save_image = np_image.copy()
        
        # Chuyển từ [0, 1] về [0, 255] kiểu uint8 để lưu
        if save_image.max() <= 1.0:
            save_image = (save_image * 255).astype(np.uint8)
        else:
            save_image = save_image.astype(np.uint8)
            
        imsave(file_path, save_image)
        return True
    except Exception as e:
        print(f"Lỗi khi lưu ảnh: {e}")
        return False

def enhance_sketch_lines(sketch_image):
    """Tăng độ đậm và tương phản của các nét vẽ (từ code gốc)"""
    # Đảm bảo ảnh đang ở [0, 1]
    if sketch_image.max() > 1:
        sketch_image = sketch_image.astype(np.float32) / 255.0
    
    # Dùng Gamma correction
    sketch_image = np.power(sketch_image, 0.6)
    # Tăng độ đậm
    sketch_image = sketch_image * 1.3
    # Cắt giá trị về lại [0, 1]
    sketch_image = np.clip(sketch_image, 0, 1)
    
    return sketch_image

def invert_sketch(sketch_image):
    """Đảo ngược ảnh (nét trắng -> đen, nền đen -> trắng)"""
    # Ảnh [0, 1]: 0 là đen, 1 là trắng
    # Đảo ngược: 1.0 - giá trị
    if sketch_image.max() <= 1.0:
        return 1.0 - sketch_image
    else:
        # Nếu ảnh đang là [0, 255]
        return 255 - sketch_image

def numpy_to_tkinter_image(np_image, max_size=(800, 600)):
    """Chuyển mảng numpy sang ảnh có thể hiển thị trên Tkinter"""
    
    # Chuyển ảnh [0, 1] về [0, 255] và kiểu uint8
    if np_image.max() <= 1.0:
        np_image = (np_image * 255).astype(np.uint8)
    else:
        np_image = np_image.astype(np.uint8)
    
    if len(np_image.shape) == 2:
        # Ảnh xám
        pil_image = Image.fromarray(np_image, mode='L')
    elif len(np_image.shape) == 3:
        # Ảnh màu
        pil_image = Image.fromarray(np_image, mode='RGB')
    else:
        raise ValueError("Định dạng ảnh numpy không hỗ trợ.")

    # Resize ảnh để vừa với cửa sổ (dùng PIL, không dùng cv2)
    pil_image.thumbnail(max_size, Image.Resampling.LANCZOS)

    return ImageTk.PhotoImage(pil_image)