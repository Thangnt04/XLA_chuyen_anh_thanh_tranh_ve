import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk 

# Import các module đã tách riêng
import app.processing as proc
import app.utils as utils

class ImageEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("XLA - Chuyển Ảnh Thành Tranh (Phiên bản tinh gọn)")
        self.root.geometry("1200x700")
        
        # Các biến lưu trữ ảnh
        self.original_rgb_image = None # Ảnh gốc (đã load, dạng float 0-1)
        self.gray_image = None       # Ảnh xám (đã xử lý, dạng float 0-1)
        self.processed_image = None  # Ảnh kết quả (dạng float 0-1)
        
        # Các biến hiển thị của Tkinter
        self.original_photo = None
        self.processed_photo = None

        # Biến điều khiển bộ lọc nâng cao (chỉ bật / tắt với thông số mặc định)
        self.use_bilateral = tk.BooleanVar(value=False)
        self.use_edge_preserving = tk.BooleanVar(value=False)
        
        self.create_widgets()
    
    def create_widgets(self):
        # KHUNG ĐIỀU KHIỂN
        control_frame = tk.Frame(self.root, bg='#f0f0f0', padx=10, pady=10)
        control_frame.pack(fill=tk.X)
        
        btn_choose = tk.Button(control_frame, text="Chọn Ảnh", command=self.choose_image,
                                bg='#4CAF50', fg='white', font=('Arial', 12, 'bold'),
                                padx=20, pady=10, cursor='hand2')
        btn_choose.pack(side=tk.LEFT, padx=5)
        
        # CÁC NÚT BẤM CÒN LẠI
        btn_process = tk.Button(control_frame, text="Xử Lý Ảnh", command=self.process_image,
                                bg='#2196F3', fg='white', font=('Arial', 12, 'bold'),
                                padx=20, pady=10, cursor='hand2')
        btn_process.pack(side=tk.LEFT, padx=5)
        
        btn_save = tk.Button(control_frame, text="Lưu Ảnh", command=self.save_image,
                                bg='#FF9800', fg='white', font=('Arial', 12, 'bold'),
                                padx=20, pady=10, cursor='hand2')
        btn_save.pack(side=tk.LEFT, padx=5)

        # Khung chứa các checkbox bộ lọc nâng cao
        advanced_frame = tk.Frame(control_frame, bg='#f0f0f0')
        advanced_frame.pack(side=tk.LEFT, padx=15)
        self._build_filter_controls(advanced_frame)
        
        # KHUNG HIỂN THỊ ẢNH
        image_frame = tk.Frame(self.root)
        image_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        original_frame = tk.LabelFrame(image_frame, text="Ảnh Gốc", font=('Arial', 12, 'bold'))
        original_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.original_label = tk.Label(original_frame, text="Chưa có ảnh", bg='white',
                                        font=('Arial', 14), fg='gray')
        self.original_label.pack(fill=tk.BOTH, expand=True)
        
        processed_frame = tk.LabelFrame(image_frame, text="Ảnh Đã Xử Lý", font=('Arial', 12, 'bold'))
        processed_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.processed_label = tk.Label(processed_frame, text="Chưa xử lý", bg='white',
                                        font=('Arial', 14), fg='gray')
        self.processed_label.pack(fill=tk.BOTH, expand=True)

    def _build_filter_controls(self, parent):
        """Tạo giao diện gọn nhẹ cho bộ lọc nâng cao."""
        ttk.Label(parent, text="Bộ lọc nâng cao").pack(anchor='w')
        ttk.Checkbutton(
            parent,
            text="Bật Bilateral Filter ",   #mặc định: kernel 5px, sigmaC=0.1, sigmaS=1.5
            variable=self.use_bilateral,
        ).pack(anchor='w', pady=2)
        ttk.Checkbutton(
            parent,
            text="Bật Edge-Preserving Filter ", #(10 vòng, κ=30, γ=0.2, option=1)
            variable=self.use_edge_preserving,
        ).pack(anchor='w', pady=2)

    def choose_image(self):
        """Hàm chọn ảnh, gọi hàm load từ utils.py"""
        image_path = filedialog.askopenfilename(
            title="Chọn ảnh",
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.tif")]
        )
        if not image_path:
            return

        
        # SỬ DỤNG HÀM TỪ UTILS
        self.original_rgb_image = utils.load_image_file(image_path)
        if self.original_rgb_image is None:
            messagebox.showerror("Lỗi", "Không thể đọc ảnh.")
            return

        # SỬ DỤNG HÀM TỪ PROCESSING
        # Chuyển sang ảnh xám ngay khi tải
        self.gray_image = proc.convert_to_gray(self.original_rgb_image)
        
        # SỬ DỤNG HÀM TỪ UTILS
        # Hiển thị ảnh gốc (ảnh màu)
        self.original_photo = utils.numpy_to_tkinter_image(self.original_rgb_image)
        self.original_label.config(image=self.original_photo, text='')
        
        # Xóa ảnh kết quả cũ
        self.processed_image = None
        self.processed_label.config(image='', text="Chưa xử lý", bg='white')

    def process_image(self):
        """Hàm xử lý ảnh, gọi các hàm từ processing.py"""
        if self.gray_image is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn ảnh trước!")
            return
        
        try:
            
            working_gray = self.gray_image

            # Áp dụng Bilateral Filter nếu checkbox được chọn
            if self.use_bilateral.get():
                working_gray = proc.bilateral_filter(
                    working_gray,
                    diameter=5,
                    sigma_color=0.1,
                    sigma_space=1.5,
                )

            # Áp dụng Edge-Preserving Filter nếu checkbox được chọn
            if self.use_edge_preserving.get():
                working_gray = proc.edge_preserving_filter(
                    working_gray,
                    num_iterations=10,
                    kappa=30.0,
                    gamma=0.2,
                    option=1,
                )

            #  Làm mịn 
            blurred_gray_image = proc.gaussian_blur(working_gray, size=5, sigma=1.0)
            
            # Phát hiện biên bằng Sobel 
            sketch = proc.sobel_edge_detection(blurred_gray_image)

            # GỌI CÁC HÀM TỪ UTILS 
            sketch = utils.enhance_sketch_lines(sketch)

            # Đảo ngược màu
            sketch = utils.invert_sketch(sketch)

            self.processed_image = sketch
            
            # --- GỌI HÀM TỪ UTILS (Hiển thị) ---
            self.processed_photo = utils.numpy_to_tkinter_image(self.processed_image)
            self.processed_label.config(image=self.processed_photo, text='')
            
            messagebox.showinfo("Thành công", "Ảnh đã được xử lý thành công!")
        
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể xử lý ảnh: {str(e)}")
            import traceback
            traceback.print_exc() # In lỗi chi tiết ra console

    def save_image(self):
        """Hàm lưu ảnh, gọi hàm save từ utils.py"""
        if self.processed_image is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng xử lý ảnh trước khi lưu!")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Lưu ảnh",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        
        if file_path:
            # SỬ DỤNG HÀM TỪ UTILS
            success = utils.save_image_file(file_path, self.processed_image)
            if success:
                messagebox.showinfo("Thành công", f"Ảnh đã được lưu tại:\n{file_path}")
            else:
                messagebox.showerror("Lỗi", "Không thể lưu ảnh.")