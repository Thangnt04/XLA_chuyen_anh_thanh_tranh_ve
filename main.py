import tkinter as tk
from app.ui import ImageEditorApp 
import sys
import os

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = ImageEditorApp(root)
        root.mainloop()
        
    except ImportError as e:
        print(f"LỖI IMPORT: {e}", file=sys.stderr)
        print("Hãy chắc chắn bạn đã ở trong môi trường (.venv) và đã chạy:", file=sys.stderr)
        print("pip install numpy scikit-image Pillow", file=sys.stderr)
    except Exception as e:
        print(f"Đã xảy ra lỗi không xác định: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()