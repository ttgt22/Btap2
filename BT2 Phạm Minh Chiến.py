import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh từ file

image = cv2.imread("C:/New folder/hyhy.jpg", cv2.IMREAD_GRAYSCALE)


# Kiểm tra xem ảnh có được đọc thành công hay không
if image is None:
    print("Không thể mở hoặc đọc tệp ảnh. Vui lòng kiểm tra lại đường dẫn.")
else:
    # Định nghĩa các toán tử Sobel
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # Gx
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # Gy

    # Áp dụng toán tử Sobel bằng cách sử dụng hàm cv2.filter2D (tích chập 2D)
    grad_x = cv2.filter2D(image, cv2.CV_64F, sobel_x)  # Biên theo hướng X
    grad_y = cv2.filter2D(image, cv2.CV_64F, sobel_y)  # Biên theo hướng Y

    # Tính toán độ lớn của gradient từ Gx và Gy
    sobel_combined = cv2.magnitude(grad_x, grad_y)

    # Hiển thị kết quả
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.title("Ảnh gốc")
    plt.imshow(image, cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title("Sobel Gx")
    plt.imshow(grad_x, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title("Sobel Gy")
    plt.imshow(grad_y, cmap='gray')

    plt.tight_layout()
    plt.show()
