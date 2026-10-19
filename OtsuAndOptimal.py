
import cv2
import numpy as np
import matplotlib.pyplot as plt

class OtsuAndOptimal:
    def __init__(self, image_path):
        # Acepta una ruta (comportamiento original) o un array NumPy ya cargado.
        if isinstance(image_path, np.ndarray):
            img = image_path
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            self.image = img.astype(np.uint8)
        else:
            self.image = self.read_image(image_path)

    def optimal_thresholding(self, initial_threshold=128, tolerance=0.25):
        gray_image = self.image
        threshold = initial_threshold if initial_threshold is not None else gray_image.mean()
        previous_threshold = 0

        while abs(threshold - previous_threshold) > tolerance:
            foreground = gray_image[gray_image >= threshold]
            background = gray_image[gray_image < threshold]

            mean_f = foreground.mean() if foreground.size > 0 else 0
            mean_b = background.mean() if background.size > 0 else 0

            previous_threshold = threshold
            threshold = (mean_f + mean_b) / 2

        binary_result = (gray_image >= threshold).astype(np.uint8) * 255
        return binary_result

    def otsu_thresholding(self):
        gray_image = self.image
        histogram, _ = np.histogram(gray_image.ravel(), bins=256, range=(0, 256))
        total_pixels = gray_image.size
        sum_all = np.dot(np.arange(256), histogram)

        max_variance = 0
        threshold = 0
        background_w = 0
        background_sum = 0

        for t in range(256):
            background_w += histogram[t]
            if background_w == 0:
                continue

            foreground_w = total_pixels - background_w
            if foreground_w == 0:
                break

            background_sum += t * histogram[t]

            mean_b = background_sum / background_w
            mean_f = (sum_all - background_sum) / foreground_w

            variance_between = (
                background_w * foreground_w * (mean_b - mean_f) ** 2
            )

            if variance_between > max_variance:
                max_variance = variance_between
                threshold = t

        binary_result = (gray_image >= threshold).astype(np.uint8) * 255
        return binary_result
    
    def read_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image from {image_path}")

        # Convert to grayscale if image is colored
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return img
    
    def local_otsu(self, window_size):
        gray_image = self.image
        h, w = gray_image.shape
        result = np.zeros_like(gray_image)

        for i in range(0, h, window_size):
            for j in range(0, w, window_size):
                window = gray_image[i:i+window_size, j:j+window_size]

                # Antes: cv2.imwrite('temp_window.png') + releer del disco por cada ventana.
                # Ahora: se procesa la ventana en memoria (mismo resultado, mucho más rápido).
                local_thresholder = OtsuAndOptimal(window)
                local_result = local_thresholder.otsu_thresholding()

                result[i:i+window_size, j:j+window_size] = local_result

        return result

    
    def local_optimal(self, window_size):
        gray_image = self.image
        h, w = gray_image.shape
        result = np.zeros_like(gray_image)

        for i in range(0, h, window_size):
            for j in range(0, w, window_size):
                window = gray_image[i:i+window_size, j:j+window_size]

                # Procesamiento en memoria (ver local_otsu).
                local_thresholder = OtsuAndOptimal(window)
                local_result = local_thresholder.optimal_thresholding()

                result[i:i+window_size, j:j+window_size] = local_result

        return result

    
    def test_thresholding(self):
        # Apply both methods
        optimal_result = self.optimal_thresholding()
        otsu_result = self.otsu_thresholding()

        # Plot all results
        plt.figure(figsize=(15,5))

        plt.subplot(1,3,1)
        plt.imshow(self.image, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1,3,2)
        plt.imshow(optimal_result, cmap='gray')
        plt.title('Optimal Thresholding')
        plt.axis('off')

        plt.subplot(1,3,3)
        plt.imshow(otsu_result, cmap='gray')
        plt.title('Otsu\'s Thresholding')
        plt.axis('off')

        plt.tight_layout()
        plt.show()
        
    