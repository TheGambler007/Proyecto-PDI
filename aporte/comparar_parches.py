"""Demuestra las dos mejoras hechas al código original del repositorio:

1. OtsuAndOptimal.local_otsu / local_optimal: antes escribían y releían un PNG
   en disco por cada ventana; ahora procesan la ventana en memoria.
2. region_growing: antes recorría píxel por píxel con una cola en Python;
   ahora calcula la región como componente conexa (vectorizado con OpenCV).

Para cada mejora verifica que el resultado sea IDÉNTICO al original y mide
la aceleración. Las versiones originales están copiadas textualmente abajo
(commit 7ef53e7 del repositorio).

Uso (desde la raíz del repo):
    python -m aporte.comparar_parches
"""
import os
import tempfile
import time

import cv2
import numpy as np

import RegionGrowingSegmentation as modulo_rg
from OtsuAndOptimal import OtsuAndOptimal


# ------------------------------------------------- versiones originales
def local_otsu_original(imagen, ventana):
    """Versión original: escribe cada ventana a disco y la relee."""
    h, w = imagen.shape
    resultado = np.zeros_like(imagen)
    ruta = os.path.join(tempfile.gettempdir(), "temp_window.png")
    for i in range(0, h, ventana):
        for j in range(0, w, ventana):
            cv2.imwrite(ruta, imagen[i:i + ventana, j:j + ventana])
            resultado[i:i + ventana, j:j + ventana] = OtsuAndOptimal(ruta).otsu_thresholding()
    return resultado


# Copia textual de region_growing original (solo se quitó la conversión desde QImage).
def region_growing_original(image, seeds, threshold=50, use_luv=True):
    """
    Perform region growing segmentation with optional CIELUV color space support

    Args:
        image: NumPy array of image data (RGB or grayscale)
        seeds: List of (x, y) tuples as starting points
        threshold: Distance threshold for region inclusion
        use_luv: Boolean to use CIELUV color space (default: True)

    Returns:
        NumPy array of binary segmentation mask
    """
    # convert qimage to cv2 image
    
    # Handle color space conversion
    if len(image.shape) == 3 and use_luv:
        # Convert BGR to CIELUV for better perceptual color distance
        # Note: OpenCV expects BGR order by default
        luv_image = modulo_rg.rgb_to_luv_vectorized(image)
        processed_image = luv_image
    elif len(image.shape) == 3 and not use_luv:
        # Keep original RGB image
        processed_image = image.copy()
    else:
        # For grayscale images
        processed_image = image.copy()
        use_luv = False  # Force use_luv to False for grayscale

    # Get dimensions based on original image
    if len(image.shape) == 3:
        height, width, _ = processed_image.shape
    else:
        height, width = processed_image.shape

    # Create segmentation mask for all seeds
    overall_segmentation = np.zeros((height, width), dtype=np.uint8)

    # Process each seed point separately
    for seed_idx, (seed_x, seed_y) in enumerate(seeds, 1):
        # Skip if out of bounds
        if 0 <= seed_x < width and 0 <= seed_y < height:
            visited = np.zeros((height, width), dtype=bool)
            seed_segment = np.zeros((height, width), dtype=np.uint8)

            # Get seed pixel value
            if len(processed_image.shape) == 3:
                seed_value = processed_image[seed_y, seed_x, :].astype(np.float32)
            else:
                seed_value = processed_image[seed_y, seed_x]

            # Initialize queue with seed
            processing_queue = [(seed_x, seed_y)]
            visited[seed_y, seed_x] = True
            seed_segment[seed_y, seed_x] = 1

            # Define 8-connected neighbors
            neighbors = [
                (-1, -1), (-1, 0), (-1, 1),
                (0, -1), (0, 1),
                (1, -1), (1, 0), (1, 1)
            ]

            # Process queue
            while processing_queue:
                x, y = processing_queue.pop(0)

                for dx, dy in neighbors:
                    nx, ny = x + dx, y + dy

                    # Check bounds
                    if 0 <= nx < width and 0 <= ny < height:
                        # Skip if already visited
                        if not visited[ny, nx]:
                            visited[ny, nx] = True

                            # Calculate color difference based on color space
                            if len(processed_image.shape) == 3:
                                pixel_value = processed_image[ny, nx, :].astype(np.float32)
                                if use_luv:
                                    # For CIELUV, use Euclidean distance in LUV space
                                    color_diff = np.sqrt(np.sum((pixel_value - seed_value) ** 2))
                                else:
                                    # For RGB, use Euclidean distance
                                    color_diff = np.sqrt(np.sum((pixel_value - seed_value) ** 2))
                            else:
                                # For grayscale
                                pixel_value = processed_image[ny, nx]
                                color_diff = abs(float(pixel_value) - float(seed_value))

                            # Check if pixel is within threshold
                            if color_diff <= threshold:
                                seed_segment[ny, nx] = 1
                                processing_queue.append((nx, ny))

        # Add this segment to the final segmentation
        overall_segmentation = np.maximum(overall_segmentation, seed_segment * seed_idx)

    return overall_segmentation


# ------------------------------------------------------------- medición
def medir(funcion, repeticiones=3):
    tiempos = []
    for _ in range(repeticiones):
        t0 = time.perf_counter()
        resultado = funcion()
        tiempos.append(time.perf_counter() - t0)
    return resultado, float(np.median(tiempos)) * 1000


def comparar(titulo, original, nuevo, repeticiones=3):
    r_orig, t_orig = medir(original, repeticiones)
    r_nuevo, t_nuevo = medir(nuevo, repeticiones)
    print(titulo)
    print(f"   original: {t_orig:9.1f} ms")
    print(f"   aporte:   {t_nuevo:9.1f} ms   -> {t_orig / t_nuevo:.0f}x más rápido")
    print(f"   resultado idéntico: {np.array_equal(r_orig, r_nuevo)}")
    return r_nuevo


def main():
    gris = cv2.imread(os.path.join("threshold_images", "brain_mri.jpeg"), cv2.IMREAD_GRAYSCALE)
    color = cv2.imread(os.path.join("segment_images", "cow-toy.jpeg"))
    print(f"Imágenes: brain_mri.jpeg {gris.shape[1]}x{gris.shape[0]} (gris), "
          f"cow-toy.jpeg {color.shape[1]}x{color.shape[0]} (color)\n")

    comparar("1) Otsu local, ventana de 16 px (gris)",
             lambda: local_otsu_original(gris, 16),
             lambda: OtsuAndOptimal(gris).local_otsu(16))

    semillas = [(5, 5), (220, 200)]
    r = comparar("\n2a) Crecimiento de regiones, 2 semillas (gris)",
                 lambda: region_growing_original(gris, semillas, threshold=20, use_luv=False),
                 lambda: modulo_rg.region_growing(gris, semillas, threshold=20, use_luv=False), 1)
    print(f"   píxeles segmentados: {int((r > 0).sum())}")

    h, w = color.shape[:2]
    semillas = [(w // 2, h // 2)]
    for luv in (False, True):
        comparar(f"\n2b) Crecimiento de regiones (color, {'CIELUV' if luv else 'RGB'})",
                 lambda: region_growing_original(color, semillas, threshold=40, use_luv=luv),
                 lambda: modulo_rg.region_growing(color, semillas, threshold=40, use_luv=luv), 1)


if __name__ == "__main__":
    main()
