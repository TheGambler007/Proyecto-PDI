"""Detección de discontinuidades: puntos, líneas, bordes y enfoque combinado.

Uso:
    python -m aporte.discontinuidades                 # imagen de prueba sintética
    python -m aporte.discontinuidades ruta/imagen.jpg # cualquier imagen
"""
import argparse
import os

import cv2
import numpy as np
from scipy import ndimage

# ---------- Máscaras clásicas (Gonzalez & Woods, cap. 10) ----------
MASCARA_PUNTO = np.array([[-1, -1, -1],
                          [-1,  8, -1],
                          [-1, -1, -1]], np.float32)

# Las diagonales se nombran por cómo se VEN en pantalla (eje y hacia abajo),
# para evitar la confusión de signos entre convenciones de ejes.
MASCARAS_LINEA = {
    "horizontal": np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], np.float32),
    "vertical": np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]], np.float32),
    "diagonal \\": np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]], np.float32),
    "diagonal /": np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]], np.float32),
}


def _a_gris(imagen):
    if imagen.ndim == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    return imagen.astype(np.float32)


def _normalizar(respuesta):
    return cv2.normalize(np.abs(respuesta), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def detectar_puntos(imagen, fraccion=0.9):
    """Punto aislado: |R| >= T, con T = fracción del máximo de |R| (Laplaciano)."""
    r = np.abs(cv2.filter2D(_a_gris(imagen), -1, MASCARA_PUNTO))
    return ((r >= fraccion * r.max()) * 255).astype(np.uint8)


def detectar_lineas(imagen, direccion="horizontal", fraccion=0.5):
    """Líneas de 1 px en una dirección (respuesta positiva de la máscara).

    El umbral es relativo al máximo de las CUATRO máscaras: si en la imagen no
    hay líneas en esa dirección, no aparece nada (en vez de umbralizar ruido)."""
    g = _a_gris(imagen)
    respuestas = {k: np.maximum(cv2.filter2D(g, -1, m), 0) for k, m in MASCARAS_LINEA.items()}
    maximo = max(r.max() for r in respuestas.values())
    if maximo == 0:
        return np.zeros(g.shape, np.uint8)
    return ((respuestas[direccion] >= fraccion * maximo) * 255).astype(np.uint8)


def bordes_sobel(imagen):
    g = _a_gris(imagen)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return _normalizar(np.hypot(gx, gy))


def bordes_prewitt(imagen):
    g = _a_gris(imagen)
    kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], np.float32)
    gx = cv2.filter2D(g, -1, kx)
    gy = cv2.filter2D(g, -1, kx.T)
    return _normalizar(np.hypot(gx, gy))


def bordes_log(imagen, sigma=2.0):
    """Marr-Hildreth: Laplaciano del Gaussiano + cruces por cero."""
    log = ndimage.gaussian_laplace(_a_gris(imagen), sigma)
    umbral = 0.1 * np.abs(log).max()
    cruces = np.zeros(log.shape, bool)
    for dy, dx in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        vecino = np.roll(np.roll(log, dy, 0), dx, 1)
        cruces |= (np.sign(log) != np.sign(vecino)) & (np.abs(log - vecino) > umbral)
    return (cruces * 255).astype(np.uint8)


def bordes_canny(imagen, bajo=50, alto=150):
    g = cv2.GaussianBlur(_a_gris(imagen).astype(np.uint8), (5, 5), 1.4)
    return cv2.Canny(g, bajo, alto)


def _rellenar(bordes):
    """Rellena los contornos cerrados (flood fill desde el borde de la imagen)."""
    return (ndimage.binary_fill_holes(bordes > 0) * 255).astype(np.uint8)


def segmentar_por_bordes(imagen, sigma=1.5):
    """Enfoque combinado: suavizado → gradiente (Sobel) → umbral de Otsu sobre
    la magnitud → cierre morfológico → relleno de contornos.

    Devuelve una máscara binaria del objeto, comparable con las demás técnicas.
    """
    g = cv2.GaussianBlur(_a_gris(imagen), (0, 0), sigma)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    magnitud = _normalizar(np.hypot(gx, gy))
    _, bordes = cv2.threshold(magnitud, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bordes = cv2.morphologyEx(bordes, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mascara = _rellenar(bordes)
    # Abrir para eliminar restos finos de borde que no encierran nada.
    return cv2.morphologyEx(mascara, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))


def imagen_de_prueba(tamano=256):
    """Fondo gris con puntos aislados, líneas en 4 direcciones y una figura."""
    img = np.full((tamano, tamano), 60, np.uint8)
    cv2.rectangle(img, (150, 150), (230, 230), 170, -1)
    for p in [(40, 40), (90, 200), (200, 60)]:
        img[p[1], p[0]] = 255
    cv2.line(img, (20, 120), (120, 120), 200, 1)
    cv2.line(img, (130, 20), (130, 110), 200, 1)
    cv2.line(img, (20, 140), (100, 220), 200, 1)
    rng = np.random.default_rng(0)
    return np.clip(img + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)


def main():
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("imagen", nargs="?", help="ruta a una imagen (opcional)")
    parser.add_argument("--salida", default=os.path.join("aporte", "resultados"))
    args = parser.parse_args()

    imagen = cv2.imread(args.imagen, cv2.IMREAD_GRAYSCALE) if args.imagen else imagen_de_prueba()
    if imagen is None:
        raise SystemExit(f"No se pudo leer {args.imagen}")

    paneles = [
        ("Original", imagen),
        ("2a. Puntos (Laplaciano)", detectar_puntos(imagen)),
        ("2b. Líneas horizontales", detectar_lineas(imagen, "horizontal")),
        ("2b. Líneas verticales", detectar_lineas(imagen, "vertical")),
        ("2b. Líneas diagonales \\", detectar_lineas(imagen, "diagonal \\")),
        ("2c. Sobel", bordes_sobel(imagen)),
        ("2c. Prewitt", bordes_prewitt(imagen)),
        ("2c. LoG (Marr-Hildreth)", bordes_log(imagen)),
        ("2c. Canny", bordes_canny(imagen)),
        ("2d. Combinada → máscara", segmentar_por_bordes(imagen)),
    ]
    fig, ejes = plt.subplots(2, 5, figsize=(18, 8.2), layout="constrained")
    for eje, (titulo, img) in zip(ejes.ravel(), paneles):
        eje.imshow(img, cmap="gray")
        eje.set_title(titulo, fontsize=10)
        eje.axis("off")
    fig.suptitle("Detección de discontinuidades", fontsize=14)
    os.makedirs(args.salida, exist_ok=True)
    destino = os.path.join(args.salida, "discontinuidades.png")
    fig.savefig(destino, dpi=120)
    print(f"Figura guardada en {destino}")
    if os.environ.get("MPLBACKEND", "").lower() != "agg":
        plt.show()


if __name__ == "__main__":
    main()
