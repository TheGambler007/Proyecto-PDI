"""División y fusión de regiones (split & merge) con quadtree.

1. División: se parte la imagen en cuadrantes hasta que cada bloque sea
   homogéneo (desvío estándar <= sigma_max) o alcance el tamaño mínimo.
2. Fusión: se unen hojas homogéneas vecinas cuya frontera es "débil"
   (diferencia media de intensidad a través del borde <= umbral_fusion).
   Comparar la frontera y no las medias de los bloques permite fusionar
   un fondo con iluminación en rampa sin cruzar los bordes del objeto.
3. Limpieza: los bloques no homogéneos (los que cruzan un borde) y las
   regiones muy chicas se absorben en el vecino con la frontera más débil.

Aporte propio: sigma_max y umbral_fusion se calculan a partir del ruido
estimado de la imagen (Immerkær, 1996), así el algoritmo no necesita
ajustarse a mano para cada escenario.

Uso:
    python -m aporte.split_merge ruta/imagen.jpg
"""
import argparse
import os

import cv2
import numpy as np

KERNEL_RUIDO = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float64)


def estimar_ruido(imagen):
    """Estimación rápida del desvío del ruido gaussiano (Immerkær, 1996).
    El kernel es una segunda derivada: ignora bordes suaves y rampas de luz."""
    g = imagen.astype(np.float64)
    h, w = g.shape
    suma = np.abs(cv2.filter2D(g, -1, KERNEL_RUIDO)[1:-1, 1:-1]).sum()
    return float(suma * np.sqrt(np.pi / 2) / (6 * (w - 2) * (h - 2)))


def dividir(imagen, sigma_max, tam_min=4, tam_max=16):
    """Fase de división. Devuelve (etiquetas_hojas, homogenea[bool por hoja]).

    tam_max fuerza a dividir bloques grandes aunque parezcan homogéneos: en un
    bloque enorme, una "astilla" de objeto casi no mueve el desvío estándar."""
    h, w = imagen.shape
    g = imagen.astype(np.float64)
    etiquetas = np.zeros((h, w), np.int32)
    homogenea = []
    pila = [(0, 0, h, w)]
    while pila:
        y, x, alto, ancho = pila.pop()
        bloque = g[y:y + alto, x:x + ancho]
        es_homogeneo = bloque.std() <= sigma_max
        if alto <= tam_min or ancho <= tam_min or (es_homogeneo and max(alto, ancho) <= tam_max):
            etiquetas[y:y + alto, x:x + ancho] = len(homogenea)
            homogenea.append(bool(es_homogeneo))
            continue
        a2, w2 = alto // 2, ancho // 2
        pila += [(y, x, a2, w2), (y, x + w2, a2, ancho - w2),
                 (y + a2, x, alto - a2, w2), (y + a2, x + w2, alto - a2, ancho - w2)]
    return etiquetas, np.array(homogenea)


def _fronteras(etiquetas, imagen):
    """Para cada par de regiones vecinas (a < b): diferencia media con signo
    de intensidad a través de su frontera (b - a). El promedio con signo
    cancela el ruido, a diferencia del promedio del valor absoluto."""
    g = imagen.astype(np.float64)
    a = np.concatenate([etiquetas[:, :-1].ravel(), etiquetas[:-1, :].ravel()])
    b = np.concatenate([etiquetas[:, 1:].ravel(), etiquetas[1:, :].ravel()])
    d = np.concatenate([(g[:, 1:] - g[:, :-1]).ravel(), (g[1:, :] - g[:-1, :]).ravel()])
    distintos = a != b
    a, b, d = a[distintos], b[distintos], d[distintos]
    invertir = a > b
    a, b, d = np.where(invertir, b, a), np.where(invertir, a, b), np.where(invertir, -d, d)
    n = int(etiquetas.max()) + 1
    claves, inversa = np.unique(a.astype(np.int64) * n + b, return_inverse=True)
    sumas = np.bincount(inversa, weights=d)
    cuentas = np.bincount(inversa)
    return claves // n, claves % n, np.abs(sumas / cuentas)


class _UnionFind:
    def __init__(self, n):
        self.padre = np.arange(n)

    def raiz(self, i):
        while self.padre[i] != i:
            self.padre[i] = self.padre[self.padre[i]]
            i = self.padre[i]
        return i

    def unir(self, i, j):
        ri, rj = self.raiz(i), self.raiz(j)
        if ri != rj:
            self.padre[rj] = ri

    def raices(self):
        return np.array([self.raiz(i) for i in range(len(self.padre))])


def fusionar(imagen, hojas, homogenea, umbral_fusion, area_min):
    """Fase de fusión + limpieza. Devuelve etiquetas de región consecutivas."""
    uf = _UnionFind(len(homogenea))
    medias = np.bincount(hojas.ravel(), weights=imagen.ravel().astype(np.float64)) / np.bincount(hojas.ravel())

    # Fusión: hojas homogéneas con frontera débil (de la más débil a la más fuerte)
    # y medias compatibles. Lo segundo evita fusionar a través de una hoja que
    # tiene una astilla del otro lado justo en la frontera.
    a, b, dif = _fronteras(hojas, imagen)
    for i in np.argsort(dif):
        if dif[i] > umbral_fusion:
            break
        if (homogenea[a[i]] and homogenea[b[i]]
                and abs(medias[a[i]] - medias[b[i]]) <= 2 * umbral_fusion):
            uf.unir(a[i], b[i])

    # Limpieza: absorber hojas de borde y regiones chicas en el vecino más parecido.
    pendiente_hoja = ~homogenea
    for _ in range(50):
        raiz = uf.raices()
        regiones = raiz[hojas]
        area = np.bincount(regiones.ravel(), minlength=len(raiz))
        pendiente = np.zeros(len(raiz), bool)
        pendiente[raiz[pendiente_hoja]] = True
        pendiente |= (area > 0) & (area < area_min)
        if not pendiente.any():
            break
        a, b, dif = _fronteras(regiones, imagen)
        mejor = {}
        for ra, rb, d in zip(a, b, dif):
            for origen, destino in ((ra, rb), (rb, ra)):
                if pendiente[origen] and not pendiente[destino]:
                    if origen not in mejor or d < mejor[origen][1]:
                        mejor[origen] = (destino, d)
        if not mejor:  # solo quedan pendientes rodeados de pendientes
            for ra, rb in zip(a, b):
                uf.unir(rb, ra)
            break
        for origen, (destino, _) in mejor.items():
            uf.unir(destino, origen)
            pendiente_hoja[raiz == origen] = False

    _, etiquetas = np.unique(uf.raices()[hojas], return_inverse=True)
    return etiquetas.reshape(hojas.shape).astype(np.int32)


def mascara_objeto(etiquetas):
    """Fondo = la región más grande que toca el borde de la imagen; el resto es objeto."""
    borde = np.concatenate([etiquetas[0], etiquetas[-1], etiquetas[:, 0], etiquetas[:, -1]])
    candidatas = np.unique(borde)
    areas = np.bincount(etiquetas.ravel())
    fondo = candidatas[np.argmax(areas[candidatas])]
    return ((etiquetas != fondo) * 255).astype(np.uint8)


def split_and_merge(imagen, sigma_max=None, umbral_fusion=None, tam_min=4, area_min=None,
                    devolver_detalle=False):
    """Segmentación completa. Sin parámetros, todo se ajusta al ruido estimado."""
    if imagen.ndim == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    ruido_original = ruido = estimar_ruido(imagen)
    # Con ruido fuerte, un bloque que cruza un borde puede pasar como "homogéneo"
    # y filtrar la fusión hacia el objeto. La mediana baja el ruido sin borrar bordes.
    # Tras una mediana 5x5, el ruido gaussiano baja aprox. a sigma·sqrt(pi / (2·25)).
    # (No se re-estima con Immerkær: la mediana deja ruido correlacionado y lo subestima.)
    pasadas_mediana = 0
    while ruido > 8 and pasadas_mediana < 3:
        imagen = cv2.medianBlur(imagen, 5)
        ruido *= np.sqrt(np.pi / 50)
        pasadas_mediana += 1
    sigma_max = 1.25 * ruido + 2 if sigma_max is None else sigma_max
    umbral_fusion = max(8.0, 2.0 * ruido) if umbral_fusion is None else umbral_fusion
    area_min = max(16, imagen.size // 1000) if area_min is None else area_min

    hojas, homogenea = dividir(imagen, sigma_max, tam_min)
    etiquetas = fusionar(imagen, hojas, homogenea, umbral_fusion, area_min)
    mascara = mascara_objeto(etiquetas)
    if not devolver_detalle:
        return mascara
    return mascara, {
        "hojas": hojas, "etiquetas": etiquetas, "ruido_estimado": ruido_original,
        "pasadas_mediana": pasadas_mediana,
        "sigma_max": sigma_max, "umbral_fusion": umbral_fusion,
        "n_hojas": len(homogenea), "n_regiones": int(etiquetas.max()) + 1,
    }


def dibujar_quadtree(imagen, hojas):
    """Superpone los límites de las hojas del quadtree sobre la imagen."""
    salida = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
    limite = np.zeros(hojas.shape, bool)
    limite[:, 1:] |= hojas[:, 1:] != hojas[:, :-1]
    limite[1:, :] |= hojas[1:, :] != hojas[:-1, :]
    salida[limite] = (0, 200, 255)
    return salida


def main():
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("imagen")
    parser.add_argument("--sigma-max", type=float, default=None,
                        help="umbral de homogeneidad (por defecto, automático según el ruido). "
                             "En imágenes reales con textura probá 10-15.")
    parser.add_argument("--umbral-fusion", type=float, default=None,
                        help="umbral de fusión (por defecto, automático según el ruido)")
    parser.add_argument("--salida", default=os.path.join("aporte", "resultados"))
    args = parser.parse_args()

    imagen = cv2.imread(args.imagen, cv2.IMREAD_GRAYSCALE)
    if imagen is None:
        raise SystemExit(f"No se pudo leer {args.imagen}")
    mascara, det = split_and_merge(imagen, sigma_max=args.sigma_max, umbral_fusion=args.umbral_fusion,
                                   devolver_detalle=True)
    print(f"Ruido estimado: {det['ruido_estimado']:.1f} | sigma_max: {det['sigma_max']:.1f} | "
          f"umbral_fusion: {det['umbral_fusion']:.1f} | hojas: {det['n_hojas']} | "
          f"regiones finales: {det['n_regiones']}")

    fig, ejes = plt.subplots(1, 4, figsize=(18, 5))
    paneles = [("Original", imagen, "gray"),
               (f"División: {det['n_hojas']} hojas", dibujar_quadtree(imagen, det["hojas"])[..., ::-1], None),
               (f"Fusión: {det['n_regiones']} regiones", det["etiquetas"], "nipy_spectral"),
               ("Máscara final", mascara, "gray")]
    for eje, (titulo, img, cmap) in zip(ejes, paneles):
        eje.imshow(img, cmap=cmap)
        eje.set_title(titulo)
        eje.axis("off")
    fig.tight_layout()
    os.makedirs(args.salida, exist_ok=True)
    destino = os.path.join(args.salida, "split_merge.png")
    fig.savefig(destino, dpi=120)
    print(f"Figura guardada en {destino}")
    if os.environ.get("MPLBACKEND", "").lower() != "agg":
        plt.show()


if __name__ == "__main__":
    main()
