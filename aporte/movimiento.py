"""Uso del movimiento en la segmentación.

Técnicas espaciales (método básico, Gonzalez & Woods 10.6.1):
  - Diferencia de cuadros consecutivos: |f(t) - f(t-1)| > T
  - Imágenes de diferencias acumuladas (ADI) absoluta, positiva y negativa
    respecto de un cuadro de referencia.
  - Sustracción de fondo por mediana temporal y MOG2 (OpenCV, enfoque moderno).

Dominio de la frecuencia:
  - Correlación de fase (teorema del desplazamiento de Fourier): estima la
    velocidad del objeto en px/cuadro a partir del espectro cruzado normalizado.

Uso:
    python -m aporte.movimiento                  # secuencia sintética + métricas
    python -m aporte.movimiento --camara 0       # demo en vivo con la webcam
    python -m aporte.movimiento --video clip.mp4 # demo con un video
"""
import argparse
import csv
import os

import cv2
import numpy as np

from aporte.metricas import evaluar


# ---------------------------------------------------------------- secuencia
def generar_secuencia(n_cuadros=40, n_vacios=30, tamano=256, velocidad=(3, 2), lado=36,
                      sigma_ruido=4, semilla=0):
    """Cámara fija: primero n_vacios cuadros solo con fondo texturado (como una
    cámara que ya estaba filmando) y luego un cuadrado que entra y se mueve a
    velocidad constante."""
    rng = np.random.default_rng(semilla)
    fondo = cv2.GaussianBlur(rng.uniform(40, 110, (tamano, tamano)), (0, 0), 6)
    cuadros, verdades = [], []
    x0, y0 = 20, 30
    for t in range(n_vacios + n_cuadros):
        cuadro, gt = fondo.copy(), np.zeros((tamano, tamano), np.uint8)
        if t >= n_vacios:
            k = t - n_vacios
            x, y = x0 + velocidad[0] * k, y0 + velocidad[1] * k
            cuadro[y:y + lado, x:x + lado] = 190
            gt[y:y + lado, x:x + lado] = 1
        cuadro += rng.normal(0, sigma_ruido, cuadro.shape)
        cuadros.append(np.clip(cuadro, 0, 255).astype(np.uint8))
        verdades.append(gt)
    return cuadros, verdades


# ---------------------------------------------------------- técnicas espaciales
def diferencia(f1, f2, umbral=25):
    return ((np.abs(f1.astype(np.int16) - f2.astype(np.int16)) > umbral) * 255).astype(np.uint8)


def adi(cuadros, umbral=25, referencia=0):
    """ADI absoluta, positiva y negativa respecto del cuadro de referencia."""
    ref = cuadros[referencia].astype(np.int16)
    absoluta = np.zeros(ref.shape, np.int32)
    positiva = np.zeros_like(absoluta)
    negativa = np.zeros_like(absoluta)
    for f in cuadros[referencia + 1:]:
        d = ref - f.astype(np.int16)
        absoluta += np.abs(d) > umbral
        positiva += d > umbral
        negativa += d < -umbral
    return absoluta, positiva, negativa


def fondo_por_mediana(cuadros):
    return np.median(np.stack(cuadros), axis=0).astype(np.uint8)


def limpiar(mascara):
    k = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(cv2.morphologyEx(mascara, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)


# -------------------------------------------------- dominio de la frecuencia
def correlacion_de_fase(a, b):
    """Desplazamiento (dx, dy) de b respecto de a usando FFT:
    R = F(a)·conj(F(b)) / |F(a)·conj(F(b))|, y el pico de IFFT(R) da el corrimiento."""
    ventana = cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F)
    (dx, dy), respuesta = cv2.phaseCorrelate(a.astype(np.float32), b.astype(np.float32), ventana)
    return dx, dy, respuesta


def estimar_velocidad(cuadros, fondo):
    """Aplica correlación de fase sobre el primer plano (|f - fondo|) de cuadros
    consecutivos: así el fondo estático no domina el espectro."""
    primeros_planos = [cv2.absdiff(f, fondo) for f in cuadros]
    desplazamientos = [correlacion_de_fase(primeros_planos[i], primeros_planos[i + 1])[:2]
                       for i in range(len(cuadros) - 1)]
    return np.median(np.array(desplazamientos), axis=0)


# ------------------------------------------------------------- evaluación
def evaluar_secuencia(cuadros, verdades, umbral=25):
    """Compara las técnicas contra el ground truth, solo en cuadros con objeto."""
    filas = []
    con_objeto = [i for i, g in enumerate(verdades) if g.any()]
    primero = con_objeto[0]

    def promedio(nombre, mascaras_por_indice):
        m = [evaluar(mascaras_por_indice[i], verdades[i]) for i in mascaras_por_indice]
        filas.append({"metodo": nombre, **{k: round(float(np.mean([x[k] for x in m])), 4) for k in m[0]}})

    promedio("Diferencia de cuadros consecutivos",
             {i: limpiar(diferencia(cuadros[i], cuadros[i - 1], umbral)) for i in con_objeto if i > 0})

    fondo = fondo_por_mediana(cuadros)
    promedio("Sustracción de fondo (mediana temporal)",
             {i: limpiar(diferencia(cuadros[i], fondo, umbral)) for i in con_objeto})

    # MOG2 aprende en línea, cuadro a cuadro, como en una cámara real.
    mog2 = cv2.createBackgroundSubtractorMOG2(detectShadows=False)  # history=500 (default)
    mascaras_mog = [limpiar(mog2.apply(f)) for f in cuadros]
    promedio("MOG2 (OpenCV, en línea)", {i: mascaras_mog[i] for i in con_objeto})

    # ADI: referencia = primer cuadro con el objeto; la ADI positiva marca dónde estaba.
    _, positiva, _ = adi(cuadros, umbral, referencia=primero)
    promedio("ADI positiva (objeto en cuadro de referencia)",
             {primero: limpiar((positiva > 0).astype(np.uint8) * 255)})
    return filas, fondo


# ------------------------------------------------------------- demo en vivo
def demo_en_vivo(fuente, umbral=25):
    captura = cv2.VideoCapture(fuente)
    if not captura.isOpened():
        raise SystemExit(f"No se pudo abrir la fuente de video: {fuente}")
    mog2 = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    anterior = None
    print("Presioná 'q' para salir.")
    while True:
        ok, cuadro = captura.read()
        if not ok:
            break
        gris = cv2.GaussianBlur(cv2.cvtColor(cuadro, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        diff = limpiar(diferencia(gris, anterior, umbral)) if anterior is not None else np.zeros_like(gris)
        mascara_mog = limpiar(mog2.apply(gris))
        if anterior is not None:
            dx, dy, _ = correlacion_de_fase(anterior, gris)
            cv2.putText(cuadro, f"desplazamiento global: ({dx:+.1f}, {dy:+.1f}) px", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        anterior = gris
        fila = np.hstack([cuadro, cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR), cv2.cvtColor(mascara_mog, cv2.COLOR_GRAY2BGR)])
        escala = 1100 / fila.shape[1]
        cv2.imshow("Original | Diferencia de cuadros | MOG2", cv2.resize(fila, None, fx=escala, fy=escala))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    captura.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camara", type=int, help="índice de la webcam (ej. 0)")
    parser.add_argument("--video", help="ruta a un archivo de video")
    parser.add_argument("--salida", default=os.path.join("aporte", "resultados"))
    args = parser.parse_args()

    if args.camara is not None or args.video:
        demo_en_vivo(args.camara if args.camara is not None else args.video)
        return

    import matplotlib.pyplot as plt

    velocidad_real = (3, 2)
    cuadros, verdades = generar_secuencia(velocidad=velocidad_real)
    filas, fondo = evaluar_secuencia(cuadros, verdades)
    primero = next(i for i, g in enumerate(verdades) if g.any())
    vx, vy = estimar_velocidad(cuadros[primero:], fondo)
    error = float(np.hypot(vx - velocidad_real[0], vy - velocidad_real[1]))

    os.makedirs(args.salida, exist_ok=True)
    with open(os.path.join(args.salida, "movimiento_metricas.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)

    print(f"{'Método':48s} {'IoU':>6s} {'Dice':>6s} {'Exact.':>7s}")
    for fila in filas:
        print(f"{fila['metodo']:48s} {fila['IoU']:6.3f} {fila['Dice']:6.3f} {fila['Exactitud']:7.3f}")
    print(f"\nDominio de la frecuencia (correlación de fase): velocidad estimada "
          f"({vx:+.2f}, {vy:+.2f}) px/cuadro | real {velocidad_real} | error {error:.2f} px/cuadro")

    t = primero + 12
    absoluta, positiva, negativa = adi(cuadros, referencia=primero)
    paneles = [
        (f"Cuadro {t}", cuadros[t], "gray"),
        ("Diferencia t vs t-1", limpiar(diferencia(cuadros[t], cuadros[t - 1])), "gray"),
        ("Fondo (mediana temporal)", fondo, "gray"),
        ("Sustracción de fondo", limpiar(diferencia(cuadros[t], fondo)), "gray"),
        ("ADI absoluta", absoluta, "magma"),
        ("ADI positiva", positiva, "magma"),
        ("ADI negativa", negativa, "magma"),
        (f"Correlación de fase\nv=({vx:+.1f}, {vy:+.1f}) px/cuadro", None, None),
    ]
    fig, ejes = plt.subplots(2, 4, figsize=(16, 8))
    for eje, (titulo, img, cmap) in zip(ejes.ravel(), paneles):
        if img is None:  # superficie de correlación de fase: el pico marca el desplazamiento
            pp =[cv2.absdiff(cuadros[k], fondo).astype(np.float32) for k in (t, t + 1)]
            espectro = np.fft.fft2(pp[0]) * np.conj(np.fft.fft2(pp[1]))
            superficie = np.fft.fftshift(np.abs(np.fft.ifft2(espectro / (np.abs(espectro) + 1e-9))))
            cy, cx = superficie.shape[0] // 2, superficie.shape[1] // 2
            eje.imshow(superficie[cy - 12:cy + 13, cx - 12:cx + 13], cmap="magma",
                       extent=(-12.5, 12.5, 12.5, -12.5))
            eje.axis("on")
            eje.set_xlabel("dx (px)")
            eje.set_ylabel("dy (px)")
            eje.set_title(titulo, fontsize=10)
            continue
        else:
            eje.imshow(img, cmap=cmap)
        eje.set_title(titulo, fontsize=10)
        eje.axis("off")
    fig.suptitle("Segmentación basada en movimiento", fontsize=14)
    fig.tight_layout()
    destino = os.path.join(args.salida, "movimiento.png")
    fig.savefig(destino, dpi=120)
    print(f"Figura guardada en {destino}")
    if os.environ.get("MPLBACKEND", "").lower() != "agg":
        plt.show()


if __name__ == "__main__":
    main()
