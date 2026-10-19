"""Benchmark: compara 6 mecanismos de segmentación en 3 escenarios con métricas.

Métodos del repositorio original: Otsu global, Óptimo global, Otsu local y
Crecimiento de regiones. Métodos del aporte: Split & merge y Bordes + relleno.

Cada método corre con sus parámetros por defecto (los mismos para los 3
escenarios), para que la comparación sea justa.

Uso (desde la raíz del repo):
    python -m aporte.benchmark                        # 3 escenarios con métricas
    python -m aporte.benchmark --imagen threshold_images/orange.jpg --semilla 220,180
"""
import argparse
import csv
import os

import cv2
import numpy as np

from aporte.discontinuidades import segmentar_por_bordes
from aporte.escenarios import ESCENARIOS, TITULOS
from aporte.metricas import cronometrar, evaluar
from aporte.split_merge import split_and_merge

# Paleta categórica validada (slots 1 y 2) + tinta neutra para texto y ejes.
COLOR_REPO = "#2a78d6"
COLOR_APORTE = "#eb6834"
TINTA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
GRILLA = "#e4e3df"


def semillas_desde_gt(gt):
    """Una semilla por objeto (su centroide): equivale a hacer clic sobre cada
    objeto en la GUI de crecimiento de regiones."""
    n, _, _, centroides = cv2.connectedComponentsWithStats(gt.astype(np.uint8))
    return [(int(round(x)), int(round(y))) for x, y in centroides[1:n]]


def metodos(semillas):
    # Importes del repo original (se ejecuta desde la raíz del repositorio).
    from OtsuAndOptimal import OtsuAndOptimal
    from RegionGrowingSegmentation import region_growing

    def a_mascara(resultado):
        return ((resultado > 0) * 255).astype(np.uint8)

    return [
        ("Otsu global", "repo", lambda img: OtsuAndOptimal(img).otsu_thresholding()),
        ("Óptimo global", "repo",
         lambda img: OtsuAndOptimal(img).optimal_thresholding(initial_threshold=128, tolerance=0.5)),
        ("Otsu local (64 px)", "repo", lambda img: OtsuAndOptimal(img).local_otsu(64)),
        ("Crecimiento de regiones", "repo",
         lambda img: a_mascara(region_growing(img, semillas, threshold=50, use_luv=False))),
        ("Split & merge", "aporte", lambda img: split_and_merge(img)),
        ("Bordes + relleno", "aporte", lambda img: segmentar_por_bordes(img)),
    ]


def figura_escenario(nombre, imagen, gt, resultados, destino):
    import matplotlib.pyplot as plt

    paneles = [("Imagen", imagen, None), ("Ground truth", gt * 255, None)]
    paneles += [(m, r["mascara"], r) for m, r in resultados.items()]
    fig, ejes = plt.subplots(2, 4, figsize=(16, 8.6), layout="constrained")
    for eje, (titulo, img, r) in zip(ejes.ravel(), paneles):
        eje.imshow(img, cmap="gray", vmin=0, vmax=255)
        eje.set_xticks([])
        eje.set_yticks([])
        if r is None:
            eje.set_title(titulo, fontsize=11, color=TINTA)
            for borde in eje.spines.values():
                borde.set_color(GRILLA)
        else:
            color = COLOR_APORTE if r["origen"] == "aporte" else COLOR_REPO
            eje.set_title(f"{titulo}\nIoU {r['IoU']:.3f} · {r['tiempo_ms']:.0f} ms", fontsize=11, color=TINTA)
            for borde in eje.spines.values():
                borde.set_color(color)
                borde.set_linewidth(3)
    fig.suptitle(f"Escenario {TITULOS[nombre]}  ·  borde azul = repositorio, naranja = aporte",
                 fontsize=13, color=TINTA)
    fig.savefig(destino, dpi=110, facecolor="white")
    plt.close(fig)


def figura_resumen(filas, destino):
    """Small multiples: un panel por escenario, IoU por método (barras horizontales)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    escenarios = list(dict.fromkeys(f["escenario"] for f in filas))
    nombres = list(dict.fromkeys(f["metodo"] for f in filas))
    fig, ejes = plt.subplots(1, len(escenarios), figsize=(15, 4.6), sharey=True, layout="constrained")
    for eje, esc in zip(ejes, escenarios):
        datos = {f["metodo"]: f for f in filas if f["escenario"] == esc}
        y = np.arange(len(nombres))[::-1]
        valores = [datos[m]["IoU"] for m in nombres]
        colores = [COLOR_APORTE if datos[m]["origen"] == "aporte" else COLOR_REPO for m in nombres]
        eje.barh(y, valores, height=0.62, color=colores)
        for yi, v in zip(y, valores):
            eje.text(v + 0.015, yi, f"{v:.2f}", va="center", fontsize=10, color=TINTA)
        eje.set_xlim(0, 1.12)
        eje.set_yticks(y, nombres, fontsize=10, color=TINTA)
        eje.set_title(TITULOS[esc], fontsize=11, color=TINTA, loc="left")
        eje.xaxis.grid(True, color=GRILLA, linewidth=0.8)
        eje.set_axisbelow(True)
        eje.tick_params(axis="x", colors=TINTA_SECUNDARIA, labelsize=9)
        eje.tick_params(axis="y", length=0)
        for lado in ("top", "right", "left"):
            eje.spines[lado].set_visible(False)
        eje.spines["bottom"].set_color(GRILLA)
        eje.set_xlabel("IoU (1 = segmentación perfecta)", fontsize=9, color=TINTA_SECUNDARIA)
    fig.legend(handles=[Patch(color=COLOR_REPO, label="Repositorio original"),
                        Patch(color=COLOR_APORTE, label="Aporte del grupo")],
               loc="outside upper right", ncols=2, frameon=False, fontsize=10)
    fig.suptitle("IoU por método y escenario", fontsize=13, color=TINTA, x=0.01, ha="left")
    fig.savefig(destino, dpi=130, facecolor="white")
    plt.close(fig)


def tabla_markdown(filas):
    escenarios = list(dict.fromkeys(f["escenario"] for f in filas))
    nombres = list(dict.fromkeys(f["metodo"] for f in filas))
    idx = {(f["escenario"], f["metodo"]): f for f in filas}
    cortos = [e.split("_", 1)[0] for e in escenarios]
    lineas = ["| Método | Origen | " + " | ".join(f"IoU {c}" for c in cortos) + " | " +
              " | ".join(f"ms {c}" for c in cortos) + " |",
              "|---|---|" + "---:|" * (2 * len(escenarios))]
    for m in nombres:
        origen = idx[(escenarios[0], m)]["origen"]
        lineas.append(f"| {m} | {origen} | " +
                      " | ".join(f"{idx[(e, m)]['IoU']:.3f}" for e in escenarios) + " | " +
                      " | ".join(f"{idx[(e, m)]['tiempo_ms']:.0f}" for e in escenarios) + " |")
    leyenda = "\n".join(f"- **{TITULOS[e]}**" for e in escenarios)
    return "\n".join(lineas) + "\n\n" + leyenda + "\n"


def correr_escenarios(salida):
    filas = []
    for nombre, generar in ESCENARIOS.items():
        imagen, gt = generar()
        resultados = {}
        for metodo, origen, funcion in metodos(semillas_desde_gt(gt)):
            mascara, ms = cronometrar(funcion, imagen, repeticiones=3)
            fila = {"escenario": nombre, "metodo": metodo, "origen": origen,
                    **evaluar(mascara, gt), "tiempo_ms": round(ms, 1)}
            filas.append(fila)
            resultados[metodo] = {**fila, "mascara": mascara}
            print(f"{nombre:28s} {metodo:26s} IoU {fila['IoU']:.3f}  Dice {fila['Dice']:.3f}  "
                  f"Exact. {fila['Exactitud']:.3f}  {ms:7.1f} ms")
        figura_escenario(nombre, imagen, gt, resultados, os.path.join(salida, f"benchmark_{nombre}.png"))
        cv2.imwrite(os.path.join(salida, f"escenario_{nombre}.png"), imagen)
        print()

    with open(os.path.join(salida, "benchmark_metricas.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    with open(os.path.join(salida, "benchmark_tabla.md"), "w", encoding="utf-8") as f:
        f.write(tabla_markdown(filas))
    figura_resumen(filas, os.path.join(salida, "benchmark_iou.png"))
    print(f"Resultados en {salida}/ (CSV, tabla Markdown y figuras)")


def correr_imagen_real(ruta, semilla, salida):
    """Sin ground truth: comparación cualitativa + tiempos (para la demo en vivo)."""
    import matplotlib.pyplot as plt

    imagen = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
    if imagen is None:
        raise SystemExit(f"No se pudo leer {ruta}")
    h, w = imagen.shape
    semilla = tuple(int(v) for v in semilla.split(",")) if semilla else (w // 2, h // 2)
    paneles = [("Imagen (semilla en rojo)", cv2.circle(cv2.cvtColor(imagen, cv2.COLOR_GRAY2RGB), semilla, 5,
                                                       (230, 40, 40), -1), "repo")]
    for metodo, origen, funcion in metodos([semilla]):
        mascara, ms = cronometrar(funcion, imagen, repeticiones=1)
        paneles.append((f"{metodo} · {ms:.0f} ms", mascara, origen))
        print(f"{metodo:26s} {ms:7.1f} ms")
    fig, ejes = plt.subplots(1, len(paneles), figsize=(3.2 * len(paneles), 3.8), layout="constrained")
    for eje, (titulo, img, origen) in zip(ejes, paneles):
        eje.imshow(img, cmap="gray", vmin=0, vmax=255)
        eje.set_title(titulo, fontsize=9, color=TINTA)
        eje.set_xticks([])
        eje.set_yticks([])
        for borde in eje.spines.values():  # marco visible: algunas máscaras tienen zonas blancas en el borde
            borde.set_color(COLOR_APORTE if origen == "aporte" else "#b8b7b2")
            borde.set_linewidth(2 if origen == "aporte" else 1)
    destino = os.path.join(salida, f"real_{os.path.splitext(os.path.basename(ruta))[0]}.png")
    fig.savefig(destino, dpi=120, facecolor="white")
    print(f"Figura guardada en {destino}")
    if os.environ.get("MPLBACKEND", "").lower() != "agg":
        plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--imagen", help="imagen real para comparación cualitativa (sin métricas)")
    parser.add_argument("--semilla", help="semilla x,y para crecimiento de regiones (por defecto, el centro)")
    parser.add_argument("--salida", default=os.path.join("aporte", "resultados"))
    args = parser.parse_args()
    os.makedirs(args.salida, exist_ok=True)
    if args.imagen:
        correr_imagen_real(args.imagen, args.semilla, args.salida)
    else:
        correr_escenarios(args.salida)


if __name__ == "__main__":
    main()
