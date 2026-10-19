"""Métricas para comparar algoritmos de segmentación contra un ground truth."""
import time

import numpy as np


def _binaria(mascara):
    return np.asarray(mascara) > 0


def iou(pred, gt):
    """Intersection over Union (índice de Jaccard)."""
    p, g = _binaria(pred), _binaria(gt)
    union = np.logical_or(p, g).sum()
    return 1.0 if union == 0 else np.logical_and(p, g).sum() / union


def dice(pred, gt):
    """Coeficiente de Dice (F1 a nivel de píxel)."""
    p, g = _binaria(pred), _binaria(gt)
    total = p.sum() + g.sum()
    return 1.0 if total == 0 else 2 * np.logical_and(p, g).sum() / total


def exactitud(pred, gt):
    """Porcentaje de píxeles bien clasificados (objeto y fondo)."""
    return float((_binaria(pred) == _binaria(gt)).mean())


def evaluar(pred, gt):
    return {
        "IoU": round(float(iou(pred, gt)), 4),
        "Dice": round(float(dice(pred, gt)), 4),
        "Exactitud": round(exactitud(pred, gt), 4),
    }


def cronometrar(funcion, *args, repeticiones=3, **kwargs):
    """Ejecuta la función varias veces y devuelve (resultado, mediana en ms)."""
    tiempos, resultado = [], None
    for _ in range(repeticiones):
        t0 = time.perf_counter()
        resultado = funcion(*args, **kwargs)
        tiempos.append((time.perf_counter() - t0) * 1000)
    return resultado, float(np.median(tiempos))
