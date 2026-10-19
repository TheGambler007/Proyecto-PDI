"""Escenarios sintéticos con ground truth conocido.

Se usa el modelo de formación de imagen f(x, y) = i(x, y) · r(x, y)
(iluminación × reflectancia), como en Gonzalez & Woods, para controlar
exactamente qué condición se está probando en cada escenario.
"""
import cv2
import numpy as np

TAMANO = 256
REFLECTANCIA_OBJETO = 0.75
REFLECTANCIA_FONDO = 0.35


def mascara_verdad(tamano=TAMANO):
    """Ground truth: un círculo y un rectángulo (1 = objeto, 0 = fondo)."""
    gt = np.zeros((tamano, tamano), np.uint8)
    cv2.circle(gt, (int(tamano * 0.30), int(tamano * 0.35)), int(tamano * 0.16), 1, -1)
    cv2.rectangle(gt, (int(tamano * 0.55), int(tamano * 0.55)),
                  (int(tamano * 0.88), int(tamano * 0.85)), 1, -1)
    return gt


def _componer(iluminacion, sigma_ruido, semilla):
    gt = mascara_verdad(iluminacion.shape[0])
    reflectancia = np.where(gt == 1, REFLECTANCIA_OBJETO, REFLECTANCIA_FONDO)
    imagen = iluminacion * reflectancia
    rng = np.random.default_rng(semilla)
    imagen = imagen + rng.normal(0, sigma_ruido, imagen.shape)
    return np.clip(imagen, 0, 255).astype(np.uint8), gt


def iluminacion_uniforme(tamano=TAMANO):
    """A) Iluminación uniforme + ruido leve: el caso ideal (histograma bimodal)."""
    return _componer(np.full((tamano, tamano), 255.0), sigma_ruido=5, semilla=1)


def iluminacion_no_uniforme(tamano=TAMANO):
    """B) Rampa de iluminación de izquierda a derecha: rompe el umbral global."""
    rampa = np.linspace(60, 255, tamano)
    return _componer(np.tile(rampa, (tamano, 1)), sigma_ruido=5, semilla=2)


def ruido_fuerte(tamano=TAMANO):
    """C) Iluminación uniforme pero ruido gaussiano fuerte (sigma = 30)."""
    return _componer(np.full((tamano, tamano), 255.0), sigma_ruido=30, semilla=3)


TITULOS = {
    "A_iluminacion_uniforme": "A) Iluminación uniforme",
    "B_iluminacion_no_uniforme": "B) Iluminación no uniforme",
    "C_ruido_fuerte": "C) Ruido fuerte (σ = 30)",
}

ESCENARIOS = {
    "A_iluminacion_uniforme": iluminacion_uniforme,
    "B_iluminacion_no_uniforme": iluminacion_no_uniforme,
    "C_ruido_fuerte": ruido_fuerte,
}
