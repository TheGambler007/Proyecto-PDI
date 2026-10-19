# Aporte del grupo — Segmentación de imágenes

Fork de [Computer-Vision-Toolkit/Segmentor_and_Threshold](https://github.com/Computer-Vision-Toolkit/Segmentor_and_Threshold).
El repo original cubría umbralización (Otsu, óptimo, espectral) y segmentación por
regiones/clustering. Este aporte completa los temas que faltaban de la guía y agrega
métricas para comparar todo bajo las mismas condiciones.

## Qué se agregó

| Archivo | Tema de la guía | Qué hace |
|---|---|---|
| `aporte/discontinuidades.py` | 2a–2d | Puntos (Laplaciano), líneas (4 máscaras), bordes (Sobel, Prewitt, LoG, Canny) y enfoque combinado (gradiente + Otsu + relleno → máscara) |
| `aporte/split_merge.py` | 4c | División y fusión con quadtree. **Umbrales automáticos** según el ruido estimado (Immerkær) + mediana si el ruido es alto |
| `aporte/movimiento.py` | 5a, 5b | Diferencia de cuadros, ADI absoluta/positiva/negativa, sustracción de fondo, MOG2 y **correlación de fase** (dominio de la frecuencia) para estimar la velocidad. Demo en vivo con webcam |
| `aporte/escenarios.py` | 3b | 3 escenarios sintéticos con ground truth: iluminación uniforme, no uniforme y ruido fuerte |
| `aporte/metricas.py` | — | IoU, Dice, exactitud y tiempo |
| `aporte/benchmark.py` | 6 | Corre 6 métodos × 3 escenarios y genera tabla, CSV y figuras |
| `aporte/comparar_parches.py` | — | Verifica que las mejoras al código original dan el mismo resultado y mide la aceleración |

### Mejoras al código original

| Archivo | Cambio | Resultado |
|---|---|---|
| `RegionGrowingSegmentation.py` | El crecimiento recorría píxel por píxel con una cola en Python. Como el criterio compara contra el valor de la semilla, la región es la componente conexa de `{distancia ≤ T}` que contiene a la semilla: se calcula vectorizado con `cv2.connectedComponents`. También se corrigió un bug con semillas fuera de la imagen | **36–73× más rápido**, resultado idéntico (gris, RGB y CIELUV) |
| `OtsuAndOptimal.py` | Acepta arrays NumPy además de rutas. `local_otsu`/`local_optimal` ya no escriben y releen un PNG por cada ventana | **~4× más rápido**, resultado idéntico |
| `requirements.txt` | El README lo pedía pero no existía | `pip install -r requirements.txt` funciona |
| `.gitignore` | Se versionaban `__pycache__/` y `.idea/` | — |

## Cómo correrlo (desde la raíz del repo)

```bash
pip install -r requirements.txt

python -m aporte.benchmark            # 6 métodos × 3 escenarios con métricas
python -m aporte.comparar_parches     # antes vs. después de las mejoras
python -m aporte.discontinuidades     # puntos, líneas y bordes
python -m aporte.split_merge threshold_images/brain_mri.jpeg --sigma-max 12 --umbral-fusion 15
python -m aporte.movimiento           # secuencia sintética + métricas
python -m aporte.movimiento --camara 0   # demo en vivo (q para salir)

# comparación cualitativa sobre una imagen real (sin ground truth)
python -m aporte.benchmark --imagen threshold_images/brain_mri.jpeg --semilla 230,150

python main.py                        # la GUI original sigue funcionando igual
```

Todo se guarda en `aporte/resultados/`.

## Resultados (IoU, 1 = perfecto)

| Método | Origen | A) Uniforme | B) No uniforme | C) Ruido fuerte |
|---|---|---:|---:|---:|
| Otsu global | repo | 1.000 | 0.669 | 0.720 |
| Óptimo global | repo | 1.000 | 0.650 | 0.720 |
| Otsu local (64 px) | repo | 0.407 | 0.392 | 0.362 |
| Crecimiento de regiones | repo | 1.000 | 0.234 | 0.158 |
| Split & merge | aporte | 0.948 | 0.949 | 0.954 |
| Bordes + relleno | aporte | 0.906 | 0.917 | 0.908 |

Movimiento (secuencia sintética, IoU promedio): diferencia de cuadros 0.10 ·
sustracción de fondo 1.00 · MOG2 0.91 · ADI positiva 1.00. Correlación de fase:
velocidad estimada (3.01, 2.01) px/cuadro vs. real (3, 2).

Todos los métodos usan los **mismos parámetros por defecto en los 3 escenarios**.

## Puntos para el debate (trade-offs)

- **Umbral global**: perfecto con iluminación uniforme, se rompe con iluminación en
  rampa (el histograma deja de ser bimodal). Es el método más rápido.
- **Otsu local**: debería resolver la iluminación no uniforme, pero en ventanas que
  son todo fondo Otsu igual "parte" el ruido en dos clases, y en ventanas constantes
  devuelve todo blanco. Falla en los tres escenarios.
- **Crecimiento de regiones**: compara contra el valor de una sola semilla. Con
  ruido, ese valor ya es ruidoso; con iluminación en rampa, el objeto cambia de
  brillo y el umbral de 50 se escapa al fondo. Muy sensible al umbral.
- **Split & merge**: el más estable (≈0.95 en todo), porque fusiona mirando la
  frontera local entre bloques, no un valor global. Costo: bordes "escalonados"
  (resolución del bloque mínimo de 4 px) y es más lento.
- **Bordes + relleno**: robusto a la iluminación (el gradiente es local), pero solo
  funciona si los contornos quedan cerrados.
- **Movimiento**: la diferencia de cuadros solo ve los bordes del objeto que
  avanzan (IoU 0.10); la sustracción de fondo necesita un fondo limpio; MOG2 depende
  del calentamiento: con 10 cuadros previos de fondo cae a 0.51, con 30 llega a 0.91.

## Mejoras futuras sugeridas

- Split & merge: refinar a nivel píxel los bloques de borde (eliminar el escalonado).
- Otsu local: no umbralizar ventanas con varianza baja (asignarlas al vecino).
- Crecimiento de regiones: comparar contra la media de la región, no solo la semilla.
- Integrar los módulos nuevos a la GUI PyQt (`task04.ui`).
