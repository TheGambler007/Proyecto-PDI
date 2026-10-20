"""Interfaz unificada: muestra lo que hacía el repositorio original y lo que agrega el aporte.

Pestañas:
  Inicio                 capacidades antes / después y acceso a cada módulo
  Antes vs. después      los 6 métodos sobre la misma imagen, con métricas
  Discontinuidades       puntos, líneas, bordes y enfoque combinado (nuevo)
  Split & merge          quadtree con parámetros automáticos (nuevo)
  Movimiento             diferencias, ADI, fondo, MOG2 y correlación de fase (nuevo)
  Mejoras al código      original vs. optimizado: mismo resultado, más rápido

Uso (desde la raíz del repo):
    python -m aporte.app
"""
import os
import sys

import cv2
import numpy as np
from PyQt5.QtCore import QObject, QSize, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QPixmap
from PyQt5.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
                             QFileDialog, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView,
                             QLabel, QMainWindow, QPushButton, QScrollArea, QSizePolicy, QSlider,
                             QSpinBox, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

import RegionGrowingSegmentation as modulo_rg
from OtsuAndOptimal import OtsuAndOptimal
from aporte.benchmark import metodos, semillas_desde_gt
from aporte.comparar_parches import local_otsu_original, region_growing_original
from aporte.discontinuidades import (bordes_canny, bordes_log, bordes_prewitt, bordes_sobel,
                                     detectar_lineas, detectar_puntos, imagen_de_prueba,
                                     segmentar_por_bordes)
from aporte.escenarios import ESCENARIOS, TITULOS
from aporte.metricas import cronometrar, evaluar
from aporte.movimiento import (adi, correlacion_de_fase, diferencia, estimar_velocidad,
                               fondo_por_mediana, generar_secuencia, limpiar)
from aporte.split_merge import dibujar_quadtree, split_and_merge

COLOR_REPO = "#2a78d6"
COLOR_APORTE = "#eb6834"
COLOR_SECUNDARIO = "#6b6a66"
LADO_MAXIMO = 512  # las imágenes cargadas se achican a este lado para que todo sea interactivo


# ================================================================ utilidades
def a_qimage(img):
    """uint8 gris (H, W) o BGR (H, W, 3) -> QImage con copia propia de los datos."""
    img = np.ascontiguousarray(img)
    if img.dtype != np.uint8:
        img = cv2.normalize(img.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    h, w = img.shape[:2]
    if img.ndim == 2:
        return QImage(img.data, w, h, w, QImage.Format_Grayscale8).copy()
    rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()


def mapa_de_calor(conteos):
    """Para las ADI: conteo por píxel -> colores (0 queda negro)."""
    if conteos.max() == 0:
        return np.zeros(conteos.shape + (3,), np.uint8)
    norm = (conteos.astype(np.float32) / conteos.max() * 255).astype(np.uint8)
    salida = cv2.applyColorMap(norm, cv2.COLORMAP_MAGMA)
    salida[conteos == 0] = 0
    return salida


def mapa_de_errores(pred, gt):
    """Aciertos en blanco/negro, falsos positivos en naranja y falsos negativos en azul."""
    p, g = pred > 0, gt > 0
    salida = np.zeros(gt.shape + (3,), np.uint8)
    salida[p & g] = (255, 255, 255)
    salida[p & ~g] = (52, 104, 235)   # BGR de COLOR_APORTE: sobra
    salida[~p & g] = (214, 120, 42)   # BGR de COLOR_REPO: falta
    return salida


def colorear_regiones(etiquetas):
    rng = np.random.default_rng(7)
    paleta = rng.integers(40, 256, (int(etiquetas.max()) + 1, 3), dtype=np.uint8)
    return paleta[etiquetas]


def dibujar_semillas(gris, semillas):
    salida = cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)
    radio = max(3, gris.shape[0] // 80)
    for x, y in semillas:
        cv2.circle(salida, (x, y), radio + 2, (255, 255, 255), -1)
        cv2.circle(salida, (x, y), radio, (0, 0, 255), -1)
    return salida


def texto_metricas(metricas, ms=None):
    partes = []
    if metricas:
        partes.append(f"IoU {metricas['IoU']:.3f} · Dice {metricas['Dice']:.3f}")
    if ms is not None:
        partes.append(f"{ms:.1f} ms")
    return " · ".join(partes)


def etiqueta(texto, tamano=None, negrita=False, color=None, ajustar=True):
    lab = QLabel(texto)
    lab.setWordWrap(ajustar)
    lab.setTextFormat(Qt.RichText)
    estilo = []
    if tamano:
        estilo.append(f"font-size: {tamano}pt")
    if negrita:
        estilo.append("font-weight: 600")
    if color:
        estilo.append(f"color: {color}")
    lab.setStyleSheet("; ".join(estilo))
    return lab


def aviso(texto, color):
    """Banda de una línea arriba de cada pestaña que dice si es nuevo o mejorado."""
    lab = QLabel(texto)
    lab.setWordWrap(True)
    r, g, b = (int(color[i:i + 2], 16) for i in (1, 3, 5))
    lab.setStyleSheet(f"background: rgba({r}, {g}, {b}, 30); border-left: 4px solid {color};"
                      f" padding: 8px 10px;")
    return lab


def con_scroll(contenido):
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.NoFrame)
    area.setWidget(contenido)
    return area


# ============================================================== hilos de cálculo
class Tarea(QThread):
    """Corre una función fuera del hilo de la interfaz y devuelve el resultado (o la excepción)."""
    listo = pyqtSignal(object)
    progreso = pyqtSignal(object)

    def __init__(self, funcion, parent=None):
        super().__init__(parent)
        self.funcion = funcion

    def run(self):
        try:
            resultado = self.funcion(self.progreso.emit)
        except Exception as e:  # se muestra en la interfaz en vez de cerrar la app
            resultado = e
        self.listo.emit(resultado)


class ConTareas:
    """Mezcla para pestañas: lanza tareas y descarta resultados viejos si se relanzó otra."""

    def lanzar(self, clave, funcion, al_terminar, al_progresar=None):
        if not hasattr(self, "_generacion"):
            self._generacion, self._tareas = {}, set()
        gen = self._generacion.get(clave, 0) + 1
        self._generacion[clave] = gen
        tarea = Tarea(funcion, self)

        def terminar(resultado):
            self._tareas.discard(tarea)
            if self._generacion.get(clave) == gen:
                al_terminar(resultado)

        tarea.listo.connect(terminar)
        tarea.finished.connect(tarea.deleteLater)
        if al_progresar:
            tarea.progreso.connect(lambda dato: self._generacion.get(clave) == gen and al_progresar(dato))
        self._tareas.add(tarea)
        tarea.start()


# ================================================================ widgets básicos
class VistaImagen(QLabel):
    """Muestra una imagen escalada al espacio disponible y avisa clics en coordenadas de imagen."""
    clic = pyqtSignal(int, int, object)

    def __init__(self, minimo=200):
        super().__init__()
        self._minimo = minimo
        self._pix = None
        self._forma = None
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background: transparent; color: #888;")
        self.setText("—")

    def sizeHint(self):
        return QSize(self._minimo, self._minimo)

    def minimumSizeHint(self):
        return QSize(self._minimo, self._minimo)

    def mostrar(self, img):
        if img is None:
            self._pix, self._forma = None, None
            self.clear()
            self.setText("—")
            return
        self._forma = img.shape[:2]
        self._pix = QPixmap.fromImage(a_qimage(img))
        self._ajustar()

    def _ajustar(self):
        if self._pix is not None:
            self.setPixmap(self._pix.scaled(self.contentsRect().size(), Qt.KeepAspectRatio,
                                            Qt.FastTransformation))

    def resizeEvent(self, evento):
        self._ajustar()
        super().resizeEvent(evento)

    def mousePressEvent(self, evento):
        pix = self.pixmap()
        if self._forma is None or pix is None or pix.isNull():
            return
        ox = (self.width() - pix.width()) / 2
        oy = (self.height() - pix.height()) / 2
        x = int((evento.x() - ox) * self._forma[1] / pix.width())
        y = int((evento.y() - oy) * self._forma[0] / pix.height())
        if 0 <= x < self._forma[1] and 0 <= y < self._forma[0]:
            self.clic.emit(x, y, evento.button())


class Panel(QFrame):
    """Título + imagen + pie de texto. El borde superior indica el origen (repo / aporte)."""

    def __init__(self, titulo, origen=None, minimo=200):
        super().__init__()
        color = {"repo": COLOR_REPO, "aporte": COLOR_APORTE}.get(origen, "#999999")
        self.setObjectName("panel")
        self.setStyleSheet(f"#panel {{ border: 1px solid #c8c7c2; border-top: 4px solid {color};"
                           f" border-radius: 4px; }}")
        capa = QVBoxLayout(self)
        capa.setContentsMargins(6, 4, 6, 6)
        capa.setSpacing(4)
        self.titulo = etiqueta(titulo, negrita=True)
        self.vista = VistaImagen(minimo)
        self.pie = etiqueta("", color=COLOR_SECUNDARIO)
        self.pie.setMinimumHeight(18)
        capa.addWidget(self.titulo)
        capa.addWidget(self.vista, 1)
        capa.addWidget(self.pie)

    def mostrar(self, img, pie=""):
        self.vista.mostrar(img)
        self.pie.setText(pie)


def grupo(titulo, color=None):
    caja = QGroupBox(titulo)
    if color:
        caja.setStyleSheet(f"QGroupBox {{ font-weight: 600; color: {color}; }}")
    return caja


def celda(texto, color=None, negrita=False, alinear=Qt.AlignCenter):
    item = QTableWidgetItem(texto)
    item.setTextAlignment(alinear | Qt.AlignVCenter)
    if color:
        item.setForeground(QColor(color))
    if negrita:
        fuente = item.font()
        fuente.setBold(True)
        item.setFont(fuente)
    return item


def tabla(columnas):
    t = QTableWidget(0, len(columnas))
    t.setHorizontalHeaderLabels(columnas)
    t.verticalHeader().setVisible(False)
    t.setWordWrap(False)
    t.setEditTriggers(QAbstractItemView.NoEditTriggers)
    t.setSelectionMode(QAbstractItemView.NoSelection)
    t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    t.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
    return t


def ajustar_alto(t):
    t.resizeRowsToContents()
    alto = t.horizontalHeader().height() + sum(t.rowHeight(i) for i in range(t.rowCount())) + 4
    t.setFixedHeight(alto)


# ================================================================ estado compartido
class Estado(QObject):
    """Imagen activa compartida por las pestañas de imagen fija."""
    cambio = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.nombre = ""
        self.gris = None
        self.color = None
        self.gt = None
        self.semillas = []
        self.semillas_defecto = []

    def _poner(self, nombre, gris, color, gt, semillas):
        self.nombre, self.gris, self.color, self.gt = nombre, gris, color, gt
        self.semillas_defecto = list(semillas)
        self.semillas = list(semillas)
        self.cambio.emit()

    def cargar_escenario(self, clave):
        imagen, gt = ESCENARIOS[clave]()
        self._poner(TITULOS[clave], imagen, cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR), gt,
                    semillas_desde_gt(gt))

    def cargar_prueba(self):
        imagen = imagen_de_prueba()
        self._poner("Imagen de prueba (puntos, líneas y figura)", imagen,
                    cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR), None, [(190, 190)])

    def cargar_archivo(self, ruta):
        # np.fromfile + imdecode: funciona con rutas con tildes en Windows (cv2.imread no).
        color = cv2.imdecode(np.fromfile(ruta, np.uint8), cv2.IMREAD_COLOR)
        if color is None:
            raise ValueError(f"No se pudo leer {ruta}")
        escala = LADO_MAXIMO / max(color.shape[:2])
        if escala < 1:
            color = cv2.resize(color, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)
        gris = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        h, w = gris.shape
        self._poner(f"Archivo: {os.path.basename(ruta)}", gris, color, None, [(w // 2, h // 2)])

    def poner_semillas(self, semillas):
        self.semillas = list(semillas)
        self.cambio.emit()


class Pestana(QWidget):
    """Base: se recalcula solo cuando está visible y la imagen cambió."""

    def __init__(self, estado=None):
        super().__init__()
        self.estado = estado
        self.sucia = True
        if estado is not None:
            estado.cambio.connect(self.marcar)

    def marcar(self):
        self.sucia = True
        if self.isVisible():
            self.refrescar()

    def showEvent(self, evento):
        super().showEvent(evento)
        if self.sucia:
            self.refrescar()

    def refrescar(self):
        self.sucia = False
        self.actualizar()

    def actualizar(self):
        pass

    def detener(self):
        pass


# ================================================================ pestaña Inicio
CAPACIDADES = [
    # (técnica, tema de la guía, antes, después, pestaña destino)
    ("Umbralización global (Otsu, óptimo)", "—", "Sí", "Igual, y ahora acepta arrays NumPy", 1),
    ("Umbralización local (ventanas)", "—", "Sí (lenta: un PNG por ventana)",
     "Varias veces más rápida, mismo resultado", 5),
    ("Crecimiento de regiones", "—", "Sí (lento + bug de semillas)",
     "36–73× más rápido, mismo resultado, bug corregido", 5),
    ("Detección de puntos y líneas", "2a–2b", "No", "Laplaciano y 4 máscaras direccionales", 2),
    ("Detección de bordes", "2c", "No", "Sobel, Prewitt, LoG y Canny", 2),
    ("Segmentación por bordes (combinada)", "2d", "No", "Gradiente + Otsu + relleno → máscara", 2),
    ("División y fusión (split & merge)", "4c", "No", "Quadtree con umbrales automáticos según el ruido", 3),
    ("Segmentación por movimiento", "5a–5b", "No",
     "Diferencias, ADI, fondo, MOG2 y correlación de fase", 4),
    ("Métricas y comparación objetiva", "6", "No (solo inspección visual)",
     "IoU, Dice, exactitud y tiempo sobre escenarios con ground truth", 1),
]


class PestanaInicio(Pestana):
    def __init__(self, ir_a):
        super().__init__()
        contenido = QWidget()
        capa = QVBoxLayout(contenido)
        capa.setSpacing(12)
        capa.addWidget(etiqueta("Qué podía hacer el programa antes y qué agrega el aporte", 15, True))
        capa.addWidget(etiqueta(
            "El repositorio original (<i>Segmentor_and_Threshold</i>) cubría umbralización y "
            "segmentación por regiones. El aporte completa los temas que faltaban de la guía, "
            "acelera el código existente sin cambiar sus resultados y agrega métricas para "
            "comparar todo bajo las mismas condiciones.", color=COLOR_SECUNDARIO))

        leyenda = QHBoxLayout()
        for texto, color in (("Repo original (antes)", COLOR_REPO), ("Aporte del grupo (después)", COLOR_APORTE)):
            chip = QLabel(f"  {texto}  ")
            chip.setStyleSheet(f"background: {color}; color: white; border-radius: 3px; padding: 3px;")
            leyenda.addWidget(chip)
        leyenda.addStretch()
        capa.addLayout(leyenda)

        t = tabla(["Técnica", "Tema", "Antes (repo original)", "Después (con el aporte)", ""])
        t.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        for fila, (tecnica, tema, antes, despues, destino) in enumerate(CAPACIDADES):
            t.insertRow(fila)
            nuevo = antes.startswith("No")
            t.setItem(fila, 0, celda(tecnica, alinear=Qt.AlignLeft))
            t.setItem(fila, 1, celda(tema))
            t.setItem(fila, 2, celda(antes, COLOR_SECUNDARIO if nuevo else COLOR_REPO, alinear=Qt.AlignLeft))
            t.setItem(fila, 3, celda(("Nuevo: " if nuevo else "Mejorado: ") + despues, COLOR_APORTE,
                                     alinear=Qt.AlignLeft))
            boton = QPushButton("Ver →")
            boton.clicked.connect(lambda _, d=destino: ir_a(d))
            t.setCellWidget(fila, 4, boton)
        ajustar_alto(t)
        capa.addWidget(t)

        capa.addWidget(etiqueta(
            "La GUI original también tenía umbralización espectral y clustering (<code>main.py</code>), "
            "que no están en esta copia del repositorio. Arriba se elige la imagen que usan las pestañas: "
            "los escenarios A–C tienen ground truth, así que muestran métricas; una imagen propia "
            "solo permite comparar visualmente.", color=COLOR_SECUNDARIO))
        capa.addStretch()
        lienzo = QVBoxLayout(self)
        lienzo.setContentsMargins(0, 0, 0, 0)
        lienzo.addWidget(con_scroll(contenido))


# ========================================================= pestaña Antes vs. después
class PestanaComparacion(Pestana, ConTareas):
    def __init__(self, estado):
        super().__init__(estado)
        contenido = QWidget()
        capa = QVBoxLayout(contenido)
        capa.addWidget(aviso(
            "Los 6 métodos sobre la misma imagen y con los mismos parámetros por defecto. "
            "<b>Clic izquierdo</b> sobre la imagen agrega una semilla para el crecimiento de regiones; "
            "<b>clic derecho</b> vuelve a las semillas por defecto.", "#888888"))

        arriba = QHBoxLayout()
        entrada = grupo("Entrada")
        capa_entrada = QHBoxLayout(entrada)
        self.panel_imagen = Panel("Imagen y semillas")
        self.panel_gt = Panel("Ground truth")
        self.panel_imagen.vista.clic.connect(self._clic)
        self.panel_imagen.vista.setCursor(Qt.CrossCursor)
        capa_entrada.addWidget(self.panel_imagen)
        capa_entrada.addWidget(self.panel_gt)
        arriba.addWidget(entrada, 1)

        resumen = grupo("IoU en los 3 escenarios (1 = perfecto)")
        capa_resumen = QVBoxLayout(resumen)
        self.tabla = tabla(["Método", "Origen", "A) Uniforme", "B) No uniforme", "C) Ruido", "Tiempo"])
        capa_resumen.addWidget(self.tabla)
        self.nota_tabla = etiqueta("Calculando…", color=COLOR_SECUNDARIO)
        capa_resumen.addWidget(self.nota_tabla)
        capa_resumen.addStretch()
        arriba.addWidget(resumen, 1)
        capa.addLayout(arriba)

        self.ver_errores = QCheckBox("Mostrar errores contra el ground truth "
                                     "(naranja = sobra, azul = falta)")
        self.ver_errores.toggled.connect(self._pintar)
        capa.addWidget(self.ver_errores)

        antes = grupo("Antes: métodos del repositorio original", COLOR_REPO)
        capa_antes = QHBoxLayout(antes)
        despues = grupo("Después: métodos agregados por el aporte", COLOR_APORTE)
        capa_despues = QHBoxLayout(despues)
        self.paneles = {}
        for nombre, origen, _ in metodos([]):
            panel = Panel(nombre, origen)
            self.paneles[nombre] = panel
            (capa_antes if origen == "repo" else capa_despues).addWidget(panel, 1)
        # Los 2 paneles nuevos ocupan lo mismo que 2 de los 4 de arriba; la explicación, el resto.
        self.explicacion = etiqueta("", color=COLOR_SECUNDARIO)
        self.explicacion.setAlignment(Qt.AlignTop)
        self.explicacion.setContentsMargins(10, 24, 10, 0)
        capa_despues.addWidget(self.explicacion, 2)
        capa.addWidget(antes)
        capa.addWidget(despues)

        lienzo = QVBoxLayout(self)
        lienzo.setContentsMargins(0, 0, 0, 0)
        lienzo.addWidget(con_scroll(contenido))
        self.resultados, self.gt_resultados = [], None
        self._tabla_lista = False

    # ---------------------------------------------------------- interacción
    def _clic(self, x, y, boton):
        if boton == Qt.RightButton:
            self.estado.poner_semillas(self.estado.semillas_defecto)
        else:
            self.estado.poner_semillas(self.estado.semillas + [(x, y)])

    # ---------------------------------------------------------- cálculo
    def actualizar(self):
        e = self.estado
        self.panel_imagen.mostrar(dibujar_semillas(e.gris, e.semillas),
                                  f"{e.nombre} · {len(e.semillas)} semilla(s)")
        self.panel_gt.mostrar(e.gt * 255 if e.gt is not None else None,
                              "" if e.gt is not None else "Sin ground truth: no hay métricas")
        self.ver_errores.setEnabled(e.gt is not None)
        self.resultados, self.gt_resultados = [], None  # no mezclar resultados viejos con el GT nuevo
        for panel in self.paneles.values():
            panel.pie.setText("Calculando…")

        gris, gt, semillas = e.gris, e.gt, list(e.semillas)

        def calcular(_):
            salida = []
            for nombre, origen, funcion in metodos(semillas):
                mascara, ms = cronometrar(funcion, gris, repeticiones=1)
                salida.append((nombre, origen, mascara, ms, evaluar(mascara, gt) if gt is not None else None))
            return salida

        self.lanzar("imagen", calcular, lambda r: self._recibir(r, gt))
        if not self._tabla_lista:
            self._tabla_lista = True
            self.lanzar("tabla", self._calcular_tabla, self._llenar_tabla)

    def _recibir(self, resultado, gt):
        if isinstance(resultado, Exception):
            for panel in self.paneles.values():
                panel.pie.setText(f"Error: {resultado}")
            return
        self.resultados, self.gt_resultados = resultado, gt
        self._pintar()

    def _pintar(self):
        gt = self.gt_resultados
        errores = self.ver_errores.isChecked() and gt is not None
        for nombre, _, mascara, ms, metricas in self.resultados:
            img = mapa_de_errores(mascara, gt) if errores else mascara
            self.paneles[nombre].mostrar(img, texto_metricas(metricas, ms))
        self._explicar()

    def _explicar(self):
        if not self.resultados or self.gt_resultados is None:
            self.explicacion.setText(
                "Sin ground truth no hay una respuesta correcta para medir: compará visualmente. "
                "Probá cambiar las semillas del crecimiento de regiones con clics sobre la imagen.")
            return
        iou = {n: m["IoU"] for n, o, _, _, m in self.resultados}
        repo = max((n for n, o, *_ in self.resultados if o == "repo"), key=iou.get)
        aporte = max((n for n, o, *_ in self.resultados if o == "aporte"), key=iou.get)
        diferencia_iou = iou[aporte] - iou[repo]
        if diferencia_iou > 0.01:
            veredicto = (f"El aporte mejora el resultado en <b>{diferencia_iou:+.3f}</b> de IoU. "
                         "Los métodos nuevos deciden con información <i>local</i> (fronteras entre "
                         "bloques o gradiente), por eso no dependen de que el histograma sea bimodal.")
        elif diferencia_iou < -0.01:
            veredicto = ("En el caso ideal el umbral global ya es perfecto: los métodos nuevos pierden "
                         "un poco de precisión (bordes escalonados del quadtree, suavizado de bordes) "
                         "a cambio de ser estables cuando las condiciones empeoran.")
        else:
            veredicto = "En esta imagen el mejor método de cada lado rinde parecido."
        self.explicacion.setText(
            f"<b>Mejor del repo:</b> {repo} ({iou[repo]:.3f})<br>"
            f"<b>Mejor del aporte:</b> {aporte} ({iou[aporte]:.3f})<br><br>{veredicto}")

    # ---------------------------------------------------------- tabla de 3 escenarios
    @staticmethod
    def _calcular_tabla(_):
        filas = {}
        for clave, generar in ESCENARIOS.items():
            imagen, gt = generar()
            for nombre, origen, funcion in metodos(semillas_desde_gt(gt)):
                mascara, ms = cronometrar(funcion, imagen, repeticiones=3)
                fila = filas.setdefault(nombre, {"origen": origen, "iou": [], "ms": []})
                fila["iou"].append(evaluar(mascara, gt)["IoU"])
                fila["ms"].append(ms)
        return filas

    def _llenar_tabla(self, filas):
        if isinstance(filas, Exception):
            self.nota_tabla.setText(f"Error: {filas}")
            return
        t = self.tabla
        t.setRowCount(0)
        mejores = [max(f["iou"][i] for f in filas.values()) for i in range(3)]
        for nombre, f in filas.items():
            r = t.rowCount()
            t.insertRow(r)
            color = COLOR_REPO if f["origen"] == "repo" else COLOR_APORTE
            t.setItem(r, 0, celda(nombre, color, alinear=Qt.AlignLeft))
            t.setItem(r, 1, celda("repo" if f["origen"] == "repo" else "aporte", color))
            for i, valor in enumerate(f["iou"]):
                t.setItem(r, 2 + i, celda(f"{valor:.3f}", negrita=valor == mejores[i]))
            t.setItem(r, 5, celda(f"{np.mean(f['ms']):.1f} ms"))
        # Filas de resumen: el mejor de cada lado por escenario.
        for origen, titulo, color in (("repo", "Mejor del repo", COLOR_REPO),
                                      ("aporte", "Mejor del aporte", COLOR_APORTE)):
            r = t.rowCount()
            t.insertRow(r)
            t.setItem(r, 0, celda(titulo, color, True, Qt.AlignLeft))
            t.setItem(r, 1, celda(""))
            for i in range(3):
                valor = max(f["iou"][i] for f in filas.values() if f["origen"] == origen)
                t.setItem(r, 2 + i, celda(f"{valor:.3f}", color, True))
            t.setItem(r, 5, celda(""))
        ajustar_alto(t)
        self.nota_tabla.setText(
            "Cada método con sus parámetros por defecto, iguales en los 3 escenarios. "
            "El umbral global es perfecto con luz uniforme pero cae con la rampa de luz o el ruido; "
            "split & merge y bordes + relleno se mantienen por encima de 0.9 en los tres.")


# ========================================================= pestaña Discontinuidades
class PestanaDiscontinuidades(Pestana):
    def __init__(self, estado):
        super().__init__(estado)
        contenido = QWidget()
        capa = QVBoxLayout(contenido)
        capa.addWidget(aviso(
            "<b>Nuevo.</b> El repositorio original no detectaba discontinuidades. Temas 2a–2d: puntos "
            "(Laplaciano), líneas (4 máscaras), bordes (Sobel, Prewitt, LoG, Canny) y el enfoque "
            "combinado que convierte bordes en una máscara. Para puntos y líneas elegí arriba la "
            "<i>imagen de prueba</i>.", COLOR_APORTE))

        controles = QHBoxLayout()
        self.f_puntos = self._spin(controles, "Puntos: fracción del máx.", 0.05, 1.0, 0.9, 0.05)
        self.f_lineas = self._spin(controles, "Líneas: fracción", 0.05, 1.0, 0.5, 0.05)
        self.s_log = self._spin(controles, "σ LoG", 0.5, 6.0, 2.0, 0.25)
        self.c_bajo = self._spin(controles, "Canny bajo", 0, 255, 50, 5, entero=True)
        self.c_alto = self._spin(controles, "Canny alto", 0, 255, 150, 5, entero=True)
        self.s_comb = self._spin(controles, "σ combinada", 0.5, 6.0, 1.5, 0.25)
        controles.addStretch()
        capa.addLayout(controles)

        rejilla = QGridLayout()
        nombres = ["Original", "2a. Puntos", "2b. Líneas horizontales", "2b. Líneas verticales",
                   "2b. Líneas diagonales \\", "2b. Líneas diagonales /", "2c. Sobel", "2c. Prewitt",
                   "2c. LoG (Marr-Hildreth)", "2c. Canny", "2d. Combinada → máscara",
                   "2d. Errores contra el ground truth"]
        self.paneles = {}
        for i, nombre in enumerate(nombres):
            origen = None if i == 0 else "aporte"
            self.paneles[nombre] = Panel(nombre, origen, minimo=170)
            rejilla.addWidget(self.paneles[nombre], i // 6, i % 6)
        capa.addLayout(rejilla)
        lienzo = QVBoxLayout(self)
        lienzo.setContentsMargins(0, 0, 0, 0)
        lienzo.addWidget(con_scroll(contenido))

    def _spin(self, capa, texto, minimo, maximo, valor, paso, entero=False):
        capa.addWidget(QLabel(texto))
        spin = QSpinBox() if entero else QDoubleSpinBox()
        spin.setRange(minimo, maximo)
        spin.setSingleStep(paso)
        spin.setValue(valor)
        spin.valueChanged.connect(lambda _: self.refrescar())
        capa.addWidget(spin)
        capa.addSpacing(10)
        return spin

    def actualizar(self):
        img, gt = self.estado.gris, self.estado.gt
        p = self.paneles
        p["Original"].mostrar(img, self.estado.nombre)
        p["2a. Puntos"].mostrar(detectar_puntos(img, self.f_puntos.value()), "|R| ≥ fracción · máx|R|")
        for direccion, clave in (("horizontal", "2b. Líneas horizontales"), ("vertical", "2b. Líneas verticales"),
                                 ("diagonal \\", "2b. Líneas diagonales \\"),
                                 ("diagonal /", "2b. Líneas diagonales /")):
            p[clave].mostrar(detectar_lineas(img, direccion, self.f_lineas.value()))
        p["2c. Sobel"].mostrar(bordes_sobel(img), "magnitud del gradiente")
        p["2c. Prewitt"].mostrar(bordes_prewitt(img), "magnitud del gradiente")
        p["2c. LoG (Marr-Hildreth)"].mostrar(bordes_log(img, self.s_log.value()), "cruces por cero")
        bajo, alto = self.c_bajo.value(), self.c_alto.value()
        p["2c. Canny"].mostrar(bordes_canny(img, min(bajo, alto), max(bajo, alto)), "histéresis bajo/alto")
        mascara, ms = cronometrar(segmentar_por_bordes, img, self.s_comb.value(), repeticiones=1)
        metricas = evaluar(mascara, gt) if gt is not None else None
        p["2d. Combinada → máscara"].mostrar(mascara, texto_metricas(metricas, ms))
        p["2d. Errores contra el ground truth"].mostrar(
            mapa_de_errores(mascara, gt) if gt is not None else None,
            "naranja = sobra, azul = falta" if gt is not None else "Sin ground truth")


# ========================================================= pestaña Split & merge
class PestanaSplitMerge(Pestana, ConTareas):
    def __init__(self, estado):
        super().__init__(estado)
        contenido = QWidget()
        capa = QVBoxLayout(contenido)
        capa.addWidget(aviso(
            "<b>Nuevo.</b> Tema 4c. El quadtree divide los bloques no homogéneos y después fusiona "
            "vecinos cuya frontera es débil. <b>Aporte propio:</b> los umbrales se calculan solos a "
            "partir del ruido estimado (Immerkær), así funciona en los 3 escenarios sin tocar nada.",
            COLOR_APORTE))

        controles = QHBoxLayout()
        self.auto = QCheckBox("Parámetros automáticos (según el ruido)")
        self.auto.setChecked(True)
        self.auto.toggled.connect(self._cambio_auto)
        controles.addWidget(self.auto)
        controles.addSpacing(16)
        controles.addWidget(QLabel("σ máx. para dividir"))
        self.sigma = QDoubleSpinBox()
        self.sigma.setRange(0.5, 80)
        self.sigma.setSingleStep(0.5)
        controles.addWidget(self.sigma)
        controles.addSpacing(10)
        controles.addWidget(QLabel("Umbral de fusión"))
        self.umbral = QDoubleSpinBox()
        self.umbral.setRange(1, 120)
        self.umbral.setSingleStep(1)
        controles.addWidget(self.umbral)
        self.aplicar = QPushButton("Aplicar")
        self.aplicar.clicked.connect(self.refrescar)
        controles.addWidget(self.aplicar)
        controles.addStretch()
        capa.addLayout(controles)
        self._cambio_auto(True, recalcular=False)

        fila = QHBoxLayout()
        self.p_img = Panel("Imagen")
        self.p_quad = Panel("1. División (hojas del quadtree)", "aporte")
        self.p_reg = Panel("2. Fusión (regiones)", "aporte")
        self.p_mask = Panel("3. Máscara del objeto", "aporte")
        self.p_err = Panel("Errores contra el ground truth", "aporte")
        for panel in (self.p_img, self.p_quad, self.p_reg, self.p_mask, self.p_err):
            fila.addWidget(panel)
        capa.addLayout(fila)
        self.info = etiqueta("", color=COLOR_SECUNDARIO)
        capa.addWidget(self.info)
        capa.addStretch()
        lienzo = QVBoxLayout(self)
        lienzo.setContentsMargins(0, 0, 0, 0)
        lienzo.addWidget(con_scroll(contenido))

    def _cambio_auto(self, automatico, recalcular=True):
        self.sigma.setEnabled(not automatico)
        self.umbral.setEnabled(not automatico)
        self.aplicar.setEnabled(not automatico)
        if recalcular:
            self.refrescar()

    def actualizar(self):
        img, gt = self.estado.gris, self.estado.gt
        self.p_img.mostrar(img, self.estado.nombre)
        self.info.setText("Calculando…")
        sigma = None if self.auto.isChecked() else self.sigma.value()
        umbral = None if self.auto.isChecked() else self.umbral.value()

        def calcular(_):
            (mascara, detalle), ms = cronometrar(split_and_merge, img, sigma, umbral,
                                                 devolver_detalle=True, repeticiones=1)
            return mascara, detalle, ms

        self.lanzar("sm", calcular, lambda r: self._recibir(r, img, gt))

    def _recibir(self, resultado, img, gt):
        if isinstance(resultado, Exception):
            self.info.setText(f"Error: {resultado}")
            return
        mascara, d, ms = resultado
        metricas = evaluar(mascara, gt) if gt is not None else None
        self.p_quad.mostrar(dibujar_quadtree(img, d["hojas"]), f"{d['n_hojas']} hojas")
        self.p_reg.mostrar(colorear_regiones(d["etiquetas"]), f"{d['n_regiones']} regiones")
        self.p_mask.mostrar(mascara, texto_metricas(metricas, ms))
        self.p_err.mostrar(mapa_de_errores(mascara, gt) if gt is not None else None,
                           "naranja = sobra, azul = falta" if gt is not None else "Sin ground truth")
        if self.auto.isChecked():
            for spin, valor in ((self.sigma, d["sigma_max"]), (self.umbral, d["umbral_fusion"])):
                spin.blockSignals(True)
                spin.setValue(valor)
                spin.blockSignals(False)
        mediana = (f"Ruido alto: se aplicó una mediana 5×5 {d['pasadas_mediana']} vez/veces antes de dividir. "
                   if d["pasadas_mediana"] else "")
        self.info.setText(
            f"Ruido estimado: <b>σ ≈ {d['ruido_estimado']:.1f}</b>. {mediana}"
            f"σ máx. = {d['sigma_max']:.1f} · umbral de fusión = {d['umbral_fusion']:.1f} · "
            f"{d['n_hojas']} hojas → {d['n_regiones']} regiones · {ms:.0f} ms.<br>"
            "El fondo es la región más grande que toca el borde de la imagen; todo lo demás es objeto. "
            "Probá desactivar los parámetros automáticos en el escenario C con un σ chico para ver "
            "por qué hacen falta.")


# ========================================================= pestaña Movimiento
class PestanaMovimiento(Pestana):
    VELOCIDAD = (3, 2)

    def __init__(self):
        super().__init__()
        contenido = QWidget()
        capa = QVBoxLayout(contenido)
        capa.addWidget(aviso(
            "<b>Nuevo.</b> Temas 5a y 5b. Secuencia sintética con cámara fija: 30 cuadros de fondo y "
            "luego un cuadrado que se mueve a (3, 2) px/cuadro. También funciona en vivo con la webcam.",
            COLOR_APORTE))

        controles = QHBoxLayout()
        self.reproducir = QPushButton("▶ Reproducir")
        self.reproducir.clicked.connect(self._alternar_reproduccion)
        controles.addWidget(self.reproducir)
        self.deslizador = QSlider(Qt.Horizontal)
        self.deslizador.valueChanged.connect(self._mostrar_cuadro)
        controles.addWidget(self.deslizador, 1)
        self.etiqueta_cuadro = QLabel()
        self.etiqueta_cuadro.setMinimumWidth(110)
        controles.addWidget(self.etiqueta_cuadro)
        controles.addWidget(QLabel("Umbral"))
        self.umbral = QSpinBox()
        self.umbral.setRange(1, 120)
        self.umbral.setValue(25)
        self.umbral.valueChanged.connect(lambda _: self.refrescar())
        controles.addWidget(self.umbral)
        controles.addSpacing(16)
        self.boton_camara = QPushButton("Cámara en vivo")
        self.boton_camara.setCheckable(True)
        self.boton_camara.toggled.connect(self._alternar_camara)
        controles.addWidget(self.boton_camara)
        capa.addLayout(controles)

        rejilla = QGridLayout()
        nombres = ["Cuadro t", "Diferencia t vs t−1", "Fondo (mediana temporal)", "Sustracción de fondo",
                   "MOG2 (OpenCV, en línea)", "ADI absoluta", "ADI positiva", "ADI negativa"]
        self.paneles = {}
        for i, nombre in enumerate(nombres):
            self.paneles[nombre] = Panel(nombre, None if i == 0 else "aporte", minimo=190)
            rejilla.addWidget(self.paneles[nombre], i // 4, i % 4)
        capa.addLayout(rejilla)
        self.info = etiqueta("", color=COLOR_SECUNDARIO)
        capa.addWidget(self.info)
        capa.addStretch()
        lienzo = QVBoxLayout(self)
        lienzo.setContentsMargins(0, 0, 0, 0)
        lienzo.addWidget(con_scroll(contenido))

        self.reloj = QTimer(self)
        self.reloj.setInterval(80)
        self.reloj.timeout.connect(self._avanzar)
        self.reloj_camara = QTimer(self)
        self.reloj_camara.setInterval(30)
        self.reloj_camara.timeout.connect(self._cuadro_camara)
        self.captura = None
        self.cuadros = None

    # ---------------------------------------------------------- secuencia sintética
    def actualizar(self):
        if self.boton_camara.isChecked():
            return
        u = self.umbral.value()
        if self.cuadros is None:
            self.cuadros, self.verdades = generar_secuencia(velocidad=self.VELOCIDAD)
            self.primero = next(i for i, g in enumerate(self.verdades) if g.any())
            self.fondo = fondo_por_mediana(self.cuadros)
            self.velocidad = estimar_velocidad(self.cuadros[self.primero:], self.fondo)
            self.deslizador.blockSignals(True)
            self.deslizador.setRange(1, len(self.cuadros) - 1)
            self.deslizador.setValue(self.primero + 12)
            self.deslizador.blockSignals(False)
        mog2 = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
        self.mascaras_mog = [limpiar(mog2.apply(f)) for f in self.cuadros]
        absoluta, positiva, negativa = adi(self.cuadros, u, referencia=self.primero)
        self.paneles["Fondo (mediana temporal)"].mostrar(self.fondo, "mediana por píxel de toda la secuencia")
        self.paneles["ADI absoluta"].mostrar(mapa_de_calor(absoluta), "cuántas veces cambió cada píxel")
        self.paneles["ADI positiva"].mostrar(
            mapa_de_calor(positiva),
            f"IoU {evaluar(positiva > 0, self.verdades[self.primero])['IoU']:.3f} con la posición inicial")
        self.paneles["ADI negativa"].mostrar(mapa_de_calor(negativa), "trayectoria del objeto")

        promedios = {}
        con_objeto = [i for i, g in enumerate(self.verdades) if g.any()]
        for nombre, funcion in (("Diferencia de cuadros", lambda i: diferencia(self.cuadros[i], self.cuadros[i - 1], u)),
                                ("Sustracción de fondo", lambda i: diferencia(self.cuadros[i], self.fondo, u)),
                                ("MOG2", lambda i: self.mascaras_mog[i])):
            promedios[nombre] = np.mean([evaluar(limpiar(funcion(i)) if nombre != "MOG2" else funcion(i),
                                                 self.verdades[i])["IoU"] for i in con_objeto])
        vx, vy = self.velocidad
        self.info.setText(
            "IoU promedio en los cuadros con objeto: " +
            " · ".join(f"{n} <b>{v:.3f}</b>" for n, v in promedios.items()) +
            f"<br><b>Dominio de la frecuencia</b> (correlación de fase sobre el primer plano): velocidad "
            f"estimada <b>({vx:+.2f}, {vy:+.2f})</b> px/cuadro, real {self.VELOCIDAD}.<br>"
            "La diferencia de cuadros solo ve los bordes que avanzan; MOG2 necesita calentamiento "
            "(mirá los primeros cuadros con objeto).")
        self._mostrar_cuadro(self.deslizador.value())

    def _mostrar_cuadro(self, t):
        if self.cuadros is None or self.boton_camara.isChecked():
            return
        u = self.umbral.value()
        gt = self.verdades[t]
        hay_objeto = gt.any()

        def pie(mascara):
            return texto_metricas(evaluar(mascara, gt)) if hay_objeto else "sin objeto todavía"

        d = limpiar(diferencia(self.cuadros[t], self.cuadros[t - 1], u))
        s = limpiar(diferencia(self.cuadros[t], self.fondo, u))
        self.etiqueta_cuadro.setText(f"cuadro {t} / {len(self.cuadros) - 1}")
        self.paneles["Cuadro t"].mostrar(self.cuadros[t], "con objeto" if hay_objeto else "solo fondo")
        self.paneles["Diferencia t vs t−1"].mostrar(d, pie(d))
        self.paneles["Sustracción de fondo"].mostrar(s, pie(s))
        self.paneles["MOG2 (OpenCV, en línea)"].mostrar(self.mascaras_mog[t], pie(self.mascaras_mog[t]))

    def _alternar_reproduccion(self):
        if self.reloj.isActive():
            self.reloj.stop()
            self.reproducir.setText("▶ Reproducir")
        else:
            self.reloj.start()
            self.reproducir.setText("⏸ Pausa")

    def _avanzar(self):
        siguiente = self.deslizador.value() + 1
        self.deslizador.setValue(siguiente if siguiente <= self.deslizador.maximum() else 1)

    # ---------------------------------------------------------- cámara en vivo
    def _alternar_camara(self, activa):
        if activa:
            self.reloj.stop()
            self.reproducir.setText("▶ Reproducir")
            self.captura = cv2.VideoCapture(0)
            if not self.captura.isOpened():
                self.captura = None
                self.info.setText("No se pudo abrir la webcam (índice 0).")
                self.boton_camara.setChecked(False)
                return
            self.mog2_vivo = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
            self.anterior = None
            for nombre in ("Fondo (mediana temporal)", "Sustracción de fondo", "ADI absoluta",
                           "ADI positiva", "ADI negativa"):
                self.paneles[nombre].mostrar(None, "solo en la secuencia sintética")
            for w in (self.reproducir, self.deslizador):
                w.setEnabled(False)
            self.reloj_camara.start()
        else:
            self.detener_camara()
            for w in (self.reproducir, self.deslizador):
                w.setEnabled(True)
            self.refrescar()

    def _cuadro_camara(self):
        ok, cuadro = self.captura.read() if self.captura else (False, None)
        if not ok:
            return
        u = self.umbral.value()
        gris = cv2.GaussianBlur(cv2.cvtColor(cuadro, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        mascara_mog = limpiar(self.mog2_vivo.apply(gris))
        texto = "moviendo una mano o un objeto frente a la cámara"
        if self.anterior is not None:
            d = limpiar(diferencia(gris, self.anterior, u))
            dx, dy, _ = correlacion_de_fase(self.anterior, gris)
            texto = f"desplazamiento global (correlación de fase): ({dx:+.1f}, {dy:+.1f}) px"
            self.paneles["Diferencia t vs t−1"].mostrar(d, "")
        self.anterior = gris
        self.paneles["Cuadro t"].mostrar(cuadro, "webcam")
        self.paneles["MOG2 (OpenCV, en línea)"].mostrar(mascara_mog, "")
        self.info.setText(texto)

    def detener_camara(self):
        self.reloj_camara.stop()
        if self.captura is not None:
            self.captura.release()
            self.captura = None

    def detener(self):
        self.reloj.stop()
        self.reproducir.setText("▶ Reproducir")
        if self.boton_camara.isChecked():
            self.boton_camara.setChecked(False)


# ========================================================= pestaña Mejoras al código
class PestanaMejoras(Pestana, ConTareas):
    def __init__(self, estado):
        super().__init__(estado)
        contenido = QWidget()
        capa = QVBoxLayout(contenido)
        capa.addWidget(aviso(
            "<b>Mejorado.</b> Dos funciones del repositorio original se reescribieron sin cambiar lo que "
            "devuelven. Esta pestaña corre la versión original (copiada textual) y la nueva sobre la "
            "imagen elegida, y verifica píxel a píxel que el resultado sea idéntico.", COLOR_REPO))

        explicacion = QHBoxLayout()
        for titulo, texto in (
                ("OtsuAndOptimal.local_otsu / local_optimal",
                 "<b>Antes:</b> por cada ventana escribía un PNG en disco y lo volvía a leer.<br>"
                 "<b>Ahora:</b> la ventana se procesa en memoria (la clase acepta arrays NumPy)."),
                ("RegionGrowingSegmentation.region_growing",
                 "<b>Antes:</b> recorría píxel por píxel con una cola de Python.<br>"
                 "<b>Ahora:</b> como cada píxel se compara contra la <i>semilla</i>, la región es la "
                 "componente conexa de {distancia ≤ T} que contiene a la semilla: un solo "
                 "<code>cv2.connectedComponents</code>.<br>"
                 "<b>Bug corregido:</b> una semilla fuera de la imagen hacía fallar la función o "
                 "reutilizaba la región de la semilla anterior.")):
            caja = grupo(titulo)
            QVBoxLayout(caja).addWidget(etiqueta(texto))
            explicacion.addWidget(caja, 1)
        capa.addLayout(explicacion)

        fila_boton = QHBoxLayout()
        self.boton = QPushButton("Medir con la imagen actual")
        self.boton.clicked.connect(self._medir)
        fila_boton.addWidget(self.boton)
        self.estado_medicion = etiqueta("", color=COLOR_SECUNDARIO)
        fila_boton.addWidget(self.estado_medicion, 1)
        capa.addLayout(fila_boton)

        self.tabla = tabla(["Prueba", "Original", "Aporte", "Aceleración", "¿Mismo resultado?"])
        capa.addWidget(self.tabla)

        paneles = QHBoxLayout()
        self.p_orig = Panel("Crecimiento de regiones: original", "repo")
        self.p_nuevo = Panel("Crecimiento de regiones: aporte", "aporte")
        self.p_dif = Panel("Diferencia entre ambos (blanco = distinto)")
        for p in (self.p_orig, self.p_nuevo, self.p_dif):
            paneles.addWidget(p)
        capa.addLayout(paneles)
        capa.addStretch()
        lienzo = QVBoxLayout(self)
        lienzo.setContentsMargins(0, 0, 0, 0)
        lienzo.addWidget(con_scroll(contenido))

    def actualizar(self):
        self.estado_medicion.setText(
            f"Imagen: {self.estado.nombre}, {len(self.estado.semillas)} semilla(s). La versión original "
            "es lenta a propósito: puede tardar unos segundos.")

    def _medir(self):
        e = self.estado
        gris, color, semillas = e.gris, e.color, list(e.semillas)
        self.boton.setEnabled(False)
        self.tabla.setRowCount(0)
        for p in (self.p_orig, self.p_nuevo, self.p_dif):
            p.mostrar(None)

        def calcular(avisar):
            pruebas = [
                ("Otsu local, ventana de 16 px (gris)",
                 lambda: local_otsu_original(gris, 16), lambda: OtsuAndOptimal(gris).local_otsu(16)),
                ("Crecimiento de regiones (gris)",
                 lambda: region_growing_original(gris, semillas, 50, False),
                 lambda: modulo_rg.region_growing(gris, semillas, 50, False)),
                ("Crecimiento de regiones (color, RGB)",
                 lambda: region_growing_original(color, semillas, 40, False),
                 lambda: modulo_rg.region_growing(color, semillas, 40, False)),
                ("Crecimiento de regiones (color, CIELUV)",
                 lambda: region_growing_original(color, semillas, 40, True),
                 lambda: modulo_rg.region_growing(color, semillas, 40, True)),
            ]
            for titulo, original, nuevo in pruebas:
                avisar(("midiendo", titulo))
                r_orig, t_orig = cronometrar(original, repeticiones=1)
                r_nuevo, t_nuevo = cronometrar(nuevo, repeticiones=3)
                avisar(("fila", titulo, t_orig, t_nuevo, bool(np.array_equal(r_orig, r_nuevo))))
                if titulo == "Crecimiento de regiones (gris)":
                    avisar(("paneles", r_orig, r_nuevo))

            # Bug de semillas: la primera semilla fuera de la imagen.
            avisar(("midiendo", "semilla fuera de la imagen"))
            fuera = [(-10, -10)] + semillas
            try:
                region_growing_original(gris, fuera, 50, False)
                antes = "sin error"
            except Exception as ex:
                antes = f"falla: {type(ex).__name__}"
            r = modulo_rg.region_growing(gris, fuera, 50, False)
            despues = "funciona (la semilla se ignora)" if (r > 0).any() else "funciona"
            avisar(("bug", antes, despues))
            return True

        self.lanzar("medir", calcular, self._terminar, self._progreso)

    def _progreso(self, dato):
        tipo = dato[0]
        t = self.tabla
        if tipo == "midiendo":
            self.estado_medicion.setText(f"Midiendo: {dato[1]}…")
        elif tipo == "fila":
            _, titulo, t_orig, t_nuevo, igual = dato
            r = t.rowCount()
            t.insertRow(r)
            t.setItem(r, 0, celda(titulo, alinear=Qt.AlignLeft))
            t.setItem(r, 1, celda(f"{t_orig:.1f} ms", COLOR_REPO))
            t.setItem(r, 2, celda(f"{t_nuevo:.1f} ms", COLOR_APORTE))
            t.setItem(r, 3, celda(f"{t_orig / max(t_nuevo, 1e-6):.0f}× más rápido", negrita=True))
            t.setItem(r, 4, celda("Sí, idéntico" if igual else "NO", "#1a7f37" if igual else "#cf222e", True))
            ajustar_alto(t)
        elif tipo == "paneles":
            _, r_orig, r_nuevo = dato
            distinto = (r_orig != r_nuevo).astype(np.uint8) * 255
            self.p_orig.mostrar((r_orig > 0).astype(np.uint8) * 255, f"{int((r_orig > 0).sum())} píxeles")
            self.p_nuevo.mostrar((r_nuevo > 0).astype(np.uint8) * 255, f"{int((r_nuevo > 0).sum())} píxeles")
            self.p_dif.mostrar(distinto, f"{int((distinto > 0).sum())} píxeles distintos")
        elif tipo == "bug":
            _, antes, despues = dato
            r = t.rowCount()
            t.insertRow(r)
            t.setItem(r, 0, celda("Primera semilla fuera de la imagen", alinear=Qt.AlignLeft))
            t.setItem(r, 1, celda(antes, COLOR_REPO))
            t.setItem(r, 2, celda(despues, COLOR_APORTE))
            t.setItem(r, 3, celda("—"))
            t.setItem(r, 4, celda("Bug corregido", "#1a7f37", True))
            ajustar_alto(t)

    def _terminar(self, resultado):
        self.boton.setEnabled(True)
        if isinstance(resultado, Exception):
            self.estado_medicion.setText(f"Error: {resultado}")
        else:
            self.estado_medicion.setText("Listo. Los tiempos dependen de la imagen y de la cantidad de semillas.")


# ================================================================ ventana principal
class VentanaPrincipal(QMainWindow):
    FUENTES = [("A_iluminacion_uniforme", TITULOS["A_iluminacion_uniforme"]),
               ("B_iluminacion_no_uniforme", TITULOS["B_iluminacion_no_uniforme"]),
               ("C_ruido_fuerte", TITULOS["C_ruido_fuerte"]),
               ("prueba", "Imagen de prueba (puntos y líneas)")]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Segmentación de imágenes: antes y después del aporte")
        self.resize(1400, 900)
        self.estado = Estado()

        central = QWidget()
        capa = QVBoxLayout(central)

        barra = QHBoxLayout()
        barra.addWidget(etiqueta("Segmentación de imágenes", 14, True, ajustar=False))
        barra.addSpacing(24)
        barra.addWidget(QLabel("Imagen:"))
        self.selector = QComboBox()
        for clave, titulo in self.FUENTES:
            self.selector.addItem(titulo, clave)
        self.selector.currentIndexChanged.connect(self._elegir_fuente)
        barra.addWidget(self.selector)
        abrir = QPushButton("Abrir imagen…")
        abrir.clicked.connect(self._abrir)
        barra.addWidget(abrir)
        barra.addStretch()
        for texto, color in (("antes (repo)", COLOR_REPO), ("después (aporte)", COLOR_APORTE)):
            chip = QLabel(f"  {texto}  ")
            chip.setStyleSheet(f"background: {color}; color: white; border-radius: 3px; padding: 2px;")
            barra.addWidget(chip)
        capa.addLayout(barra)

        self.pestanas = QTabWidget()
        self.inicio = PestanaInicio(self.pestanas.setCurrentIndex)
        self.lista = [
            (self.inicio, "Inicio"),
            (PestanaComparacion(self.estado), "Antes vs. después"),
            (PestanaDiscontinuidades(self.estado), "Discontinuidades"),
            (PestanaSplitMerge(self.estado), "Split && merge"),
            (PestanaMovimiento(), "Movimiento"),
            (PestanaMejoras(self.estado), "Mejoras al código"),
        ]
        for pestana, titulo in self.lista:
            self.pestanas.addTab(pestana, titulo)  # "&&" se muestra como "&" en las pestañas
        self.pestanas.currentChanged.connect(self._cambio_pestana)
        capa.addWidget(self.pestanas, 1)
        self.setCentralWidget(central)
        self._anterior = 0
        self.estado.cargar_escenario("A_iluminacion_uniforme")

    def _elegir_fuente(self, indice):
        clave = self.selector.itemData(indice)
        if clave == "prueba":
            self.estado.cargar_prueba()
        elif clave in ESCENARIOS:
            self.estado.cargar_escenario(clave)
        elif clave:  # archivo
            self._cargar_archivo(clave)

    def _abrir(self):
        ruta, _ = QFileDialog.getOpenFileName(self, "Abrir imagen", "",
                                              "Imágenes (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if not ruta:
            return
        # Un solo ítem de archivo al final del selector, reemplazado en cada apertura.
        if self.selector.count() > len(self.FUENTES):
            self.selector.removeItem(len(self.FUENTES))
        self.selector.addItem(f"Archivo: {os.path.basename(ruta)}", ruta)
        self.selector.setCurrentIndex(self.selector.count() - 1)

    def _cargar_archivo(self, ruta):
        try:
            self.estado.cargar_archivo(ruta)
        except Exception as e:
            self.statusBar().showMessage(str(e), 6000)

    def _cambio_pestana(self, indice):
        anterior = self.lista[self._anterior][0]
        if anterior is not self.lista[indice][0]:
            anterior.detener()
        self._anterior = indice

    def closeEvent(self, evento):
        for pestana, _ in self.lista:
            pestana.detener()
            for tarea in list(getattr(pestana, "_tareas", ())):
                tarea.wait(3000)
        super().closeEvent(evento)


def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 9))
    ventana = VentanaPrincipal()

    # PyQt5 cierra la app ante cualquier excepción no atrapada en un slot: mejor avisar y seguir.
    def informar(tipo, valor, traza):
        sys.__excepthook__(tipo, valor, traza)
        ventana.statusBar().showMessage(f"Error: {tipo.__name__}: {valor}", 10000)

    sys.excepthook = informar
    ventana.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
