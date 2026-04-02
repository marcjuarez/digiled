from pathlib import Path
import numpy as np

from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QBrush, QPen, QPixmap, QPainter, QPainterPath
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGraphicsEllipseItem,
    QGraphicsPixmapItem,
    QGraphicsPathItem,
    QGraphicsScene,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView
)

from digitizer_math import AxisCalib, GraphCalib, fit_poly_degree3, eval_poly
from digitizer_io import save_points_csv, save_fit_json


class ClickPoint:
    def __init__(self, px: float, py: float):
        self.px = px
        self.py = py


# ----------------------------- View -----------------------------
class ImageView(QGraphicsView):
    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHints(self.renderHints() | QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setCursor(Qt.CrossCursor)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

        self._cb = None
        self._pan = False
        self._last = None

    def set_click_callback(self, cb):
        self._cb = cb

    def wheelEvent(self, e):
        factor = 1.25 if e.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)

    def mousePressEvent(self, e):
        if e.button() in (Qt.RightButton, Qt.MiddleButton):
            self._pan = True
            self._last = e.position()
            self.setCursor(Qt.ClosedHandCursor)
            e.accept()
            return

        if e.button() == Qt.LeftButton and self._cb:
            p = self.mapToScene(e.position().toPoint())
            self._cb(float(p.x()), float(p.y()))
            e.accept()
            return

        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._pan and self._last is not None:
            d = e.position() - self._last
            self._last = e.position()
            self.translate(d.x(), d.y())
            e.accept()
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if e.button() in (Qt.RightButton, Qt.MiddleButton) and self._pan:
            self._pan = False
            self._last = None
            self.setCursor(Qt.CrossCursor)
            e.accept()
            return
        super().mouseReleaseEvent(e)


# ----------------------------- Main Window -----------------------------
class DigitizerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simple Curve Digitizer")

        # Graphic scene setup
        self.scene = QGraphicsScene(self)
        self.view = ImageView(self.scene, self)
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.view.set_click_callback(self.on_click)

        self.image_path = None
        self.pixmap_item = None

        # Digitizer state
        self.calib_stage = 0
        self.calib_clicks = []
        self.curve_clicks = []
        self.calib_items = []
        self.curve_items = []
        self.fit_item = None
        
        # Math tracking
        self.current_coeffs = None  
        self.x_min_fit = None
        self.x_max_fit = None

        # UI Elements (Right Panel)
        self.btn_load = QPushButton("Load Image...")
        self.btn_load.clicked.connect(self.load_image)

        self.btn_reset = QPushButton("Reset Points")
        self.btn_reset.clicked.connect(self.reset_all)

        self.mode = QComboBox()
        self.mode.addItems(["Calibrate Axes (4 clicks: Xmin, Xmax, Ymin, Ymax)", "Capture Curve"])

        self.chk_x_log = QCheckBox("X log")
        self.chk_y_log = QCheckBox("Y log")
        self.xmin = QLineEdit("0")
        self.xmax = QLineEdit("1")
        self.ymin = QLineEdit("0")
        self.ymax = QLineEdit("1")

        self.chk_overlay = QCheckBox("Show Overlay (Red Curve)")
        self.chk_overlay.setChecked(True)
        self.chk_overlay.stateChanged.connect(self.toggle_overlay)

        self.btn_fit = QPushButton("Calculate Fit (3rd Degree)")
        self.btn_fit.clicked.connect(self.fit_curve)

        self.btn_export = QPushButton("Export Points & Fit...")
        self.btn_export.clicked.connect(self.export_curve_dialog)

        self.poly_lbl = QLabel("Polynomial: —")
        self.poly_lbl.setWordWrap(True)
        self.poly_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.stats_lbl = QLabel("RMSE: — | X Range: —")
        self.stats_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.info = QLabel("Calibration: 0/4 | Curve Points: 0")
        self.status = QLabel("Controls: Wheel (Zoom) | Right/Middle Click (Pan) | Left Click (Mark).")

        # Points Table Setup
        self.points_table = QTableWidget()
        self.points_table.setColumnCount(2)
        self.points_table.setHorizontalHeaderLabels(["X", "Y"])
        self.points_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.points_table.setEditTriggers(QTableWidget.NoEditTriggers)

        # Calculator Setup
        self.calc_mode = QComboBox()
        self.calc_mode.addItems(["Input X, find Y", "Input Y, find X"])
        self.calc_input = QLineEdit("0")
        self.calc_input.returnPressed.connect(self.calculate_value)
        
        self.btn_calc = QPushButton("Calculate")
        self.btn_calc.clicked.connect(self.calculate_value)
        
        self.btn_clear_calc = QPushButton("Clear")
        self.btn_clear_calc.clicked.connect(self.clear_calculator)
        
        self.calc_result = QLabel("Result: —")
        self.calc_result.setTextInteractionFlags(Qt.TextSelectableByMouse)
        font = self.calc_result.font()
        font.setBold(True)
        self.calc_result.setFont(font)

        self._build_layout()

    def _build_layout(self):
        # Left Panel (Image)
        left = QWidget()
        l = QVBoxLayout(left)
        l.setContentsMargins(0, 0, 0, 0)
        l.addWidget(self.view, 1)

        # Right Panel (Controls)
        right = QWidget()
        r = QVBoxLayout(right)
        r.setContentsMargins(8, 8, 8, 8)

        g_in = QGroupBox("Input")
        f1 = QFormLayout(g_in)
        f1.addRow(self.btn_load)
        f1.addRow(self.btn_reset)
        f1.addRow("Mode", self.mode)
        r.addWidget(g_in)

        g_cal = QGroupBox("Real Axis Values")
        f2 = QFormLayout(g_cal)
        f2.addRow("X min", self.xmin)
        f2.addRow("X max", self.xmax)
        f2.addRow("Y min", self.ymin)
        f2.addRow("Y max", self.ymax)
        
        sc = QWidget()
        scl = QHBoxLayout(sc)
        scl.setContentsMargins(0, 0, 0, 0)
        scl.addWidget(self.chk_x_log)
        scl.addWidget(self.chk_y_log)
        f2.addRow("Log Scale", sc)
        r.addWidget(g_cal)

        g_res = QGroupBox("Results")
        vr = QVBoxLayout(g_res)
        vr.addWidget(self.poly_lbl)
        vr.addWidget(self.stats_lbl)
        r.addWidget(g_res)

        r.addWidget(self.info)

        g_act = QGroupBox("Actions")
        va = QVBoxLayout(g_act)
        va.addWidget(self.chk_overlay)
        va.addWidget(self.btn_fit)
        va.addWidget(self.btn_export)
        r.addWidget(g_act)

        r.addWidget(self.points_table)

        g_calc = QGroupBox("Curve Calculator")
        fc = QFormLayout(g_calc)
        fc.addRow("Find", self.calc_mode)
        fc.addRow("Value", self.calc_input)
        
        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.addWidget(self.btn_calc)
        btn_layout.addWidget(self.btn_clear_calc)
        fc.addRow(btn_layout)
        
        fc.addRow(self.calc_result)
        r.addWidget(g_calc)

        # Main Layout
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1) # Gives more space to the image
        splitter.setSizes([800, 350])

        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(splitter, 1)
        outer.addWidget(self.status)
        self.setCentralWidget(root)

    @staticmethod
    def to_float(text: str) -> float:
        return float(text.strip().replace(",", "."))

    def update_info(self):
        self.info.setText(f"Calibration: {self.calib_stage}/4 | Curve Points: {len(self.curve_clicks)}")

    def _refresh_table(self):
        """Updates the visual table with the calculated real X,Y points."""
        self.points_table.setRowCount(0)
        if self.calib_stage < 4:
            return  
        
        try:
            pts = self.curve_points_xy()
            self.points_table.setRowCount(len(pts))
            for i, (x, y) in enumerate(pts):
                self.points_table.setItem(i, 0, QTableWidgetItem(f"{x:.4f}"))
                self.points_table.setItem(i, 1, QTableWidgetItem(f"{y:.4f}"))
        except ValueError:
            pass 

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not path:
            return

        pix = QPixmap(path)
        if pix.isNull():
            QMessageBox.critical(self, "Error", "Could not load the image.")
            return

        self.reset_all()
        self.image_path = path
        self.pixmap_item = QGraphicsPixmapItem(pix)
        self.scene.addItem(self.pixmap_item)
        self.scene.setSceneRect(QRectF(pix.rect()))
        self.view.resetTransform()
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def on_click(self, x: float, y: float):
        if self.pixmap_item is None:
            return

        if self.mode.currentIndex() == 0:  
            if self.calib_stage >= 4:
                QMessageBox.information(self, "Notice", "Calibration complete. Switch to 'Capture Curve' mode.")
                return
            
            self.calib_clicks.append(ClickPoint(x, y))
            color = Qt.yellow if self.calib_stage < 2 else Qt.magenta
            self._draw_point(x, y, color, store="calib")
            self.calib_stage += 1
            self._clear_fit()
        
        else:  
            if self.calib_stage < 4:
                QMessageBox.warning(self, "Attention", "You must make the 4 calibration clicks first.")
                self.mode.setCurrentIndex(0)
                return
                
            self.curve_clicks.append(ClickPoint(x, y))
            self._draw_point(x, y, Qt.cyan, store="curve")
            self._clear_fit()

        self.update_info()
        self._refresh_table()

    def _draw_point(self, x: float, y: float, color, store: str):
        r = 4.0
        it = QGraphicsEllipseItem(x - r, y - r, 2 * r, 2 * r)
        it.setBrush(QBrush(color))
        it.setPen(QPen(Qt.black))
        it.setZValue(10)
        self.scene.addItem(it)
        if store == "calib":
            self.calib_items.append(it)
        else:
            self.curve_items.append(it)

    def build_calib(self) -> GraphCalib:
        if self.calib_stage < 4:
            raise ValueError("Incomplete calibration.")

        xmin = self.to_float(self.xmin.text())
        xmax = self.to_float(self.xmax.text())
        ymin = self.to_float(self.ymin.text())
        ymax = self.to_float(self.ymax.text())

        c = self.calib_clicks
        x_axis = AxisCalib(
            p_min=(c[0].px, c[0].py), p_max=(c[1].px, c[1].py),
            v_min=xmin, v_max=xmax, is_log=self.chk_x_log.isChecked(), invert_pixel=False
        )
        y_axis = AxisCalib(
            p_min=(c[2].px, c[2].py), p_max=(c[3].px, c[3].py),
            v_min=ymin, v_max=ymax, is_log=self.chk_y_log.isChecked(), invert_pixel=True
        )
        return GraphCalib(x=x_axis, y=y_axis)

    def curve_points_xy(self) -> list[tuple[float, float]]:
        calib = self.build_calib()
        pts = []
        for p in self.curve_clicks:
            xx, yy = calib.pixel_to_xy(p.px, p.py)
            pts.append((xx, yy))
        pts.sort(key=lambda t: t[0])
        return pts

    def fit_curve(self):
        if self.calib_stage < 4 or len(self.curve_clicks) < 4:
            QMessageBox.warning(self, "Notice", "You need to calibrate axes and at least 4 points to fit a 3rd degree polynomial.")
            return

        try:
            pts = self.curve_points_xy()
            x = np.array([p[0] for p in pts], dtype=float)
            y = np.array([p[1] for p in pts], dtype=float)

            fit = fit_poly_degree3(x, y)
            coeffs = fit["coeffs_high_to_low"]
            
            # Save coefficients and bounds for the calculator
            self.current_coeffs = coeffs  
            self.x_min_fit = fit['x_min']
            self.x_max_fit = fit['x_max']
            
            a, b, c, d = coeffs

            self.poly_lbl.setText(f"Polynomial: y = ({a:.8g})·x³ + ({b:.8g})·x² + ({c:.8g})·x + ({d:.8g})")
            self.stats_lbl.setText(f"RMSE: {fit['rmse']:.6g} | X Range: [{fit['x_min']:.6g}, {fit['x_max']:.6g}]")

            if self.pixmap_item is not None:
                self.draw_overlay(coeffs)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to calculate: {str(e)}")

    def draw_overlay(self, coeffs: list[float], n_samples: int = 600):
        self._clear_fit()
        calib = self.build_calib()
        pts = self.curve_points_xy()

        x_min = float(min(p[0] for p in pts))
        x_max = float(max(p[0] for p in pts))
        xs = np.linspace(x_min, x_max, int(n_samples))
        ys = eval_poly(coeffs, xs)

        path = QPainterPath()
        started = False

        for x, y in zip(xs, ys):
            px, py = calib.xy_to_pixel(float(x), float(y))
            if not started:
                path.moveTo(px, py)
                started = True
            else:
                path.lineTo(px, py)

        it = QGraphicsPathItem(path)
        it.setPen(QPen(Qt.red, 2))
        it.setZValue(20)
        it.setVisible(self.chk_overlay.isChecked())
        self.scene.addItem(it)
        self.fit_item = it

    def toggle_overlay(self):
        if self.fit_item:
            self.fit_item.setVisible(self.chk_overlay.isChecked())

    def _clear_fit(self):
        if self.fit_item:
            try:
                self.scene.removeItem(self.fit_item)
            except Exception:
                pass
            self.fit_item = None

    def reset_all(self):
        self.calib_stage = 0
        self.calib_clicks.clear()
        self.curve_clicks.clear()
        
        self.current_coeffs = None
        self.x_min_fit = None
        self.x_max_fit = None

        for it in self.calib_items + self.curve_items:
            try:
                self.scene.removeItem(it)
            except Exception:
                pass
        
        self.calib_items.clear()
        self.curve_items.clear()
        self._clear_fit()

        if self.pixmap_item:
            self.scene.removeItem(self.pixmap_item)
            self.pixmap_item = None
            
        self.image_path = None
        self.poly_lbl.setText("Polynomial: —")
        self.stats_lbl.setText("RMSE: — | X Range: —")
        
        self.clear_calculator()
        
        self.mode.setCurrentIndex(0)
        self.update_info()
        self._refresh_table()

    def clear_calculator(self):
        """Clears the text field and calculator result"""
        self.calc_input.setText("")
        self.calc_result.setText("Result: —")
        self.calc_input.setFocus() 

    def calculate_value(self):
        if self.current_coeffs is None or self.x_min_fit is None or self.x_max_fit is None:
            QMessageBox.warning(self, "Notice", "You must click 'Calculate Fit' first before using the calculator.")
            return
            
        try:
            val = self.to_float(self.calc_input.text())
            a, b, c, d = self.current_coeffs
            
            if self.calc_mode.currentIndex() == 0:
                # Mode: Input X, find Y
                # Prevent calculating X values outside the bounds of our clicked points
                if val < self.x_min_fit or val > self.x_max_fit:
                    self.calc_result.setText("Result Y: Out of bounds")
                    QMessageBox.warning(self, "Out of Bounds", f"X must be between {self.x_min_fit:.4g} and {self.x_max_fit:.4g}")
                    return

                y = (a * val**3) + (b * val**2) + (c * val) + d
                self.calc_result.setText(f"Result Y = {y:.6g}")
                
            else:
                # Mode: Input Y, find X
                poly = np.poly1d([a, b, c, d - val])
                roots = poly.roots
                
                # Filter to only get real roots that also fall inside our X boundaries
                valid_roots = [
                    r.real for r in roots 
                    if abs(r.imag) < 1e-8 and self.x_min_fit <= r.real <= self.x_max_fit
                ]
                
                if not valid_roots:
                    self.calc_result.setText("Result X: Out of bounds")
                    QMessageBox.warning(self, "Out of Bounds", "No corresponding X value exists for this Y within the captured curve.")
                else:
                    # Pick the one closest to the center if there are somehow multiple inside the bounds
                    mid_x = (self.x_min_fit + self.x_max_fit) / 2.0
                    best_root = min(valid_roots, key=lambda r: abs(r - mid_x))
                    self.calc_result.setText(f"Result X = {best_root:.6g}")
                    
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Invalid input or math error: {e}")

    def export_curve_dialog(self):
        if self.calib_stage < 4 or len(self.curve_clicks) < 2:
            QMessageBox.warning(self, "Notice", "Not enough data to export.")
            return

        base_name = "digitized_curve"
        path, _ = QFileDialog.getSaveFileName(self, "Save Points CSV", base_name, "CSV (*.csv)")
        
        if not path:
            return

        try:
            pts = self.curve_points_xy()
            save_points_csv(path, pts, header=("x", "y"))
            
            if len(pts) >= 4:
                x = np.array([p[0] for p in pts], dtype=float)
                y = np.array([p[1] for p in pts], dtype=float)
                fit = fit_poly_degree3(x, y)
                
                json_path = str(Path(path).with_suffix(".json"))
                save_fit_json(json_path, {"points": pts, "fit": fit})
                
                txt_path = str(Path(path).with_suffix(".txt"))
                coeffs = fit["coeffs_high_to_low"]
                a, b, c, d = coeffs
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write("Polynomial Fit (3rd Degree)\n")
                    f.write("---------------------------\n")
                    f.write(f"Equation: y = ({a:.8g})*x^3 + ({b:.8g})*x^2 + ({c:.8g})*x + ({d:.8g})\n")
                    f.write(f"RMSE:     {fit['rmse']:.6g}\n")
                    f.write(f"X Range:  [{fit['x_min']:.6g}, {fit['x_max']:.6g}]\n")

                QMessageBox.information(self, "Success", f"Exported:\n{path}\n{json_path}\n{txt_path}")
            else:
                QMessageBox.information(self, "Success", f"Points exported:\n{path}")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error exporting: {str(e)}")