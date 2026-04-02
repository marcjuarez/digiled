from dataclasses import dataclass
import numpy as np

@dataclass
class AxisCalib:
    p_min: tuple[float, float]
    p_max: tuple[float, float]
    v_min: float
    v_max: float
    is_log: bool
    invert_pixel: bool

    def _get_p(self, point: tuple[float, float]) -> float:
        # If invert_pixel is True, we are looking at the Y axis (index 1)
        # Otherwise, we are looking at the X axis (index 0)
        return point[1] if self.invert_pixel else point[0]

    def pixel_to_val(self, p_val: float) -> float:
        p0 = self._get_p(self.p_min)
        p1 = self._get_p(self.p_max)
        
        if p1 == p0: 
            return self.v_min
            
        t = (p_val - p0) / (p1 - p0)

        if self.is_log:
            vmin = max(self.v_min, 1e-12)
            vmax = max(self.v_max, 1e-12)
            log_v = np.log10(vmin) + t * (np.log10(vmax) - np.log10(vmin))
            return 10**log_v
        else:
            return self.v_min + t * (self.v_max - self.v_min)

    def val_to_pixel(self, v_val: float) -> float:
        p0 = self._get_p(self.p_min)
        p1 = self._get_p(self.p_max)
        
        if self.is_log:
            vmin = max(self.v_min, 1e-12)
            vmax = max(self.v_max, 1e-12)
            v = max(v_val, 1e-12)
            if vmax == vmin:
                t = 0
            else:
                t = (np.log10(v) - np.log10(vmin)) / (np.log10(vmax) - np.log10(vmin))
        else:
            if self.v_max == self.v_min:
                t = 0
            else:
                t = (v_val - self.v_min) / (self.v_max - self.v_min)

        return p0 + t * (p1 - p0)


@dataclass
class GraphCalib:
    x: AxisCalib
    y: AxisCalib

    def pixel_to_xy(self, px: float, py: float) -> tuple[float, float]:
        return (self.x.pixel_to_val(px), self.y.pixel_to_val(py))

    def xy_to_pixel(self, x_val: float, y_val: float) -> tuple[float, float]:
        return (self.x.val_to_pixel(x_val), self.y.val_to_pixel(y_val))


def fit_poly_degree3(x: np.ndarray, y: np.ndarray) -> dict:
    """Fits a 3rd degree polynomial to the given x and y data arrays."""
    # polyfit returns highest power first: a*x^3 + b*x^2 + c*x + d
    coeffs = np.polyfit(x, y, 3)
    
    # Calculate RMSE (Root Mean Square Error)
    p = np.poly1d(coeffs)
    y_pred = p(x)
    rmse = np.sqrt(np.mean((y - y_pred)**2))
    
    # Ensure all values are standard Python floats for JSON serialization later
    return {
        "coeffs_high_to_low": [float(c) for c in coeffs],
        "rmse": float(rmse),
        "x_min": float(np.min(x)),
        "x_max": float(np.max(x))
    }


def eval_poly(coeffs: list[float], xs: np.ndarray) -> np.ndarray:
    """Evaluates the polynomial at the given x points."""
    p = np.poly1d(coeffs)
    return p(xs)