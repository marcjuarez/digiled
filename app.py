import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates

# Importamos tu módulo matemático
from digitizer_math import AxisCalib, GraphCalib, fit_poly_degree3, eval_poly

st.set_page_config(layout="wide", page_title="Digitalizador Web")

# --- 1. INICIALIZAR MEMORIA (STATE) ---
if 'calib_clicks' not in st.session_state:
    st.session_state.calib_clicks = []
if 'curve_clicks' not in st.session_state:
    st.session_state.curve_clicks = []
if 'last_click' not in st.session_state:
    st.session_state.last_click = None

def reset_state():
    st.session_state.calib_clicks = []
    st.session_state.curve_clicks = []
    st.session_state.last_click = None

def build_calib(xmin, xmax, ymin, ymax, x_log, y_log):
    if len(st.session_state.calib_clicks) < 4:
        return None
    c = st.session_state.calib_clicks
    x_axis = AxisCalib(p_min=c[0], p_max=c[1], v_min=xmin, v_max=xmax, is_log=x_log, invert_pixel=False)
    y_axis = AxisCalib(p_min=c[2], p_max=c[3], v_min=ymin, v_max=ymax, is_log=y_log, invert_pixel=True)
    return GraphCalib(x=x_axis, y=y_axis)

def draw_markers_on_image(img_pil, calib, coeffs):
    """Pinta los puntos y la curva roja sobre la foto"""
    img_draw = img_pil.copy()
    draw = ImageDraw.Draw(img_draw)
    r = 4 # Radio del punto
    
    # 1. Pintar clics de calibración
    for i, (x, y) in enumerate(st.session_state.calib_clicks):
        color = "yellow" if i < 2 else "magenta"
        draw.ellipse((x-r, y-r, x+r, y+r), fill=color, outline="black")
        
    # 2. Pintar clics de la curva
    for (x, y) in st.session_state.curve_clicks:
        draw.ellipse((x-r, y-r, x+r, y+r), fill="cyan", outline="black")
        
    # 3. Pintar línea roja del polinomio
    if coeffs is not None and calib is not None and len(st.session_state.curve_clicks) >= 2:
        real_pts = [calib.pixel_to_xy(px, py) for px, py in st.session_state.curve_clicks]
        x_min = min(p[0] for p in real_pts)
        x_max = max(p[0] for p in real_pts)
        
        xs = np.linspace(x_min, x_max, 300)
        ys = eval_poly(coeffs, xs)
        
        line_pts = []
        for rx, ry in zip(xs, ys):
            px, py = calib.xy_to_pixel(rx, ry)
            line_pts.append((px, py))
            
        draw.line(line_pts, fill="red", width=3)
        
    return img_draw

# --- 2. BARRA LATERAL (CONTROLES) ---
with st.sidebar:
    st.title("Controles")
    uploaded_file = st.file_uploader("Cargar Imagen", type=["png", "jpg", "jpeg", "bmp"])
    
    st.button("Resetear Puntos", on_click=reset_state)
    mode = st.radio("Modo de Clic", ["Calibrar (4 Clics)", "Capturar Curva"])
    
    st.divider()
    st.subheader("Valores Reales de Ejes")
    col1, col2 = st.columns(2)
    xmin = col1.number_input("X Min", value=0.0)
    xmax = col2.number_input("X Max", value=1.0)
    ymin = col1.number_input("Y Min", value=0.0)
    ymax = col2.number_input("Y Max", value=1.0)
    
    col3, col4 = st.columns(2)
    x_log = col3.checkbox("X Log")
    y_log = col4.checkbox("Y Log")

# --- 3. ÁREA PRINCIPAL ---
st.title("Digitalizador Web de Curvas")

if uploaded_file is not None:
    # Variables de matemáticas
    calib = build_calib(xmin, xmax, ymin, ymax, x_log, y_log)
    real_pts = []
    coeffs = None
    fit = None
    
    # Calcular puntos reales si hay calibración
    if calib is not None and len(st.session_state.curve_clicks) > 0:
        real_pts = [calib.pixel_to_xy(px, py) for px, py in st.session_state.curve_clicks]
        real_pts.sort(key=lambda t: t[0])
    
    # Calcular Polinomio si hay suficientes puntos
    if len(real_pts) >= 4:
        real_x = np.array([p[0] for p in real_pts], dtype=float)
        real_y = np.array([p[1] for p in real_pts], dtype=float)
        fit = fit_poly_degree3(real_x, real_y)
        coeffs = fit["coeffs_high_to_low"]

    # --- MOSTRAR IMAGEN CON CLICS INTERACTIVOS ---
    img_original = Image.open(uploaded_file)
    img_drawn = draw_markers_on_image(img_original, calib, coeffs)
    
    st.write(f"**Estado:** Calibración {len(st.session_state.calib_clicks)}/4 | Puntos curva: {len(st.session_state.curve_clicks)}")
    
    # Capturar el clic en la imagen
    value = streamlit_image_coordinates(img_drawn, key="clicker")
    
    if value is not None:
        point = (value['x'], value['y'])
        
        # Evitar bucles infinitos de Streamlit guardando el último clic
        if st.session_state.last_click != point:
            st.session_state.last_click = point
            
            if mode == "Calibrar (4 Clics)":
                if len(st.session_state.calib_clicks) < 4:
                    st.session_state.calib_clicks.append(point)
                    st.rerun() # Forzar recarga para pintar el punto
                else:
                    st.warning("Calibración completada. Cambia a 'Capturar Curva'.")
            else:
                if len(st.session_state.calib_clicks) < 4:
                    st.error("Debes hacer los 4 clics de calibración primero.")
                else:
                    st.session_state.curve_clicks.append(point)
                    st.rerun() # Forzar recarga para pintar el punto

    st.divider()

    # --- 4. RESULTADOS, TABLA Y CALCULADORA ---
    colA, colB = st.columns([1, 1])
    
    with colA:
        st.subheader("Puntos Capturados (X, Y)")
        if len(real_pts) > 0:
            df = pd.DataFrame(real_pts, columns=["X Real", "Y Real"])
            st.dataframe(df, use_container_width=True)
        else:
            st.info("Captura puntos en la curva para ver la tabla.")

    with colB:
        st.subheader("Resultados del Ajuste")
        if fit is not None:
            a, b, c, d = coeffs
            st.success(f"**Polinomio:** y = ({a:.8g})·x³ + ({b:.8g})·x² + ({c:.8g})·x + ({d:.8g})")
            st.write(f"**RMSE:** {fit['rmse']:.6g} | **Rango X:** [{fit['x_min']:.6g}, {fit['x_max']:.6g}]")
            
            # --- CALCULADORA ---
            st.divider()
            st.subheader("Calculadora de Curva")
            calc_mode = st.radio("Modo:", ["Introducir X, buscar Y", "Introducir Y, buscar X"], horizontal=True)
            calc_val = st.number_input("Valor:", value=0.0, format="%.4f")
            
            if st.button("Calcular"):
                if calc_mode == "Introducir X, buscar Y":
                    if calc_val < fit['x_min'] or calc_val > fit['x_max']:
                        st.error(f"Fuera de rango. X debe estar entre {fit['x_min']:.4g} y {fit['x_max']:.4g}")
                    else:
                        y_res = (a * calc_val**3) + (b * calc_val**2) + (c * calc_val) + d
                        st.metric(label="Resultado Y", value=f"{y_res:.6g}")
                else:
                    poly = np.poly1d([a, b, c, d - calc_val])
                    roots = poly.roots
                    real_roots = [r.real for r in roots if abs(r.imag) < 1e-8 and fit['x_min'] <= r.real <= fit['x_max']]
                    
                    if not real_roots:
                        st.error("Fuera de rango. No existe un valor X válido para ese Y en esta curva.")
                    else:
                        mid_x = (fit['x_min'] + fit['x_max']) / 2.0
                        best_root = min(real_roots, key=lambda r: abs(r - mid_x))
                        st.metric(label="Resultado X", value=f"{best_root:.6g}")
        else:
            st.info("Necesitas capturar al menos 4 puntos para ver el ajuste y la calculadora.")
else:
    st.info("Sube una imagen usando el panel izquierdo para comenzar.")