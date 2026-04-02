import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates

# Import your math module
from digitizer_math import AxisCalib, GraphCalib, fit_poly_degree3, eval_poly

st.set_page_config(layout="wide", page_title="Web Curve Digitizer")

# --- 1. INITIALIZE SESSION STATE ---
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
    """Draws the markers and the red polynomial curve on the image"""
    img_draw = img_pil.copy()
    draw = ImageDraw.Draw(img_draw)
    r = 4 # Point radius
    
    # 1. Draw calibration clicks
    for i, (x, y) in enumerate(st.session_state.calib_clicks):
        color = "yellow" if i < 2 else "magenta"
        draw.ellipse((x-r, y-r, x+r, y+r), fill=color, outline="black")
        
    # 2. Draw curve clicks
    for (x, y) in st.session_state.curve_clicks:
        draw.ellipse((x-r, y-r, x+r, y+r), fill="cyan", outline="black")
        
    # 3. Draw the red polynomial line
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

# --- 2. SIDEBAR (CONTROLS) ---
with st.sidebar:
    st.title("Controls")
    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg", "bmp"])
    
    st.button("Reset Points", on_click=reset_state)
    mode = st.radio("Click Mode", ["Calibrate (4 Clicks)", "Capture Curve"])
    
    st.divider()
    st.subheader("Real Axis Values")
    col1, col2 = st.columns(2)
    xmin = col1.number_input("X Min", value=0.0)
    xmax = col2.number_input("X Max", value=1.0)
    ymin = col1.number_input("Y Min", value=0.0)
    ymax = col2.number_input("Y Max", value=1.0)
    
    col3, col4 = st.columns(2)
    x_log = col3.checkbox("X Log")
    y_log = col4.checkbox("Y Log")

# --- 3. MAIN AREA ---
st.title("Web Curve Digitizer")

if uploaded_file is not None:
    # Math variables
    calib = build_calib(xmin, xmax, ymin, ymax, x_log, y_log)
    real_pts = []
    coeffs = None
    fit = None
    
    # Calculate real points if calibration is complete
    if calib is not None and len(st.session_state.curve_clicks) > 0:
        real_pts = [calib.pixel_to_xy(px, py) for px, py in st.session_state.curve_clicks]
        real_pts.sort(key=lambda t: t[0])
    
    # Calculate Polynomial if there are enough points
    if len(real_pts) >= 4:
        real_x = np.array([p[0] for p in real_pts], dtype=float)
        real_y = np.array([p[1] for p in real_pts], dtype=float)
        fit = fit_poly_degree3(real_x, real_y)
        coeffs = fit["coeffs_high_to_low"]

    # --- DISPLAY IMAGE WITH INTERACTIVE CLICKS ---
    img_original = Image.open(uploaded_file)
    img_drawn = draw_markers_on_image(img_original, calib, coeffs)
    
    st.write(f"**Status:** Calibration {len(st.session_state.calib_clicks)}/4 | Curve points: {len(st.session_state.curve_clicks)}")
    
    # Capture the click on the image
    value = streamlit_image_coordinates(img_drawn, key="clicker")
    
    if value is not None:
        point = (value['x'], value['y'])
        
        # Prevent Streamlit infinite reload loops by saving the last click
        if st.session_state.last_click != point:
            st.session_state.last_click = point
            
            if mode == "Calibrate (4 Clicks)":
                if len(st.session_state.calib_clicks) < 4:
                    st.session_state.calib_clicks.append(point)
                    st.rerun() # Force reload to draw the point
                else:
                    st.warning("Calibration complete. Switch to 'Capture Curve'.")
            else:
                if len(st.session_state.calib_clicks) < 4:
                    st.error("You must make the 4 calibration clicks first.")
                else:
                    st.session_state.curve_clicks.append(point)
                    st.rerun() # Force reload to draw the point

    st.divider()

    # --- 4. RESULTS, TABLE, AND CALCULATOR ---
    colA, colB = st.columns([1, 1])
    
    with colA:
        st.subheader("Captured Points (X, Y)")
        if len(real_pts) > 0:
            df = pd.DataFrame(real_pts, columns=["Real X", "Real Y"])
            st.dataframe(df, use_container_width=True)
        else:
            st.info("Capture points on the curve to see the table.")

    with colB:
        st.subheader("Fit Results")
        if fit is not None:
            a, b, c, d = coeffs
            st.success(f"**Polynomial:** y = ({a:.8g})·x³ + ({b:.8g})·x² + ({c:.8g})·x + ({d:.8g})")
            st.write(f"**RMSE:** {fit['rmse']:.6g} | **X Range:** [{fit['x_min']:.6g}, {fit['x_max']:.6g}]")
            
            # --- CALCULATOR ---
            st.divider()
            st.subheader("Curve Calculator")
            calc_mode = st.radio("Mode:", ["Input X, find Y", "Input Y, find X"], horizontal=True)
            calc_val = st.number_input("Value:", value=0.0, format="%.4f")
            
            if st.button("Calculate"):
                if calc_mode == "Input X, find Y":
                    if calc_val < fit['x_min'] or calc_val > fit['x_max']:
                        st.error(f"Out of bounds. X must be between {fit['x_min']:.4g} and {fit['x_max']:.4g}")
                    else:
                        y_res = (a * calc_val**3) + (b * calc_val**2) + (c * calc_val) + d
                        st.metric(label="Result Y", value=f"{y_res:.6g}")
                else:
                    poly = np.poly1d([a, b, c, d - calc_val])
                    roots = poly.roots
                    real_roots = [r.real for r in roots if abs(r.imag) < 1e-8 and fit['x_min'] <= r.real <= fit['x_max']]
                    
                    if not real_roots:
                        st.error("Out of bounds. No valid X value exists for this Y on this curve.")
                    else:
                        mid_x = (fit['x_min'] + fit['x_max']) / 2.0
                        best_root = min(real_roots, key=lambda r: abs(r - mid_x))
                        st.metric(label="Result X", value=f"{best_root:.6g}")
        else:
            st.info("You need to capture at least 4 points to see the fit and use the calculator.")
else:
    st.info("Upload an image using the left panel to begin.")
