"""
PCB Component Detection — Streamlit Web Application (v3)
Giao diện sáng · Menu bên trái · Màu riêng cho từng lớp · Biểu tượng sơ đồ SVG
Số liệu đã đối chiếu với eda.ipynb, evaluation_metrics.csv, confusion_matrix.png và các biểu đồ huấn luyện.
Chạy:  streamlit run app/app.py   (đặt .streamlit/config.toml ở thư mục gốc dự án)
"""
from __future__ import annotations

import base64
import io
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps
from ultralytics import YOLO

# ───────────────────────── đường dẫn ─────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "model" / "best.pt"
ASSET_ROOTS = [PROJECT_ROOT / n for n in ("assets", "assest") if (PROJECT_ROOT / n).is_dir()]
ASSET_ROOTS = ASSET_ROOTS or [PROJECT_ROOT / "assets"]
SEARCH_DIRS = [r / s if s else r for r in ASSET_ROOTS for s in ("", "graph", "pred", "results")]
SAMPLE_DIRS = [r / "samples" for r in ASSET_ROOTS]


def find_asset(name: str) -> Path | None:
    for d in SEARCH_DIRS:
        p = d / name
        if p.is_file():
            return p
    return None


def glob_assets(pattern: str) -> list[Path]:
    found: dict[str, Path] = {}
    for d in SEARCH_DIRS:
        for p in sorted(d.glob(pattern)):
            found.setdefault(p.name, p)
    return [found[k] for k in sorted(found)]


def sample_images() -> list[Path]:
    out: list[Path] = []
    for d in SAMPLE_DIRS:
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
            out += sorted(d.glob(ext))
    return out


st.set_page_config(page_title="PCB Component Detector", page_icon="🔬",
                   layout="wide", initial_sidebar_state="expanded")

# ───────────────────────── hằng số dự án ─────────────────────────
CLASS_DESC = {
    "resistor": "Điện trở", "capacitor": "Tụ điện", "connector": "Đầu nối",
    "ic": "Vi mạch IC", "led": "Đèn LED", "diode": "Điốt",
    "transistor": "Transistor", "inductor": "Cuộn cảm", "jumper": "Cầu nối",
    "button": "Nút bấm",
}
CLASS_SUB = {
    "resistor": "Kiểm soát dòng điện", "capacitor": "Lưu trữ năng lượng",
    "connector": "Giao tiếp giữa các khối", "ic": "Xử lý và logic",
    "led": "Tín hiệu quang học", "diode": "Chỉnh lưu dòng điện",
    "transistor": "Khuếch đại / chuyển mạch", "inductor": "Lọc, lưu trữ từ trường",
    "jumper": "Kết nối / ngắt mạch", "button": "Tương tác thủ công",
}
CLASS_COLORS = {
    "resistor": "#e4572e", "capacitor": "#0969da", "connector": "#2da44e",
    "ic": "#8250df", "led": "#d4a72c", "diode": "#bf3989", "transistor": "#0e8a9b",
    "inductor": "#fb8500", "jumper": "#6e7781", "button": "#1f6f43",
}
ACCENT = "#0969da"

# --- Số liệu EDA (eda.ipynb: toàn bộ 460 ảnh, 6.630 đối tượng) ---
DATASET_CLASS_COUNTS = {"capacitor": 2292, "resistor": 1963, "connector": 900, "ic": 636, "led": 453,
                        "jumper": 100, "button": 79, "diode": 76, "inductor": 68, "transistor": 63}
N_IMAGES = 460
N_OBJECTS = sum(DATASET_CLASS_COUNTS.values())              # 6630
AVG_OBJ_PER_IMG = 14.41                                      # EDA: 6630 / 460
MAX_OBJ_PER_IMG = 15                                         # EDA
IMBALANCE = max(DATASET_CLASS_COUNTS.values()) / min(DATASET_CLASS_COUNTS.values())   # 36,38 (capacitor / transistor)
SPLIT = {"Train": 368, "Val": 46, "Test": 46}                # 80 / 10 / 10, seed 42

# --- Kết quả trên tập test (evaluation_metrics.csv: trung bình macro 10 lớp) ---
TEST_OVERALL = {"Precision": 0.236, "Recall": 0.374, "mAP50": 0.281, "mAP50-95": 0.197}
TEST_CLASS_COUNTS = {"capacitor": 202, "resistor": 199, "connector": 101, "ic": 76, "led": 51,
                     "diode": 11, "button": 10, "jumper": 9, "transistor": 7, "inductor": 4}
TEST_OBJECTS = sum(TEST_CLASS_COUNTS.values())               # 670

# --- Confusion matrix (confusion_matrix.png; cột = nhãn thật, hàng = dự đoán) ---
CM_DIAG = {"resistor": 98, "capacitor": 123, "connector": 83, "ic": 47, "led": 24,
           "diode": 1, "transistor": 2, "inductor": 1, "jumper": 3, "button": 6}
CM_FP_BG = {"resistor": 3250, "capacitor": 4254, "connector": 1288, "ic": 884, "led": 1154,
            "diode": 221, "transistor": 98, "inductor": 222, "jumper": 304, "button": 105}   # dự đoán lớp X trên nền
CM_FN_BG = {"resistor": 45, "capacitor": 35, "connector": 10, "ic": 10, "led": 9,
            "diode": 1, "transistor": 3, "inductor": 1, "jumper": 2, "button": 0}            # linh kiện thật bị bỏ sót
CM_TOP_CONFUSIONS = [("resistor", "capacitor", 23), ("capacitor", "resistor", 22), ("capacitor", "led", 16),
                     ("resistor", "ic", 13), ("led", "capacitor", 10), ("resistor", "inductor", 10)]  # (thật → dự đoán, số lượng)


def fmt_int(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def fmt_dec(x: float, d: int = 1) -> str:
    return f"{x:.{d}f}".replace(".", ",")


# ───────────────────────── CSS ─────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, button, input { font-family: 'Inter', 'Segoe UI', Arial, sans-serif; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1240px; }

/* sidebar */
[data-testid="stSidebar"] { border-right: 1px solid #d0d7de; }
[data-testid="stSidebar"] div[data-testid="stRadio"] label {
    background: #fff; border: 1px solid #d0d7de; border-radius: 10px;
    padding: 9px 14px; margin-bottom: 6px; width: 100%; cursor: pointer;
    transition: all .18s ease; font-weight: 600;
}
[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover { border-color: #0969da; background: #f0f7ff; }
[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) {
    background: #0969da; border-color: #0969da; box-shadow: 0 2px 8px rgba(9,105,218,.28); }
[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) * { color: #fff !important; }
[data-testid="stSidebar"] div[data-testid="stRadio"] label > div:first-child { display: none; }

/* hero */
.hero { position: relative; overflow: hidden; border-radius: 20px; padding: 56px 44px; margin-bottom: 26px;
    text-align: center; color: #fff;
    background: radial-gradient(circle at 15% 20%, rgba(255,255,255,.18), transparent 40%),
                linear-gradient(135deg, #0550ae 0%, #0969da 55%, #388bfd 100%);
    box-shadow: 0 10px 34px rgba(9,105,218,.28); }
.hero h1 { font-size: 2.5rem; font-weight: 800; margin: 0 0 12px; color: #fff; }
.hero p { font-size: 1.08rem; line-height: 1.8; margin: 0; color: rgba(255,255,255,.92); }
.hero .tags { margin-top: 20px; }
.hero .tag { display: inline-block; margin: 3px 4px; padding: 5px 14px; border-radius: 99px; font-size: .8rem;
    font-weight: 600; background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.35); }

/* pills, cards */
.stat-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 14px; margin-bottom: 26px; }
.stat-pill { background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 14px; padding: 18px 14px; text-align: center; }
.stat-pill .val { font-size: 1.65rem; font-weight: 800; color: #0969da; line-height: 1.1; }
.stat-pill .lbl { font-size: .78rem; color: #57606a; margin-top: 6px; }
.card { background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 14px; padding: 22px; margin-bottom: 14px;
    height: calc(100% - 14px); transition: border-color .18s, box-shadow .18s, transform .18s; }
.card:hover { border-color: #0969da; box-shadow: 0 6px 18px rgba(9,105,218,.12); transform: translateY(-2px); }
.card h3 { color: #0969da; margin: 0 0 8px; font-size: 1rem; font-weight: 700; }
.card p, .card li { color: #57606a; margin: 0; font-size: .9rem; line-height: 1.6; }
.comp-card { background: #fff; border: 1px solid #d0d7de; border-radius: 14px; padding: 20px 10px 16px;
    margin-bottom: 12px; text-align: center; transition: all .18s; }
.comp-card:hover { border-color: #0969da; box-shadow: 0 6px 18px rgba(9,105,218,.12); transform: translateY(-2px); }
.comp-card .n { font-size: .92rem; font-weight: 700; color: #1f2328; margin: 10px 0 3px; }
.comp-card .d { font-size: .76rem; color: #57606a; }
.comp-card .c { font-size: .72rem; color: #0969da; font-weight: 600; margin-top: 4px; }
.section-title { font-size: 1.3rem; font-weight: 700; color: #1f2328; border-left: 4px solid #0969da;
    padding-left: 14px; margin: 30px 0 16px; }
.note { background: #fff8c5; border: 1px solid #eac54f; border-radius: 12px; padding: 12px 16px; font-size: .86rem; color: #4d2d00; }

/* chips, legend, bars */
.chip { display: inline-flex; align-items: center; gap: 7px; background: #fff; border: 1px solid #d0d7de;
    border-radius: 99px; padding: 5px 13px; font-size: .82rem; font-weight: 600; margin: 3px; color: #1f2328; }
.dot { width: 11px; height: 11px; border-radius: 50%; display: inline-block; }
.bar-row { display: flex; align-items: center; gap: 12px; margin: 9px 0; font-size: .86rem; }
.bar-bg { flex: 1; height: 10px; background: #eaeef2; border-radius: 99px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 99px; }

/* widgets */
div[data-testid="stFileUploader"] section { border: 2px dashed #b6c2cf; border-radius: 14px; background: #f6f8fa; transition: all .2s; }
div[data-testid="stFileUploader"] section:hover { border-color: #0969da; background: #f0f7ff; }
[data-testid="stMetric"] { background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 12px; padding: 14px 16px; }
[data-testid="stMetricValue"] { color: #0969da; font-weight: 800; }
[data-testid="stExpander"] { border: 1px solid #d0d7de; border-radius: 12px; }
button[data-baseweb="tab"] { font-weight: 600; }
[data-testid="stImage"] img { border-radius: 10px; }

@media (max-width: 768px) {
    .hero { padding: 36px 20px; } .hero h1 { font-size: 1.8rem; }
    .block-container { padding-left: 1rem; padding-right: 1rem; }
}
</style>
""",
    unsafe_allow_html=True,
)


# ───────────────────────── biểu tượng SVG ─────────────────────────
def _ln(x1, y1, x2, y2, w=2.5):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{ACCENT}" stroke-width="{w}" stroke-linecap="round"/>'


def _svg(vb, w, h, body):
    return f'<svg viewBox="{vb}" xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">{body}</svg>'


def _stroke(extra=""):
    return f'fill="none" stroke="{ACCENT}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" {extra}'


COMPONENT_SVG = {
    "resistor": _svg("0 0 90 36", 80, 32, _ln(0, 18, 14, 18) + _ln(76, 18, 90, 18)
        + f'<polyline points="14,18 19,6 25,30 31,6 37,30 43,6 49,30 55,6 61,30 67,18 76,18" {_stroke()}/>'),
    "capacitor": _svg("0 0 50 60", 42, 52, _ln(25, 2, 25, 24) + _ln(5, 24, 45, 24, 4) + _ln(5, 36, 45, 36, 4) + _ln(25, 36, 25, 58)),
    "connector": _svg("0 0 60 54", 56, 50, f'<rect x="4" y="6" width="28" height="42" rx="4" {_stroke()}/>'
        + f'<circle cx="18" cy="20" r="5" fill="{ACCENT}"/><circle cx="18" cy="36" r="5" fill="{ACCENT}"/>'
        + _ln(32, 20, 56, 20) + _ln(32, 36, 56, 36)),
    "ic": _svg("0 0 60 56", 56, 52, f'<rect x="14" y="8" width="32" height="40" rx="3" fill="#ddf4ff" stroke="{ACCENT}" stroke-width="2.5"/>'
        + "".join(_ln(14, y, 4, y, 2) + _ln(46, y, 56, y, 2) for y in (18, 28, 38))
        + f'<text x="30" y="32" text-anchor="middle" font-size="9" font-weight="800" font-family="monospace" fill="{ACCENT}">IC</text>'),
    "led": _svg("0 0 72 50", 66, 46, _ln(0, 25, 14, 25) + f'<polygon points="14,10 14,40 40,25" fill="{ACCENT}"/>'
        + _ln(40, 10, 40, 40, 3) + _ln(40, 25, 58, 25) + _ln(46, 8, 54, 2, 2) + _ln(52, 16, 60, 10, 2)
        + f'<polyline points="54,2 54,6 50,6" {_stroke()}/><polyline points="60,10 60,14 56,14" {_stroke()}/>'),
    "diode": _svg("0 0 70 42", 64, 38, _ln(0, 21, 16, 21) + f'<polygon points="16,8 16,34 42,21" fill="{ACCENT}"/>'
        + _ln(42, 8, 42, 34, 3) + _ln(42, 21, 70, 21)),
    "transistor": _svg("0 0 62 62", 58, 58, f'<circle cx="36" cy="31" r="22" {_stroke()}/>' + _ln(2, 31, 24, 31)
        + _ln(24, 16, 24, 46, 3) + _ln(24, 22, 52, 10) + _ln(24, 40, 52, 52) + f'<polygon points="46,9 52,10 48,16" fill="{ACCENT}"/>'),
    "inductor": _svg("0 0 90 36", 84, 32, _ln(0, 18, 10, 18) + _ln(82, 18, 90, 18)
        + f'<path d="M10,18 C14,18 14,6 19,6 C24,6 24,18 28,18 C32,18 32,6 37,6 C42,6 42,18 46,18 C50,18 50,6 55,6 C60,6 60,18 64,18 C68,18 68,6 73,6 C78,6 78,18 82,18" {_stroke()}/>'),
    "jumper": _svg("0 0 70 50", 64, 46, _ln(0, 36, 14, 36) + _ln(22, 36, 48, 36) + _ln(56, 36, 70, 36)
        + f'<circle cx="18" cy="36" r="4" {_stroke()}/><circle cx="52" cy="36" r="4" {_stroke()}/>'
        + f'<path d="M18,32 Q35,10 52,32" {_stroke()}/>'),
    "button": _svg("0 0 70 52", 64, 48, _ln(0, 38, 18, 38) + _ln(18, 38, 18, 24) + _ln(12, 24, 28, 24, 3)
        + _ln(52, 38, 70, 38) + _ln(52, 38, 52, 24) + _ln(42, 24, 58, 24, 3) + _ln(30, 38, 52, 38)
        + _ln(35, 15, 35, 6, 2) + _ln(30, 6, 40, 6, 2)),
}


def svg_img(svg: str, alt: str = "") -> str:
    b64 = base64.b64encode(svg.strip().encode()).decode()
    return f'<img src="data:image/svg+xml;base64,{b64}" alt="{alt}" style="max-height:48px;max-width:84px;object-fit:contain;"/>'


# ───────────────────────── tiện ích ─────────────────────────
def stretch(fn, *args, **kwargs):
    """Tương thích Streamlit cũ (use_container_width) và mới (width='stretch')."""
    try:
        return fn(*args, width="stretch", **kwargs)
    except Exception:
        return fn(*args, use_container_width=True, **kwargs)


def section(title: str) -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def conf_color(c: float) -> str:
    return "#1a7f37" if c >= 0.7 else "#9a6700" if c >= 0.45 else "#cf222e"


def legend(classes) -> str:
    return "".join(
        f'<span class="chip"><span class="dot" style="background:{CLASS_COLORS[c]}"></span>{CLASS_DESC[c]}</span>'
        for c in classes)


@st.cache_resource(show_spinner=False)
def load_model(path: str) -> YOLO:
    return YOLO(path)


@st.cache_data(show_spinner=False, max_entries=16)
def load_image(data: bytes) -> Image.Image:
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    img.thumbnail((3200, 3200))
    return img


@st.cache_data(show_spinner=False, max_entries=32)
def run_inference(data: bytes, conf: float, iou: float, imgsz: int):
    model = load_model(str(MODEL_PATH))
    res = model.predict(load_image(data), conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]
    dets = []
    for b in (res.boxes if res.boxes is not None else []):
        x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
        dets.append({"cls": res.names[int(b.cls.item())], "conf": float(b.conf.item()), "box": (x1, y1, x2, y2)})
    return dets, float(sum(res.speed.values()))


def _font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "arialbd.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def draw_detections(img: Image.Image, dets: list[dict], labels: bool, show_conf: bool) -> Image.Image:
    out = img.copy()
    d = ImageDraw.Draw(out)
    scale = max(out.size) / 1000
    lw, font = max(2, round(2 * scale)), _font(max(12, round(14 * scale)))
    for det in sorted(dets, key=lambda x: x["conf"]):
        x1, y1, x2, y2 = det["box"]
        col = CLASS_COLORS.get(det["cls"], ACCENT)
        d.rectangle([x1, y1, x2, y2], outline=col, width=lw)
        if labels:
            txt = det["cls"] + (f" {det['conf']:.0%}" if show_conf else "")
            l, t, r, b = d.textbbox((0, 0), txt, font=font)
            pad = 3
            ty = y1 - (b - t) - 2 * pad
            ty = y1 if ty < 0 else ty
            d.rectangle([x1, ty, x1 + (r - l) + 2 * pad, ty + (b - t) + 2 * pad], fill=col)
            d.text((x1 + pad - l, ty + pad - t), txt, fill="white", font=font)
    return out


# ───────────────────────── sidebar ─────────────────────────
MENU = {"🏠  Trang chủ": "home", "🔍  Nhận diện": "detect",
        "📊  Hiệu suất mô hình": "perf", "🖼️  Gallery mẫu": "gallery"}
with st.sidebar:
    st.markdown(
        """<div style="text-align:center;padding:10px 0 18px;">
        <div style="font-size:2.6rem;line-height:1;">🔬</div>
        <h2 style="font-size:1.25rem;font-weight:800;color:#0969da;margin:8px 0 2px;">PCB Detector</h2>
        <p style="font-size:.8rem;color:#57606a;margin:0;">YOLO26n Component Analysis</p></div>""",
        unsafe_allow_html=True)
    choice = st.radio("Điều hướng", list(MENU), label_visibility="collapsed")
    st.markdown("---")
    st.markdown(
        f"""<div style="font-size:.78rem;color:#57606a;line-height:1.7;"><b>Thông tin hệ thống</b><br>
        • Mô hình: YOLO26n (transfer từ COCO)<br>• 10 lớp linh kiện<br>
        • {N_IMAGES} ảnh · {fmt_int(N_OBJECTS)} đối tượng<br>
        • Chia dữ liệu: {SPLIT['Train']} / {SPLIT['Val']} / {SPLIT['Test']} ảnh<br>
        • mAP50 (test): {fmt_dec(TEST_OVERALL['mAP50'] * 100)}%</div>""", unsafe_allow_html=True)
page = MENU[choice]


# ───────────────────────── TRANG CHỦ ─────────────────────────
def page_home() -> None:
    st.markdown(
        f"""<div class="hero"><h1>🔬 PCB Component Detector</h1>
        <p>Nhận diện linh kiện điện tử trên bo mạch in (PCB)<br>
        bằng <b>YOLO26n</b>, huấn luyện trên <b>{N_IMAGES} ảnh</b> ({fmt_int(N_OBJECTS)} đối tượng) với <b>10 loại linh kiện</b>.</p>
        <div class="tags"><span class="tag">Phát hiện đối tượng</span><span class="tag">Transfer learning</span>
        <span class="tag">Ứng dụng Streamlit chạy local</span></div></div>
        <div class="stat-row">
        <div class="stat-pill"><div class="val">{N_IMAGES}</div><div class="lbl">Ảnh dữ liệu</div></div>
        <div class="stat-pill"><div class="val">{fmt_int(N_OBJECTS)}</div><div class="lbl">Đối tượng được gán nhãn</div></div>
        <div class="stat-pill"><div class="val">10</div><div class="lbl">Loại linh kiện</div></div>
        <div class="stat-pill"><div class="val">{fmt_dec(TEST_OVERALL['mAP50'] * 100)}%</div><div class="lbl">mAP50 (tập test)</div></div>
        <div class="stat-pill"><div class="val">640 px</div><div class="lbl">Kích thước huấn luyện</div></div></div>""",
        unsafe_allow_html=True)

    section("Cách sử dụng")
    steps = [("① Mở mục Nhận diện", "Chọn <b>Nhận diện</b> trong menu bên trái."),
             ("② Tải ảnh PCB lên", "Kéo thả hoặc chọn ảnh <b>JPG / PNG / BMP / WEBP</b>."),
             ("③ Xem kết quả", "Mỗi linh kiện có <b>khung màu theo lớp, tên và độ tin cậy</b>.")]
    for col, (t, b) in zip(st.columns(3), steps):
        col.markdown(f'<div class="card"><h3>{t}</h3><p>{b}</p></div>', unsafe_allow_html=True)

    section("10 loại linh kiện được nhận diện")
    cols = st.columns(5)
    for i, c in enumerate(CLASS_DESC):
        cols[i % 5].markdown(
            f"""<div class="comp-card" style="border-top:3px solid {CLASS_COLORS[c]};">
            <div style="display:flex;justify-content:center;align-items:center;height:52px;">{svg_img(COMPONENT_SVG[c], c)}</div>
            <div class="n">{CLASS_DESC[c]}</div><div class="d">{CLASS_SUB[c]}</div>
            <div class="c">{fmt_int(DATASET_CLASS_COUNTS[c])} đối tượng</div></div>""",
            unsafe_allow_html=True)

    section("Pipeline xử lý")
    pipe = [("📥", "Thu thập dữ liệu", f"{N_IMAGES} ảnh PCB, {fmt_int(N_OBJECTS)} box, nhãn định dạng YOLO"),
            ("🧹", "Phân tích & chuẩn hoá", f"EDA, kiểm tra nhãn, chia 80/10/10 ({SPLIT['Train']}/{SPLIT['Val']}/{SPLIT['Test']} ảnh)"),
            ("🧠", "Huấn luyện", "YOLO26n, transfer từ COCO, 94 epoch (tốt nhất epoch 74)"),
            ("🚀", "Triển khai", "Ứng dụng Streamlit chạy local")]
    for col, (ic, t, d) in zip(st.columns(4), pipe):
        col.markdown(f'<div class="card" style="text-align:center;"><div style="font-size:2rem;">{ic}</div>'
                     f'<h3 style="margin-top:8px;">{t}</h3><p>{d}</p></div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="note"><b>Lưu ý:</b> đây là bản mẫu. Mô hình khá tốt ở đầu nối (mAP50 57,2%), nút bấm (52,5%), IC (51,9%) '
        f'và LED (44,8%); còn yếu ở điện trở (30,9%) và tụ điện (16,7%) vì hai lớp này hay bị nhầm với nhau, '
        f'và rất yếu ở các lớp hiếm (diode, transistor, inductor, jumper: mAP50 dưới 10%, chỉ có 63–100 mẫu mỗi lớp trong '
        f'{fmt_int(N_OBJECTS)} đối tượng). Dữ liệu mất cân bằng {fmt_dec(IMBALANCE, 1)} lần giữa lớp nhiều nhất (tụ điện) và ít nhất (transistor). '
        f'Hãy kiểm tra lại kết quả trước khi dùng thực tế.</div>', unsafe_allow_html=True)


# ───────────────────────── NHẬN DIỆN ─────────────────────────
def page_detect() -> None:
    section("Nhận diện linh kiện PCB")
    if not MODEL_PATH.is_file():
        st.error(f"Không tìm thấy model tại `{MODEL_PATH}`.\n\nĐặt file `best.pt` vào thư mục `model/`.")
        st.stop()

    with st.expander("⚙️  Cài đặt nhận diện", expanded=True):
        c1, c2, c3 = st.columns(3)
        conf = c1.slider("Ngưỡng tin cậy (Conf)", 0.05, 0.95, 0.20, 0.05,
                         help="Chỉ giữ box có độ tin cậy ≥ ngưỡng. Mô hình còn yếu (mAP50 ≈ 28%) nên thường cho điểm thấp, đừng đặt quá cao.")
        iou = c2.slider("Ngưỡng IoU (NMS)", 0.10, 0.90, 0.45, 0.05, help="Giảm box trùng lặp.")
        imgsz = c3.select_slider("Kích thước suy luận", options=[640, 960, 1280], value=640,
                                 help="Mô hình được huấn luyện ở 640 px. Ảnh gốc trong dữ liệu có cạnh từ ≈300 đến hơn 5000 px; "
                                      "kích thước lớn hơn có thể giúp bắt linh kiện nhỏ nhưng chậm hơn.")
        classes = st.multiselect("Lớp hiển thị", list(CLASS_DESC), default=list(CLASS_DESC),
                                 format_func=lambda c: CLASS_DESC[c])
        t1, t2, t3 = st.columns(3)
        show_original = t1.toggle("Hiển thị ảnh gốc", True)
        show_labels = t2.toggle("Hiện nhãn trên ảnh", True)
        show_conf = t3.toggle("Hiện % tin cậy", True)

    up = st.file_uploader("Kéo thả hoặc chọn ảnh PCB (JPG · PNG · BMP · WEBP)",
                          type=["jpg", "jpeg", "png", "bmp", "webp"])
    data: bytes | None = up.getvalue() if up else None
    samples = sample_images()
    if data is None and samples:
        pick = st.selectbox("Hoặc thử một ảnh mẫu", ["— Không dùng —"] + [p.name for p in samples])
        if pick != "— Không dùng —":
            data = next(p for p in samples if p.name == pick).read_bytes()

    if data is None:
        st.markdown("""<div style="text-align:center;padding:56px 0;color:#57606a;"><div style="font-size:3rem;">📤</div>
            <div style="font-size:1.1rem;margin-top:12px;font-weight:600;">Tải ảnh PCB lên để bắt đầu nhận diện</div>
            <div style="font-size:.85rem;margin-top:6px;">Ảnh chụp thẳng, đủ sáng và rõ nét cho kết quả tốt hơn</div></div>""",
                    unsafe_allow_html=True)
        return

    image = load_image(data)
    with st.spinner("Đang phân tích ảnh với YOLO26n…"):
        all_dets, ms = run_inference(data, conf, iou, imgsz)
    dets = [d for d in all_dets if d["cls"] in classes]
    counter = Counter(d["cls"] for d in dets)
    annotated = draw_detections(image, dets, show_labels, show_conf)

    m = st.columns(4)
    m[0].metric("Linh kiện phát hiện", len(dets))
    m[1].metric("Số loại", len(counter))
    m[2].metric("Thời gian suy luận", f"{ms:.0f} ms")
    m[3].metric("Độ tin cậy trung bình", f"{(sum(d['conf'] for d in dets) / len(dets)):.0%}" if dets else "—")

    if not dets:
        st.warning("Không có linh kiện nào với thiết lập hiện tại. Thử **giảm ngưỡng tin cậy**, tăng kích thước suy luận hoặc chọn thêm lớp.")
        return

    tab_img, tab_list, tab_stat = st.tabs(["🖼️  Hình ảnh", "📋  Danh sách", "📈  Thống kê"])

    with tab_img:
        st.markdown(legend(c for c in CLASS_DESC if c in counter), unsafe_allow_html=True)
        if show_original:
            a, b = st.columns(2)
            a.markdown("**Ảnh gốc**")
            stretch(a.image, image)
            b.markdown("**Kết quả nhận diện**")
            stretch(b.image, annotated)
        else:
            stretch(st.image, annotated)

    rows = [{"Linh kiện": CLASS_DESC[d["cls"]], "Lớp": d["cls"], "Tin cậy (%)": round(d["conf"] * 100, 1),
             "Rộng (px)": round(d["box"][2] - d["box"][0]), "Cao (px)": round(d["box"][3] - d["box"][1]),
             "x1": round(d["box"][0]), "y1": round(d["box"][1]), "x2": round(d["box"][2]), "y2": round(d["box"][3])}
            for d in sorted(dets, key=lambda x: -x["conf"])]
    df = pd.DataFrame(rows)

    with tab_list:
        cfg = {"Tin cậy (%)": st.column_config.ProgressColumn("Tin cậy", min_value=0, max_value=100, format="%.0f%%")}
        stretch(st.dataframe, df, hide_index=True, column_config=cfg)

    with tab_stat:
        left, right = st.columns(2)
        with left:
            st.markdown("**Số lượng theo lớp**")
            top = max(counter.values())
            html = "".join(
                f'<div class="bar-row"><div style="width:110px;">{CLASS_DESC[c]}</div>'
                f'<div class="bar-bg"><div class="bar-fill" style="width:{n / top * 100:.0f}%;background:{CLASS_COLORS[c]};"></div></div>'
                f'<div style="width:28px;text-align:right;font-weight:700;">{n}</div></div>'
                for c, n in counter.most_common())
            st.markdown(html, unsafe_allow_html=True)
        with right:
            st.markdown("**Độ tin cậy trung bình theo lớp**")
            html = ""
            for c, _ in counter.most_common():
                avg = sum(d["conf"] for d in dets if d["cls"] == c) / counter[c]
                html += (f'<div class="bar-row"><div style="width:110px;">{CLASS_DESC[c]}</div>'
                         f'<div class="bar-bg"><div class="bar-fill" style="width:{avg * 100:.0f}%;background:{conf_color(avg)};"></div></div>'
                         f'<div style="width:42px;text-align:right;font-weight:700;color:{conf_color(avg)};">{avg:.0%}</div></div>')
            st.markdown(html, unsafe_allow_html=True)

    buf = io.BytesIO()
    annotated.save(buf, format="PNG")
    d1, d2, d3 = st.columns([1, 1, 3])
    d1.download_button("⬇️  Tải ảnh kết quả", buf.getvalue(), "pcb_detection_result.png", "image/png")
    d2.download_button("⬇️  Tải CSV", df.to_csv(index=False).encode("utf-8-sig"), "pcb_detections.csv", "text/csv")
    st.caption(f"Mô hình còn hạn chế (mAP50 ≈ {TEST_OVERALL['mAP50'] * 100:.0f}% trên tập test): hãy kiểm tra lại các box có độ tin cậy thấp.")


# ───────────────────────── HIỆU SUẤT ─────────────────────────
def page_perf() -> None:
    section("Hiệu suất và đánh giá mô hình")
    t_over, t_cls, t_curve, t_cm = st.tabs(["Tổng quan", "Theo lớp", "Đường cong huấn luyện", "Ma trận nhầm lẫn"])

    with t_over:
        st.markdown(f"**Kết quả trên tập test** ({SPLIT['Test']} ảnh, {TEST_OBJECTS} đối tượng; trung bình macro 10 lớp)")
        cols = st.columns(4)
        for col, (k, v) in zip(cols, TEST_OVERALL.items()):
            col.metric(k, f"{v:.1%}")
        st.markdown("")
        p1, p2, p3 = st.columns(3)
        p1.markdown('<div class="card"><h3>Kiến trúc</h3><p><b>YOLO26n</b>, 2,39 triệu tham số<br>Pretrained COCO<br>Input 640×640</p></div>', unsafe_allow_html=True)
        p2.markdown('<div class="card"><h3>Huấn luyện</h3><p>Tối đa 100 epoch (patience 20)<br>Dừng sớm tại epoch 94, tốt nhất epoch 74<br>Batch 16 · AdamW · LR 0,001</p></div>', unsafe_allow_html=True)
        p3.markdown('<div class="card"><h3>Augmentation</h3><p>Mosaic 1,0 · Mixup 0,15<br>CopyPaste 0,3 · Rotate ±90°<br>FlipUD 0,5 · FlipLR 0,5</p></div>', unsafe_allow_html=True)

        st.markdown("**Dữ liệu (kết quả EDA)**")
        d = st.columns(4)
        d[0].metric("Tổng ảnh hợp lệ", f"{N_IMAGES}")
        d[1].metric(f"Train (80%)", f"{SPLIT['Train']} ảnh")
        d[2].metric(f"Val (10%)", f"{SPLIT['Val']} ảnh")
        d[3].metric(f"Test (10%)", f"{SPLIT['Test']} ảnh")
        e = st.columns(4)
        e[0].metric("Tổng số đối tượng", fmt_int(N_OBJECTS))
        e[1].metric("Đối tượng / ảnh (TB)", fmt_dec(AVG_OBJ_PER_IMG, 2).replace(",", ","))
        e[2].metric("Nhiều nhất / ảnh", f"{MAX_OBJ_PER_IMG}")
        e[3].metric("Mất cân bằng lớp", f"{fmt_dec(IMBALANCE, 2)}×", help="Tụ điện (2.292) so với transistor (63)")
        st.caption("Gần như toàn bộ ảnh (≈420/460) có đúng 15 đối tượng; kích thước ảnh gốc từ ≈300 đến hơn 5000 px, "
                   "tỉ lệ khung hình tập trung quanh 1,0–2,0.")

        g1, g2 = st.columns(2)
        with g1:
            st.markdown(f"**Phân bố đối tượng theo lớp (toàn bộ dữ liệu, {fmt_int(N_OBJECTS)})**")
            st.bar_chart(pd.Series(DATASET_CLASS_COUNTS, name="Số đối tượng"))
        with g2:
            st.markdown(f"**Phân bố đối tượng theo lớp (tập test, {TEST_OBJECTS})**")
            st.bar_chart(pd.Series(TEST_CLASS_COUNTS, name="Số đối tượng"))

    with t_cls:
        csv_path = next((r / "evaluation_metrics.csv" for r in ASSET_ROOTS if (r / "evaluation_metrics.csv").is_file()), None)
        if csv_path is None:
            st.warning("Không tìm thấy file `evaluation_metrics.csv` trong thư mục assets.")
        else:
            dm = pd.read_csv(csv_path)
            metrics = ["Precision", "Recall", "mAP50", "mAP50-95"]
            view = dm.copy()
            view.insert(1, "Số mẫu (toàn bộ)", view["Class"].map(DATASET_CLASS_COUNTS))
            view.insert(2, "Số mẫu (test)", view["Class"].map(TEST_CLASS_COUNTS))
            view["Class"] = view["Class"].map(lambda c: f"{CLASS_DESC.get(c, c)} ({c})")
            view[metrics] = view[metrics] * 100
            cfg = {m: st.column_config.ProgressColumn(m, min_value=0, max_value=100, format="%.1f%%") for m in metrics}
            stretch(st.dataframe, view, hide_index=True, column_config=cfg)
            st.markdown("**mAP50 theo lớp**")
            st.bar_chart(dm.set_index("Class")["mAP50"])
            weak = dm[dm["mAP50"] < 0.10]["Class"].tolist()
            if weak:
                st.warning("Các lớp mAP50 dưới 10% (đều là lớp hiếm, cần thêm dữ liệu): "
                           + ", ".join(f"{CLASS_DESC.get(c, c)} ({DATASET_CLASS_COUNTS.get(c, '?')} mẫu)" for c in weak))
            avg = dm[metrics].mean()
            st.caption("Trung bình các lớp (macro): " + " · ".join(f"{m} {avg[m]:.1%}" for m in metrics))

    with t_curve:
        p = find_asset("comprehensive_training_metrics.png")
        if p:
            stretch(st.image, str(p), caption="Loss, Precision, Recall, mAP50 và mAP50-95 theo epoch")
        else:
            st.info("Chưa có ảnh `comprehensive_training_metrics.png`.")
        q = find_asset("training_curve.png")
        if q:
            with st.expander("Biểu đồ gốc của Ultralytics"):
                stretch(st.image, str(q))
        st.info("**Nhận xét:** huấn luyện dừng ở epoch 94 (patience 20), tốt nhất ở epoch 74. Box loss giảm từ ≈2,4 (train) xuống ≈1,6, "
                "val box loss ≈1,74; cls loss train và val chập nhau ở ≈2,0, nên chưa có dấu hiệu overfit mà có xu hướng underfit. "
                "mAP50 đạt đỉnh ≈0,27 quanh epoch 67–74, mAP50-95 đạt đỉnh ≈0,175 quanh epoch 74. "
                "Precision đạt đỉnh ≈0,68 rất sớm (epoch 5) rồi giảm về ≈0,24, trong khi recall tăng dần lên ≈0,3–0,45.")

    with t_cm:
        p = find_asset("confusion_matrix.png")
        if p:
            stretch(st.image, str(p), caption="Confusion matrix trên tập test (số lượng, chưa chuẩn hoá)")
            st.info("**Cách đọc:** cột = nhãn thật, hàng = dự đoán. Ô chéo = dự đoán đúng. "
                    "Ô 'background' ở **cột cuối** là dự đoán sai trên nền (false positive); "
                    "ô 'background' ở **hàng cuối** là linh kiện bị bỏ sót (false negative).")
        else:
            st.info("Chưa có ảnh `confusion_matrix.png`.")

        correct = sum(CM_DIAG.values())
        c1, c2, c3 = st.columns(3)
        c1.metric("Dự đoán đúng lớp", f"{correct}/{TEST_OBJECTS}", f"{correct / TEST_OBJECTS:.1%}", delta_color="off")
        c2.metric("Bỏ sót (dự đoán là nền)", fmt_int(sum(CM_FN_BG.values())))
        c3.metric("Dự đoán thừa trên nền", fmt_int(sum(CM_FP_BG.values())))
        st.caption("Số dự đoán thừa trên nền rất lớn so với số đối tượng thật vì ma trận được tính ở ngưỡng tin cậy thấp; "
                   "vì vậy các con số ở đây không khớp tuyệt đối với Precision/Recall trong bảng theo lớp.")

        cm_rows = [{"Lớp": f"{CLASS_DESC[c]} ({c})", "Số thật (test)": TEST_CLASS_COUNTS[c], "Đúng": CM_DIAG[c],
                    "Tỉ lệ đúng (%)": round(CM_DIAG[c] / TEST_CLASS_COUNTS[c] * 100, 1),
                    "Bỏ sót": CM_FN_BG[c], "Dự đoán thừa trên nền": CM_FP_BG[c]} for c in CLASS_DESC]
        cfg = {"Tỉ lệ đúng (%)": st.column_config.ProgressColumn("Tỉ lệ đúng", min_value=0, max_value=100, format="%.1f%%")}
        stretch(st.dataframe, pd.DataFrame(cm_rows), hide_index=True, column_config=cfg)

        st.markdown("**Các nhầm lẫn giữa các lớp nhiều nhất** (thật → dự đoán)")
        conf_rows = [{"Nhãn thật": CLASS_DESC[a], "Bị dự đoán thành": CLASS_DESC[b], "Số lượng": n}
                     for a, b, n in CM_TOP_CONFUSIONS]
        stretch(st.dataframe, pd.DataFrame(conf_rows), hide_index=True)


# ───────────────────────── GALLERY ─────────────────────────
def page_gallery() -> None:
    section("Gallery kết quả nhận diện mẫu")
    st.markdown("<p style='color:#57606a;'>Kết quả trên tập test, kèm khung, tên linh kiện và độ tin cậy.</p>", unsafe_allow_html=True)
    imgs = glob_assets("prediction_*.jpg")
    if not imgs:
        st.warning("Chưa có ảnh mẫu `prediction_*.jpg` trong thư mục assets.")
    else:
        n = st.select_slider("Số cột", options=[1, 2, 3, 4], value=3)
        for start in range(0, len(imgs), n):
            cols = st.columns(n)
            for col, p in zip(cols, imgs[start:start + n]):
                with col:
                    stretch(st.image, str(p), caption=p.stem.replace("prediction_", "Mẫu "))
    sample = find_asset("test_predictions_sample.png")
    if sample:
        section("Tổng hợp ảnh mẫu test")
        stretch(st.image, str(sample))


{"home": page_home, "detect": page_detect, "perf": page_perf, "gallery": page_gallery}[page]()