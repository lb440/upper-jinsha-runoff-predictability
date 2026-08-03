from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "新的图"

MAP_PATH = FIG_DIR / "金沙江地形图 （初始版）.png"
HYDRO_PATH = FIG_DIR / "预见期1天折线图.png"
SHAP_PATH = FIG_DIR / "SHAP_feature_lag_mean_abs_heatmap_H1d.png"

OUT_PNG = FIG_DIR / "图形摘要_修改版_600dpi.png"
OUT_PREVIEW = FIG_DIR / "图形摘要_修改版_preview.png"

SCALE = 3
BASE_W, BASE_H = 1672, 941
W, H = BASE_W * SCALE, BASE_H * SCALE

BLUE = (0, 40, 165)
DARK_BLUE = (0, 61, 132)
MID_BLUE = (0, 86, 185)
LIGHT_LINE = (92, 133, 190)
GREEN = (45, 145, 62)
WATER = (20, 100, 190)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

FONT_DIR = Path(r"C:\Windows\Fonts")


def sc(value: float) -> int:
    return int(round(value * SCALE))


def font(filename: str, size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        FONT_DIR / filename,
        FONT_DIR / filename.lower(),
        FONT_DIR / filename.upper(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), sc(size))
    return ImageFont.load_default()


F_TITLE = font("timesbd.ttf", 44)
F_SUBTITLE = font("timesi.ttf", 28)
F_HEAD = font("timesbd.ttf", 30)
F_SECTION = font("times.ttf", 24)
F_BOX_TITLE = font("timesbd.ttf", 28)
F_BOX_TITLE_SMALL = font("timesbd.ttf", 25)
F_TEXT = font("times.ttf", 23)
F_TEXT_SMALL = font("times.ttf", 20)
F_TEXT_ITALIC = font("timesi.ttf", 20)
F_FORMULA = font("timesi.ttf", 31)
F_FORMULA_SMALL = font("timesi.ttf", 19)
F_FORMULA_SUP = font("times.ttf", 18)
F_FORMULA_SYMBOL = font("cambria.ttc", 27)
F_VAR = font("timesi.ttf", 28)
F_VAR_SUB = font("timesi.ttf", 18)
F_BAR = font("times.ttf", 26)
F_BAR_ITALIC = font("timesi.ttf", 27)
F_BAR_SUB = font("timesi.ttf", 17)


def bxy(x1: float, y1: float, x2: float, y2: float) -> tuple[int, int, int, int]:
    return sc(x1), sc(y1), sc(x2), sc(y2)


def text_bbox(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont) -> tuple[int, int, int, int]:
    return draw.textbbox((0, 0), text, font=fnt)


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont) -> tuple[int, int]:
    box = text_bbox(draw, text, fnt)
    return box[2] - box[0], box[3] - box[1]


def center_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    fnt: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int] = BLACK,
) -> None:
    x, y = sc(xy[0]), sc(xy[1])
    tw, th = text_size(draw, text, fnt)
    draw.text((x - tw // 2, y - th // 2), text, font=fnt, fill=fill)


def rounded(
    draw: ImageDraw.ImageDraw,
    box: tuple[float, float, float, float],
    outline: tuple[int, int, int] = BLUE,
    fill: tuple[int, int, int] | None = None,
    width: int = 2,
    radius: int = 12,
) -> None:
    draw.rounded_rectangle(bxy(*box), radius=sc(radius), fill=fill, outline=outline, width=sc(width))


def draw_down_arrow(
    draw: ImageDraw.ImageDraw,
    x: float,
    y1: float,
    y2: float,
    color: tuple[int, int, int] = DARK_BLUE,
    width: int = 5,
) -> None:
    x0, yy1, yy2 = sc(x), sc(y1), sc(y2)
    draw.line((x0, yy1, x0, yy2 - sc(12)), fill=color, width=sc(width))
    draw.polygon(
        [
            (x0 - sc(10), yy2 - sc(12)),
            (x0 + sc(10), yy2 - sc(12)),
            (x0, yy2 + sc(9)),
        ],
        fill=color,
    )


def crop_near_white(image: Image.Image, threshold: int = 248, margin: int = 12) -> Image.Image:
    img = image.convert("RGB")
    pix = img.load()
    w, h = img.size
    xs: list[int] = []
    ys: list[int] = []
    step = max(1, min(w, h) // 900)
    for y in range(0, h, step):
        for x in range(0, w, step):
            r, g, b = pix[x, y]
            if min(r, g, b) < threshold:
                xs.append(x)
                ys.append(y)
    if not xs:
        return img
    left = max(0, min(xs) - margin)
    right = min(w, max(xs) + margin)
    top = max(0, min(ys) - margin)
    bottom = min(h, max(ys) + margin)
    return img.crop((left, top, right, bottom))


def fit_image(
    canvas: Image.Image,
    image: Image.Image,
    box: tuple[float, float, float, float],
    bg: tuple[int, int, int] = WHITE,
) -> None:
    x1, y1, x2, y2 = bxy(*box)
    bw, bh = x2 - x1, y2 - y1
    img = image.convert("RGB")
    iw, ih = img.size
    ratio = min(bw / iw, bh / ih)
    nw, nh = max(1, int(iw * ratio)), max(1, int(ih * ratio))
    resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
    holder = Image.new("RGB", (bw, bh), bg)
    holder.paste(resized, ((bw - nw) // 2, (bh - nh) // 2))
    canvas.paste(holder, (x1, y1))


def add_padding(
    image: Image.Image,
    left: int = 0,
    top: int = 0,
    right: int = 0,
    bottom: int = 0,
    color: tuple[int, int, int] = WHITE,
) -> Image.Image:
    img = image.convert("RGB")
    out = Image.new("RGB", (img.width + left + right, img.height + top + bottom), color)
    out.paste(img, (left, top))
    return out


def draw_var(draw: ImageDraw.ImageDraw, x: int, y: int, main: str, sub: str, color=BLACK) -> int:
    draw.text((x, y), main, font=F_VAR, fill=color)
    mw, _ = text_size(draw, main, F_VAR)
    draw.text((x + mw - sc(2), y + sc(15)), sub, font=F_VAR_SUB, fill=color)
    sw, _ = text_size(draw, sub, F_VAR_SUB)
    return mw + sw + sc(2)


def draw_var_list(draw: ImageDraw.ImageDraw, x: int, y: int, vars_: list[tuple[str, str]], color=BLACK) -> int:
    cx = x
    for i, (main, sub) in enumerate(vars_):
        cx += draw_var(draw, cx, y, main, sub, color)
        if i < len(vars_) - 1:
            draw.text((cx + sc(2), y + sc(3)), ",", font=F_VAR, fill=color)
            comma_w, _ = text_size(draw, ",", F_VAR)
            cx += comma_w + sc(15)
    return cx - x


def measure_var_list_width(draw: ImageDraw.ImageDraw, vars_: list[tuple[str, str]]) -> int:
    total = 0
    for i, (main, sub) in enumerate(vars_):
        mw, _ = text_size(draw, main, F_VAR)
        sw, _ = text_size(draw, sub, F_VAR_SUB)
        total += mw + sw + sc(2)
        if i < len(vars_) - 1:
            comma_w, _ = text_size(draw, ",", F_VAR)
            total += comma_w + sc(15)
    return total


def formula_segments(draw: ImageDraw.ImageDraw) -> list[tuple[str, ImageFont.FreeTypeFont, int, int]]:
    return [
        ("X", F_FORMULA, 0, 0),
        ("t", F_FORMULA_SMALL, -2, 15),
        (" ∈ ", F_FORMULA_SYMBOL, 0, -1),
        ("R", F_FORMULA, 0, 0),
        ("90×9", F_FORMULA_SUP, -1, -15),
        (" → Q", F_FORMULA, 0, 0),
        ("Shigu", F_FORMULA_SMALL, -3, 16),
        ("(t+h)", F_FORMULA, 0, 0),
    ]


def draw_formula_centered(draw: ImageDraw.ImageDraw, center_x: float, y: float) -> None:
    segments = formula_segments(draw)
    widths = []
    for text, fnt, _, _ in segments:
        w, _ = text_size(draw, text, fnt)
        widths.append(w)
    total = sum(widths)
    x = sc(center_x) - total // 2
    base_y = sc(y)
    for (text, fnt, dx, dy), width in zip(segments, widths):
        draw.text((x + sc(dx), base_y + sc(dy)), text, font=fnt, fill=BLACK)
        x += width


def draw_bottom_phrase_left(draw: ImageDraw.ImageDraw, x: float, y: float) -> None:
    cx, cy = sc(x), sc(y)
    draw.text((cx, cy), "1–3 d: ", font=F_BAR, fill=WHITE)
    start_w, _ = text_size(draw, "1–3 d: ", F_BAR)
    cx += start_w
    for i, var in enumerate([("Q", "z"), ("P", "z"), ("P", "i")]):
        cx += draw_var(draw, cx, cy - sc(2), var[0], var[1], WHITE)
        if i == 0:
            sep = ", "
        elif i == 1:
            sep = ", and "
        else:
            sep = " dominate short-lead predictability"
        draw.text((cx + sc(2), cy), sep, font=F_BAR, fill=WHITE)
        sep_w, _ = text_size(draw, sep, F_BAR)
        cx += sep_w + sc(3)


def main() -> None:
    canvas = Image.new("RGB", (W, H), WHITE)
    draw = ImageDraw.Draw(canvas)

    # Title block
    center_text(draw, (BASE_W / 2, 36), "Lead-time-dependent shifts in dominant predictors and lag effects", F_TITLE, BLACK)
    center_text(draw, (BASE_W / 2, 89), "Daily runoff forecasting in the Upper Jinsha River cold-region basin", F_SUBTITLE, BLACK)

    left = (10, 115, 605, 823)
    mid = (625, 115, 1075, 823)
    right = (1095, 115, 1662, 823)
    for box in (left, mid, right):
        rounded(draw, box, outline=BLUE, fill=None, width=2, radius=14)

    center_text(draw, ((left[0] + left[2]) / 2, 145), "1. Basin setting & predictors", F_HEAD, BLUE)
    center_text(draw, ((mid[0] + mid[2]) / 2, 145), "2. Multi-lead runoff forecasting", F_HEAD, BLUE)
    center_text(draw, ((right[0] + right[2]) / 2, 145), "3. Hydrological interpretation", F_HEAD, BLUE)

    # Left panel: map and predictors
    map_img = crop_near_white(Image.open(MAP_PATH), threshold=252, margin=25)
    fit_image(canvas, map_img, (43, 165, 565, 596))

    core_box = (42, 614, 570, 706)
    storage_box = (42, 718, 570, 812)
    rounded(draw, core_box, outline=GREEN, fill=None, width=2, radius=9)
    rounded(draw, storage_box, outline=WATER, fill=None, width=2, radius=9)
    center_text(draw, (306, 642), "Core 3-day predictor group:", F_BOX_TITLE_SMALL, BLACK)
    core_width = measure_var_list_width(draw, [("Q", "z"), ("P", "z"), ("P", "i")])
    draw_var_list(draw, sc(306) - core_width // 2, sc(666), [("Q", "z"), ("P", "z"), ("P", "i")], BLACK)

    center_text(draw, (306, 744), "Storage-related longer-lag signals:", F_BOX_TITLE_SMALL, BLACK)
    storage_width = measure_var_list_width(draw, [("S", "z"), ("S", "i"), ("E", "z"), ("E", "i")])
    draw_var_list(draw, sc(306) - storage_width // 2, sc(768), [("S", "z"), ("S", "i"), ("E", "z"), ("E", "i")], BLACK)

    # Middle panel: forecasting workflow
    win = (656, 208, 1038, 345)
    lead = (656, 405, 1038, 523)
    models = (656, 585, 1038, 724)
    protocol = (700, 752, 1000, 795)
    for box in (win, lead, models, protocol):
        rounded(draw, box, outline=LIGHT_LINE, fill=WHITE, width=1.2, radius=10)

    center_text(draw, (847, 247), "90-day input window", F_BOX_TITLE, BLACK)
    draw_formula_centered(draw, 847, 284)
    draw_down_arrow(draw, 847, 355, 385)

    center_text(draw, (847, 446), "Lead times", F_BOX_TITLE, BLACK)
    center_text(draw, (847, 488), "1 d | 3 d | 7 d | 15 d", F_TEXT, BLACK)
    draw_down_arrow(draw, 847, 533, 565)

    center_text(draw, (847, 621), "Seven forecasting models", F_BOX_TITLE_SMALL, BLACK)
    center_text(draw, (847, 659), "Persistence | XGBoost | MLP | GRU", F_TEXT_SMALL, BLACK)
    center_text(draw, (847, 694), "LSTM | Transformer | LSTM–Transformer", F_TEXT_SMALL, BLACK)
    center_text(draw, (850, 775), "Same split and evaluation protocol", F_TEXT_ITALIC, BLACK)

    # Right panel: hydrograph and SHAP
    draw.text((sc(1130), sc(176)), "A. Test-period hydrograph", font=F_SECTION, fill=BLACK)
    hydro_img = crop_near_white(Image.open(HYDRO_PATH), threshold=252, margin=8)
    fit_image(canvas, hydro_img, (1138, 206, 1612, 486))

    draw.text((sc(1130), sc(522)), "B. Lag-specific SHAP attribution", font=F_SECTION, fill=BLACK)
    shap_raw = Image.open(SHAP_PATH).convert("RGB")
    shap_crop = shap_raw.crop((0, int(shap_raw.height * 0.13), shap_raw.width, shap_raw.height))
    shap_crop = crop_near_white(shap_crop, threshold=253, margin=8)
    shap_crop = add_padding(
        shap_crop,
        left=sc(60),
        top=sc(6),
        right=sc(28),
        bottom=sc(18),
    )
    fit_image(canvas, shap_crop, (1114, 562, 1610, 818))

    # Bottom synthesis strip
    bar = bxy(10, 838, 1662, 924)
    draw.rounded_rectangle(bar, radius=sc(10), fill=DARK_BLUE)
    draw.line((sc(847), sc(858), sc(847), sc(904)), fill=(210, 225, 245), width=sc(1))
    draw_bottom_phrase_left(draw, 130, 865)
    center_text(draw, (1238, 866), "7–15 d: snow, evaporation, and water-balance signals", F_BAR, WHITE)
    center_text(draw, (1238, 896), "become more important", F_BAR, WHITE)

    canvas.save(OUT_PNG, dpi=(600, 600))
    canvas.resize((BASE_W, BASE_H), Image.Resampling.LANCZOS).save(OUT_PREVIEW, dpi=(200, 200))
    print(f"Saved: {OUT_PNG}")
    print(f"Saved: {OUT_PREVIEW}")


if __name__ == "__main__":
    main()
