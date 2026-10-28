"""
Pasfoto Batch App (Gradio)
Jalankan: python pasfoto_app.py
Buka    : http://127.0.0.1:7860 (otomatis terbuka di browser)

Install:
    pip install -U gradio rembg onnxruntime opencv-python pillow numpy
"""
import io
import tempfile
import zipfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from rembg import new_session, remove

SIZES = {
    "2x3 cm (236x354)": (236, 354),
    "3x4 cm (472x591)": (472, 591),
    "4x6 cm (591x827)": (591, 827),
}
COLORS = {"Merah": "#DB1515", "Biru": "#0072CE", "Putih": "#FFFFFF", "Custom": None}
MODELS = ["birefnet-portrait", "isnet-general-use", "u2net_human_seg"]
RAW_EXTS = {".cr3", ".cr2", ".nef", ".arw", ".dng"}
EXTS = {".jpg", ".jpeg", ".png", ".webp"} | RAW_EXTS
MAX_SIDE = 2400  # sisi terpanjang saat diproses (hasil akhir kecil, jadi cukup; jauh lebih cepat)

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
    "Arial.ttf",
    "arial.ttf",
]

_sessions = {}
_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Global in-memory cache untuk edit manual instan (~12ms per render tanpa rembg ulang)
_processed_cache = {}
_photo_order = []
_current_out_dir = None


def get_session(name):
    if name not in _sessions:
        _sessions[name] = new_session(name)
    return _sessions[name]


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def load_image(path):
    """Buka JPG/PNG/WEBP atau RAW (CR3 dll) jadi PIL RGB, sudah terputar & diperkecil."""
    if path.suffix.lower() in RAW_EXTS:
        import rawpy

        with rawpy.imread(str(path)) as raw:
            # Utamakan thumbnail JPEG penuh dari ISP kamera (warna & exposure identik dengan JPG, 50x lebih cepat)
            img = None
            try:
                thumb = raw.extract_thumb()
                if thumb.format == rawpy.ThumbFormat.JPEG:
                    img = ImageOps.exif_transpose(Image.open(io.BytesIO(thumb.data)))
            except Exception:
                img = None

            if img is None:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    output_bps=8,
                    output_color=rawpy.ColorSpace.sRGB,
                    no_auto_bright=True,
                )
                img = Image.fromarray(rgb)
        img = img.convert("RGB")
    else:
        with Image.open(path) as im:
            img = ImageOps.exif_transpose(im).convert("RGB")

    img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    return img


def detect_face(img):
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    min_size = max(60, img.width // 12)
    faces = _cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_size, min_size)
    )
    if len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2] * f[3])


def edge_touch(cut, thresh=0.04):
    """(kiri, kanan, bawah): True kalau badan menempel/terpotong di tepi foto sumber."""
    a = np.asarray(cut.split()[-1]) > 128
    return (
        bool(a[:, :3].mean() > thresh),
        bool(a[:, -3:].mean() > thresh),
        bool(a[-3:, :].mean() > thresh),
    )


def crop_pasfoto(img, face, size, fill, crop_h_ratio, top_margin, head_top=None,
                 body=None, side=0.04, edges=(False, False, False),
                 zoom=1.0, offset_x=0.0, offset_y=0.0):
    iw, ih = img.size
    ratio = size[0] / size[1]

    if face is None:
        crop_h = ih * 0.8 / max(zoom, 0.1)
        crop_w = crop_h * ratio
        cx = iw / 2.0 - offset_x * crop_w
        cy = ih / 2.0 + offset_y * crop_h
        left = cx - crop_w / 2.0
        top = cy - crop_h / 2.0
        right, bottom = left + crop_w, top + crop_h
        pad = int(max(0, -left, -top, right - iw, bottom - ih)) + 1
        if pad > 1:
            img = ImageOps.expand(img, border=pad, fill=fill)
            left, right, top, bottom = left + pad, right + pad, top + pad, bottom + pad
        box = tuple(int(round(v)) for v in (left, top, right, bottom))
        return img.crop(box).resize(size, Image.LANCZOS)

    t_left, t_right, t_bottom = edges
    x, y, w, h = [int(v) for v in face]
    crop_h = h * crop_h_ratio
    crop_w = crop_h * ratio
    if body:  # zoom otomatis: lebarkan crop sampai bahu/hijab muat + ruang samping
        need_w = (body[1] - body[0]) / max(1 - 2 * side, 0.5)
        if need_w > crop_w:
            crop_w = need_w
            crop_h = crop_w / ratio
    # badan menempel di kiri DAN kanan foto sumber: tidak ada gambar lebih, jangan lebihi lebar foto
    if t_left and t_right and crop_w > iw:
        crop_w = float(iw)
        crop_h = crop_w / ratio

    cx = x + w / 2
    if head_top is None:  # fallback perkiraan dari wajah
        head_top = y - 0.25 * h
    top = head_top - top_margin * crop_h
    left = cx - crop_w / 2
    if body:  # geser agar seluruh bahu masuk dengan ruang samping
        m = side * crop_w
        left = min(left, body[0] - m)
        left = max(left, body[1] + m - crop_w)

    def clamp_x(lf):
        # sisi yang badannya terpotong tidak boleh diisi merah
        if t_left and lf < 0:
            lf = 0.0
        if t_right and lf + crop_w > iw:
            lf = iw - crop_w
        return lf

    left = clamp_x(left)

    # badan terpotong di bawah foto sumber: jangan sampai ada strip merah di bawah
    if t_bottom and top + crop_h > ih:
        dy = top + crop_h - ih
        shift = max(0.0, min(dy, top - (head_top - 0.22 * crop_h)))  # geser naik, maks 22% ruang atas
        top -= shift
        if top + crop_h > ih:  # masih kurang: perkecil crop (zoom in)
            new_h = ih - top
            s = new_h / crop_h
            ccx = left + crop_w / 2
            crop_h = new_h
            crop_w *= s
            left = clamp_x(ccx - crop_w / 2)

    # Fine-tuning manual jika zoom atau offset digeser
    if zoom != 1.0 or offset_x != 0.0 or offset_y != 0.0:
        base_cx = left + crop_w / 2.0
        if zoom > 0 and zoom != 1.0:
            crop_w = crop_w / zoom
            crop_h = crop_w / ratio
            # Kunci margin kepala agar saat zoom in kepala tidak terpotong & baju bawah tersembunyi
            top = head_top - top_margin * crop_h

        # Geser manual (offset_x: + geser subjek ke kanan; offset_y: + geser subjek ke atas)
        left = (base_cx - crop_w / 2.0) - offset_x * crop_w
        top = top + offset_y * crop_h

    right, bottom = left + crop_w, top + crop_h

    # pad merah hanya untuk area yang memang background (atas kepala dsb.)
    pad = int(max(0, -left, -top, right - img.width, bottom - img.height)) + 1
    if pad > 1:
        img = ImageOps.expand(img, border=pad, fill=fill)
        left, right, top, bottom = left + pad, right + pad, top + pad, bottom + pad

    box = tuple(int(round(v)) for v in (left, top, right, bottom))
    return img.crop(box).resize(size, Image.LANCZOS)


def load_font(px):
    for f in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(f, px)
        except OSError:
            continue
    try:
        return ImageFont.load_default(px)
    except Exception:
        return ImageFont.load_default()


def layout(size, frame, caption):
    """Return (margin, strip_h, photo_w, photo_h) di dalam kanvas size."""
    W, H = size
    m = round(W * 0.035) if frame else 0
    strip = round(H * 0.085) if caption else m
    return m, strip, W - 2 * m, H - m - strip


def compose(photo, size, m, strip, caption):
    """Kanvas putih + foto + tulisan di strip bawah."""
    W, H = size
    canvas = Image.new("RGB", size, (255, 255, 255))
    canvas.paste(photo, (m, m))
    if caption:
        draw = ImageDraw.Draw(canvas)
        max_w = W - 2 * max(m, round(W * 0.03))
        px = int(strip * 0.6)
        font = load_font(px)
        while px > 8 and draw.textlength(caption, font=font) > max_w:
            px -= 1
            font = load_font(px)
        draw.text((W / 2, H - strip / 2), caption, font=font,
                  fill=(0, 0, 0), anchor="mm")
    return canvas


def body_extent(cut, face):
    """(kiri, kanan) piksel terlebar badan/hijab di sekitar level bahu, dari mask."""
    if face is None:
        return None
    alpha = np.asarray(cut.split()[-1])
    H, W = alpha.shape
    x, y, w, h = [int(v) for v in face]
    r0 = min(y + int(h * 1.9), H - 2)
    r1 = min(y + int(h * 2.3), H - 1)
    if r1 <= r0:
        return None
    band = alpha[r0:r1] > 128
    rows = band.any(axis=1)
    if rows.sum() < 3:
        return None
    lefts = band.argmax(axis=1)[rows]
    rights = (W - 1 - band[:, ::-1].argmax(axis=1))[rows]
    return float(np.percentile(lefts, 10)), float(np.percentile(rights, 90))


def _lum(a):
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def adjust_foreground(cut, face, auto=True, strength=0.8, bright=0, warm=0):
    """Koreksi orang saja (RGBA hasil cutout), background merah tidak ikut berubah.
    auto    : normalisasi warna kulit & kecerahan seragam berdasarkan deteksi kulit
    strength: 0..1 seberapa kuat normalisasi otomatis
    bright  : -50..50 koreksi manual kecerahan
    warm    : -30..30 koreksi manual (positif = lebih hangat, negatif = lebih dingin)
    """
    rgb = np.asarray(cut.convert("RGB")).astype(np.float32) / 255.0
    alpha = np.asarray(cut.split()[-1]).astype(np.float32) / 255.0
    H, W = alpha.shape
    fg = alpha > 0.9

    if auto and fg.sum() > 1000 and face is not None:
        x, y, w, h = [int(v) for v in face]
        face_rgb = rgb[max(y, 0):min(y + h, H), max(x, 0):min(x + w, W)]

        if face_rgb.size > 0:
            face_uint8 = (face_rgb * 255).astype(np.uint8)
            ycrcb = cv2.cvtColor(face_uint8, cv2.COLOR_RGB2YCrCb)
            # Mask kulit manusia standar di ruang warna YCrCb (kebal rambut, kerudung, mata, dan bayangan)
            skin_mask = (
                (ycrcb[..., 1] >= 133) & (ycrcb[..., 1] <= 173) &
                (ycrcb[..., 2] >= 77) & (ycrcb[..., 2] <= 127)
            )
            face_lum = _lum(face_rgb)

            if np.sum(skin_mask) > 100:
                cur_skin = float(np.median(face_lum[skin_mask]))
            else:
                cur_skin = float(np.percentile(face_lum, 65))

            # Target skin luminance untuk pasfoto ijazah standar (128 / 255 = ~0.50)
            target = 0.50
            cur_skin = np.clip(cur_skin, 0.12, 0.90)

            # Hitung gamma exposure
            g = float(np.clip(np.log(target) / np.log(cur_skin), 0.50, 1.40))
            eff_g = 1.0 + (g - 1.0) * strength

            # Tone curve midtone lift dengan proteksi bayangan/rambut (shadow anchor)
            # agar rambut hitam tetap hitam pekat alami dan tidak menjadi abu-abu pudar
            w_mat = np.sin(np.clip(rgb, 0.0, 1.0) * np.pi) ** 0.55
            local_g = 1.0 + (eff_g - 1.0) * w_mat
            rgb = np.power(np.clip(rgb, 1e-6, 1.0), local_g)

        # Mild neutral white balance pada pakaian putih (jika terdeteksi stabil)
        y0 = int(face[1] + face[3] * 1.3)
        rows = np.zeros_like(fg)
        rows[min(max(y0, 0), H - 1):, :] = True
        cand = fg & rows
        if cand.sum() > 500:
            lum = _lum(rgb)
            thr = np.percentile(lum[cand], 85)
            sel = cand & (lum >= thr)
            if sel.sum() > 300:
                m = rgb[sel].mean(axis=0)
                sat = (m.max() - m.min()) / max(m.max(), 1e-6)
                if sat < 0.15 and m.min() > 0.40:  # benar-benar kain putih/netral
                    gains = np.clip(m.mean() / m, 0.90, 1.10)
                    gains = 1.0 + (gains - 1.0) * (strength * 0.5)
                    rgb = np.clip(rgb * gains, 0, 1)

    # --- Koreksi manual (slider) ---
    if bright:
        rgb = np.power(np.clip(rgb, 0, 1), 2.0 ** (-bright / 100.0))
    if warm:
        k = warm / 200.0
        rgb = np.clip(rgb * np.array([1 + k, 1.0, 1 - k], dtype=np.float32), 0, 1)

    out = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8), "RGB").convert("RGBA")
    out.putalpha(cut.split()[-1])
    return out


def apply_manual_grade(cut, bright=0, warm=0):
    """Koreksi manual kecerahan dan warmth pada foreground (RGBA). Sangat cepat (~5ms)."""
    if bright == 0 and warm == 0:
        return cut
    rgb = np.asarray(cut.convert("RGB")).astype(np.float32) / 255.0
    if bright:
        rgb = np.power(np.clip(rgb, 0, 1), 2.0 ** (-bright / 100.0))
    if warm:
        k = warm / 200.0
        rgb = np.clip(rgb * np.array([1 + k, 1.0, 1 - k], dtype=np.float32), 0, 1)
    out = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8), "RGB").convert("RGBA")
    out.putalpha(cut.split()[-1])
    return out


def render_item(item, zoom=1.0, offset_x=0.0, offset_y=0.0, bright=0, warm=0):
    """Render instan (~12ms) menggunakan cut_base yang sudah tersimpan tanpa panggil rembg ulang."""
    cut_base = item["cut_base"]
    bg_rgb = item["bg_rgb"]
    size = item["size"]
    face = item["face"]
    head_top = item["head_top"]
    body = item["body"]
    edges = item["edges"]
    crop_h_ratio = item["crop_h_ratio"]
    top_margin = item["top_margin"]
    side = item["side"]
    frame = item["frame"]
    caption = item["caption"]

    cut = apply_manual_grade(cut_base, bright, warm)
    bg = Image.new("RGBA", cut.size, bg_rgb + (255,))
    result = Image.alpha_composite(bg, cut).convert("RGB")

    if frame or caption:
        m, strip, pw, ph = layout(size, frame, caption)
        photo = crop_pasfoto(
            result, face, (pw, ph), bg_rgb, crop_h_ratio, top_margin, head_top,
            body, side, edges, zoom=zoom, offset_x=offset_x, offset_y=offset_y
        )
        return compose(photo, size, m, strip, caption)
    else:
        return crop_pasfoto(
            result, face, size, bg_rgb, crop_h_ratio, top_margin, head_top,
            body, side, edges, zoom=zoom, offset_x=offset_x, offset_y=offset_y
        )


def process_one(path, bg_rgb, size, session, matting, crop_h_ratio, top_margin,
                caption="", frame=False, auto=True, strength=0.8, bright=0, warm=0,
                auto_zoom=True, side=0.04):
    img = load_image(path)
    face = detect_face(img)

    kwargs = {}
    if matting:
        kwargs = dict(
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=10,
        )
    cut = remove(img, session=session, **kwargs)
    # Simpan hasil normalisasi otomatis sebagai cut_base murni
    cut_base = adjust_foreground(cut, face, auto=auto, strength=strength, bright=0, warm=0)

    # puncak kepala/hijab dari mask (lebih akurat dari perkiraan wajah)
    mask = cut_base.split()[-1].point(lambda v: 255 if v > 128 else 0)
    bbox = mask.getbbox()
    head_top = bbox[1] if bbox else None
    body = body_extent(cut_base, face) if auto_zoom else None
    edges = edge_touch(cut_base)

    cache_data = {
        "cut_base": cut_base,
        "bg_rgb": bg_rgb,
        "size": size,
        "face": face,
        "head_top": head_top,
        "body": body,
        "edges": edges,
        "crop_h_ratio": crop_h_ratio,
        "top_margin": top_margin,
        "side": side,
        "frame": frame,
        "caption": caption,
        "zoom": 1.0,
        "offset_x": 0.0,
        "offset_y": 0.0,
        "bright": bright,
        "warm": warm,
        "version": 0,
    }

    result = render_item(cache_data, zoom=1.0, offset_x=0.0, offset_y=0.0, bright=bright, warm=warm)
    return result, face is not None, cache_data


def _extract_path(f):
    if isinstance(f, str):
        return Path(f)
    if hasattr(f, "path") and f.path:
        return Path(f.path)
    if hasattr(f, "name") and f.name:
        return Path(f.name)
    if isinstance(f, dict) and "path" in f:
        return Path(f["path"])
    return Path(str(f))


def run(files, folder_path, warna, custom_hex, ukuran, caption, frame, model, matting,
        crop_h_ratio, top_margin, auto, strength, bright, warm, auto_zoom, side,
        progress=gr.Progress()):
    global _processed_cache, _photo_order, _current_out_dir
    _processed_cache.clear()
    _photo_order.clear()

    paths = []
    log_folder_warn = None

    if files:
        paths.extend([_extract_path(f) for f in files])

    if folder_path and folder_path.strip():
        fdir = Path(folder_path.strip().strip('"').strip("'"))
        if fdir.exists() and fdir.is_dir():
            for p in sorted(fdir.iterdir()):
                if p.is_file() and p.suffix.lower() in EXTS:
                    paths.append(p)
        else:
            log_folder_warn = f"Folder tidak ditemukan: {fdir}"

    seen = set()
    raw_paths = []
    for p in paths:
        if p.suffix.lower() in EXTS and str(p) not in seen and p.exists():
            seen.add(str(p))
            raw_paths.append(p)

    # Cegah duplikasi jika ada file RAW dan JPG untuk foto yang sama (misal _MG_8971.CR3 dan _MG_8971.JPG)
    # Utamakan JPG untuk kecepatan decoding instan
    stem_map = {}
    for p in raw_paths:
        stem = p.stem.lower()
        if stem not in stem_map:
            stem_map[stem] = p
        else:
            if stem_map[stem].suffix.lower() in RAW_EXTS and p.suffix.lower() not in RAW_EXTS:
                stem_map[stem] = p
    valid_paths = sorted(list(stem_map.values()), key=lambda p: p.name)

    if not valid_paths:
        err_msg = "Upload foto dulu atau masukkan path folder yang berisi foto (.jpg, .png, .cr3, dsb.)."
        if log_folder_warn:
            err_msg += f" ({log_folder_warn})"
        raise gr.Error(err_msg)

    bg_rgb = hex_to_rgb(custom_hex if warna == "Custom" else COLORS[warna])
    size = SIZES[ukuran]
    session = get_session(model)

    out_dir = Path(tempfile.mkdtemp(prefix="pasfoto_"))
    _current_out_dir = out_dir

    gallery, log, used = [], [], set()
    if log_folder_warn:
        log.append(f"PERINGATAN: {log_folder_warn}")

    for p in progress.tqdm(valid_paths, desc="Memproses"):
        try:
            result, found, cache_data = process_one(
                p, bg_rgb, size, session, matting, crop_h_ratio, top_margin,
                caption.strip(), frame, auto, strength, bright, warm,
                auto_zoom, side,
            )
            name = f"{p.stem}_{warna.lower()}"
            n, k = name, 1
            while n in used:
                k += 1
                n = f"{name}_{k}"
            used.add(n)
            out = out_dir / f"{n}.jpg"
            result.save(out, quality=95, subsampling=0, dpi=(300, 300))

            cache_data["orig_name"] = p.name
            cache_data["display_name"] = n
            cache_data["out_path"] = out

            _processed_cache[p.name] = cache_data
            _photo_order.append(p.name)

            gallery.append((str(out), p.name))
            log.append(("OK    " if found else "OK*   ") + p.name
                       + ("" if found else "  (wajah tidak terdeteksi, crop default)"))
        except Exception as e:
            log.append(f"GAGAL {p.name}: {e}")

    zip_path = out_dir / "pasfoto_hasil.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for img_path, _ in gallery:
            z.write(img_path, Path(img_path).name)

    import gc
    gc.collect()

    ok = len(gallery)
    log.append(f"\nSelesai: {ok}/{len(valid_paths)} berhasil.")

    first_choice = _photo_order[0] if _photo_order else None
    first_preview = None
    if first_choice and first_choice in _processed_cache:
        it = _processed_cache[first_choice]
        first_preview = render_item(it, it["zoom"], it["offset_x"], it["offset_y"], it["bright"], it["warm"])

    dropdown_update = gr.update(choices=_photo_order, value=first_choice)
    status_update = f"💡 Foto **{first_choice}** siap diedit. Geser slider untuk mengatur posisi, zoom, atau kecerahan." if first_choice else ""

    return (
        gallery,
        str(zip_path),
        "\n".join(log),
        dropdown_update,
        first_preview,
        1.0,
        0.0,
        0.0,
        0,
        0,
        status_update,
    )


def on_gallery_select(evt: gr.SelectData):
    if evt.index is not None and 0 <= evt.index < len(_photo_order):
        key = _photo_order[evt.index]
        item = _processed_cache[key]
        prev = render_item(item, item["zoom"], item["offset_x"], item["offset_y"], item["bright"], item["warm"])
        stat = f"🔍 Sedang mengedit: **{item['orig_name']}**"
        return gr.update(value=key), prev, item["zoom"], item["offset_x"], item["offset_y"], item["bright"], item["warm"], stat
    return gr.update(), None, 1.0, 0.0, 0.0, 0, 0, ""


def on_dropdown_select(key):
    if key and key in _processed_cache:
        item = _processed_cache[key]
        prev = render_item(item, item["zoom"], item["offset_x"], item["offset_y"], item["bright"], item["warm"])
        stat = f"🔍 Sedang mengedit: **{item['orig_name']}**"
        return prev, item["zoom"], item["offset_x"], item["offset_y"], item["bright"], item["warm"], stat
    return None, 1.0, 0.0, 0.0, 0, 0, ""


def on_slider_change(key, zoom, shift_x, shift_y, bright, warm):
    if not key or key not in _processed_cache:
        return None
    item = _processed_cache[key]
    return render_item(item, zoom, shift_x, shift_y, bright, warm)


def on_apply_edit(key, zoom, shift_x, shift_y, bright, warm):
    if not key or key not in _processed_cache:
        return gr.update(), gr.update(), None, "⚠️ Pilih foto terlebih dahulu."
    item = _processed_cache[key]
    item["zoom"] = float(zoom)
    item["offset_x"] = float(shift_x)
    item["offset_y"] = float(shift_y)
    item["bright"] = int(bright)
    item["warm"] = int(warm)
    item["version"] += 1

    final_img = render_item(item, zoom, shift_x, shift_y, bright, warm)
    out_dir = item["out_path"].parent
    new_out = out_dir / f"{item['display_name']}_v{item['version']}.jpg"
    final_img.save(new_out, quality=95, subsampling=0, dpi=(300, 300))

    old_out = item["out_path"]
    if old_out.exists() and old_out != new_out:
        try:
            old_out.unlink()
        except Exception:
            pass
    item["out_path"] = new_out

    # Update file ZIP
    zip_path = out_dir / "pasfoto_hasil.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for k in _photo_order:
            it = _processed_cache[k]
            z.write(it["out_path"], f"{it['display_name']}.jpg")

    new_gallery = [(str(_processed_cache[k]["out_path"]), _processed_cache[k]["orig_name"]) for k in _photo_order]
    stat = f"✅ Perubahan untuk **{item['orig_name']}** berhasil disimpan & ZIP telah diperbarui!"
    return new_gallery, str(zip_path), final_img, stat


def on_reset_edit(key):
    if not key or key not in _processed_cache:
        return gr.update(), gr.update(), None, 1.0, 0.0, 0.0, 0, 0, ""
    new_gallery, zip_file, prev, _ = on_apply_edit(key, 1.0, 0.0, 0.0, 0, 0)
    stat = f"↺ Foto **{key}** telah direset ke posisi dan warna default."
    return new_gallery, zip_file, prev, 1.0, 0.0, 0.0, 0, 0, stat


with gr.Blocks(title="Pasfoto Batch") as demo:
    gr.Markdown("## Pasfoto Batch\nUpload foto atau masukkan path folder, pilih warna & ukuran, klik **Proses**, download ZIP.")

    with gr.Row():
        with gr.Column(scale=1):
            files = gr.File(
                label="Foto JPG/PNG atau RAW .CR3 (bisa banyak / drag & drop)",
                file_count="multiple",
                file_types=None,
            )
            folder_path = gr.Textbox(
                label="Atau Path Folder Lokal (Opsional - sangat praktis untuk ratusan foto)",
                placeholder="Contoh: /Users/gin2/Documents/ALWILDAN/PHOTO IJAZAH/...",
            )
            warna = gr.Radio(list(COLORS), value="Merah", label="Warna background")
            custom = gr.ColorPicker(value="#DB1515", label="Warna custom (pilih 'Custom')")
            ukuran = gr.Dropdown(list(SIZES), value="4x6 cm (591x827)", label="Ukuran")
            caption = gr.Textbox(value="SD AL-WILDAN ISLAMIC SCHOOL",
                                 label="Tulisan di bawah foto (kosongkan jika tidak perlu)")
            frame = gr.Checkbox(value=True, label="Bingkai putih di sekeliling foto")
            auto = gr.Checkbox(value=True,
                               label="Koreksi otomatis (warna & kecerahan wajah/baju)")
            auto_zoom = gr.Checkbox(value=True,
                                    label="Zoom otomatis (bahu/hijab tidak terpotong)")
            with gr.Accordion("Pengaturan lanjut", open=False):
                model = gr.Dropdown(MODELS, value="birefnet-portrait", label="Model")
                matting = gr.Checkbox(value=False, label="Alpha matting (tepi rambut lebih halus, lebih lambat)")
                crop_ratio = gr.Slider(1.4, 3.6, value=2.2, step=0.05,
                                       label="Zoom kepala minimum (kecil = wajah lebih besar)")
                top_margin = gr.Slider(0.0, 0.25, value=0.10, step=0.01,
                                       label="Jarak puncak kepala ke tepi atas")
                strength = gr.Slider(0.0, 1.0, value=0.8, step=0.05,
                                     label="Kekuatan koreksi otomatis")
                bright = gr.Slider(-50, 50, value=0, step=1,
                                   label="Kecerahan manual (+ lebih terang)")
                warm = gr.Slider(-30, 30, value=0, step=1,
                                 label="Warna manual (+ lebih hangat, - lebih dingin)")
                side = gr.Slider(0.0, 0.15, value=0.04, step=0.005,
                                 label="Ruang kiri-kanan bahu (zoom otomatis)")
            btn = gr.Button("Proses Batch", variant="primary")

        with gr.Column(scale=2):
            gallery = gr.Gallery(
                label="Hasil Pasfoto (Klik salah satu foto untuk mengedit posisi & kecerahan)",
                columns=4,
                height=360,
                object_fit="contain",
            )

            with gr.Accordion("🛠️ Edit Manual Foto Terpilih (Geser Posisi, Zoom, & Kecerahan)", open=True):
                with gr.Row():
                    photo_dropdown = gr.Dropdown(
                        label="Pilih Foto untuk Diedit (atau klik langsung di galeri)",
                        choices=[],
                        interactive=True,
                        scale=3,
                    )
                    btn_reset = gr.Button("↺ Reset ke Default", variant="secondary", scale=1)

                with gr.Row():
                    with gr.Column(scale=1):
                        preview_edit = gr.Image(
                            label="Live Preview Hasil Edit",
                            interactive=False,
                            height=360,
                            object_fit="contain",
                        )
                        edit_status = gr.Markdown("💡 *Pilih foto di galeri atau dropdown untuk mulai mengedit.*")
                    with gr.Column(scale=1):
                        edit_zoom = gr.Slider(0.7, 2.2, value=1.0, step=0.02,
                                              label="🔍 Zoom (Perbesar / Perkecil)")
                        edit_shift_x = gr.Slider(-0.35, 0.35, value=0.0, step=0.01,
                                                 label="↔️ Geser: Kiri (-) / Kanan (+)")
                        edit_shift_y = gr.Slider(-0.35, 0.35, value=0.0, step=0.01,
                                                 label="↕️ Geser: Turun (-) / Naik (+)")
                        edit_bright = gr.Slider(-50, 50, value=0, step=1,
                                                label="☀️ Kecerahan (+ Terang / - Gelap)")
                        edit_warm = gr.Slider(-30, 30, value=0, step=1,
                                              label="🎨 Warna (+ Hangat / - Dingin)")
                        btn_apply = gr.Button("💾 Terapkan ke Foto Ini & Update ZIP", variant="primary")

            zip_out = gr.File(label="Download ZIP Hasil Pasfoto")
            log = gr.Textbox(label="Log Proses", lines=5)

    # Event binding
    btn.click(
        run,
        [files, folder_path, warna, custom, ukuran, caption, frame, model, matting, crop_ratio, top_margin,
         auto, strength, bright, warm, auto_zoom, side],
        [gallery, zip_out, log, photo_dropdown, preview_edit, edit_zoom, edit_shift_x, edit_shift_y, edit_bright, edit_warm, edit_status],
    )

    gallery.select(
        on_gallery_select,
        inputs=None,
        outputs=[photo_dropdown, preview_edit, edit_zoom, edit_shift_x, edit_shift_y, edit_bright, edit_warm, edit_status],
    )

    photo_dropdown.change(
        on_dropdown_select,
        inputs=[photo_dropdown],
        outputs=[preview_edit, edit_zoom, edit_shift_x, edit_shift_y, edit_bright, edit_warm, edit_status],
    )

    for slider in [edit_zoom, edit_shift_x, edit_shift_y, edit_bright, edit_warm]:
        slider.input(
            on_slider_change,
            inputs=[photo_dropdown, edit_zoom, edit_shift_x, edit_shift_y, edit_bright, edit_warm],
            outputs=[preview_edit],
            show_progress="hidden",
        )
        slider.release(
            on_slider_change,
            inputs=[photo_dropdown, edit_zoom, edit_shift_x, edit_shift_y, edit_bright, edit_warm],
            outputs=[preview_edit],
            show_progress="hidden",
        )

    btn_apply.click(
        on_apply_edit,
        inputs=[photo_dropdown, edit_zoom, edit_shift_x, edit_shift_y, edit_bright, edit_warm],
        outputs=[gallery, zip_out, preview_edit, edit_status],
    )

    btn_reset.click(
        on_reset_edit,
        inputs=[photo_dropdown],
        outputs=[gallery, zip_out, preview_edit, edit_zoom, edit_shift_x, edit_shift_y, edit_bright, edit_warm, edit_status],
    )

if __name__ == "__main__":
    try:
        demo.queue().launch(inbrowser=True, server_port=7860, max_file_size="5gb")
    except OSError:
        demo.queue().launch(inbrowser=True, max_file_size="5gb")
