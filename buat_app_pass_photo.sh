#!/bin/bash
# Membuat "Pass Photo.app" di Desktop (ikon merah + siluet pasfoto).
# Jalankan sekali: bash buat_app_pass_photo.sh
set -e

PROJECT="/Users/gin2/Documents/ALWILDAN/PHOTO IJAZAH"
APP="$HOME/Desktop/Pass Photo.app"

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

# ---------- 1. Gambar ikon (PIL) ----------
TMP=$(mktemp -d)
python3 - "$TMP/icon.png" <<'PY'
import sys
from PIL import Image, ImageDraw

S = 1024
RED = (219, 21, 21, 255)
WHITE = (255, 255, 255, 255)
BLUE = (0, 114, 206, 255)

# mask kotak membulat
mask = Image.new("L", (S, S), 0)
ImageDraw.Draw(mask).rounded_rectangle((60, 60, S - 60, S - 60), radius=210, fill=255)

layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
d = ImageDraw.Draw(layer)
d.rectangle((0, 0, S, S), fill=RED)

# kartu pasfoto putih dengan bingkai
d.rounded_rectangle((250, 190, S - 250, S - 190), radius=40, fill=WHITE)
d.rectangle((285, 225, S - 285, S - 300), fill=RED)

# siluet kepala + bahu
d.ellipse((392, 300, 632, 540), fill=WHITE)
d.ellipse((312, 560, 712, 960), fill=WHITE)

# strip tulisan (garis hitam kecil) di bawah foto
d.rectangle((330, S - 270, S - 330, S - 245), fill=(30, 30, 30, 255))

# potong sisa agar sesuai kotak membulat
d.rectangle((0, S - 190, S, S), fill=RED)  # tutup bagian bawah kartu
d.rounded_rectangle((250, 190, S - 250, S - 190), radius=40, outline=WHITE, width=0)

out = Image.new("RGBA", (S, S), (0, 0, 0, 0))
out.paste(layer, (0, 0), mask)
out.save(sys.argv[1])
PY

ICONSET="$TMP/icon.iconset"
mkdir -p "$ICONSET"
for s in 16 32 128 256 512; do
  sips -z $s $s "$TMP/icon.png" --out "$ICONSET/icon_${s}x${s}.png" >/dev/null
  sips -z $((s*2)) $((s*2)) "$TMP/icon.png" --out "$ICONSET/icon_${s}x${s}@2x.png" >/dev/null
done
iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/AppIcon.icns"

# ---------- 2. Eksekusi app: buka Terminal & jalankan ----------
cat > "$APP/Contents/MacOS/PassPhoto" <<EOF
#!/bin/bash
osascript <<'OSA'
tell application "Terminal"
  activate
  do script "pkill -f 'pasfoto_app.py' 2>/dev/null || true; lsof -ti :7860 | xargs kill -9 2>/dev/null || true; cd \"$PROJECT\" && if [ -f venv/bin/activate ]; then source venv/bin/activate; fi; python3 pasfoto_app.py"
end tell
OSA
EOF
chmod +x "$APP/Contents/MacOS/PassPhoto"

# ---------- 3. Info.plist ----------
cat > "$APP/Contents/Info.plist" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>Pass Photo</string>
  <key>CFBundleDisplayName</key><string>Pass Photo</string>
  <key>CFBundleIdentifier</key><string>local.alwildan.passphoto</string>
  <key>CFBundleExecutable</key><string>PassPhoto</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleVersion</key><string>1.0</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>LSUIElement</key><false/>
</dict>
</plist>
EOF

xattr -cr "$APP" 2>/dev/null || true
touch "$APP"
rm -rf "$TMP"

echo "Selesai: $APP"
echo "Klik dua kali 'Pass Photo' di Desktop."
