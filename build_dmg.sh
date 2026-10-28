#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================================="
echo "  Membangun Standalone App & DMG: Pass Photo Studio v1.0  "
echo "  Developed by gnzgn (@ginzagun) - GINZAGUN PRODUCTION   "
echo "=========================================================="

# 1. Clean previous builds
echo "==> Membersihkan build sebelumnya..."
rm -rf build dist

# 2. Build .app with PyInstaller
echo "==> Mengompilasi aplikasi mandiri dengan PyInstaller..."
python3 -m PyInstaller --noconfirm PassPhoto.spec

if [ ! -d "dist/Pass Photo.app" ]; then
    echo "ERROR: dist/Pass Photo.app gagal dibuat!"
    exit 1
fi

echo "==> dist/Pass Photo.app berhasil dibuat!"

# 3. Create DMG staging
echo "==> Menyiapkan folder staging untuk DMG..."
STAGING_DIR="dist/dmg_staging"
mkdir -p "$STAGING_DIR"
cp -R "dist/Pass Photo.app" "$STAGING_DIR/"
ln -s /Applications "$STAGING_DIR/Applications"

# 4. Generate .dmg using hdiutil
DMG_NAME="PassPhoto-v1.0-macOS-arm64.dmg"
OUTPUT_DMG="dist/$DMG_NAME"
echo "==> Mengemas installer DMG: $OUTPUT_DMG..."
rm -f "$OUTPUT_DMG"
hdiutil create -volname "Pass Photo Studio" -srcfolder "$STAGING_DIR" -ov -format UDZO "$OUTPUT_DMG"

# Clean staging
rm -rf "$STAGING_DIR"

echo "=========================================================="
echo "  SELESAI! Installer DMG berhasil dibuat:                 "
ls -lh "$OUTPUT_DMG"
echo "=========================================================="
