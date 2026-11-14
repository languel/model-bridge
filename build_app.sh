#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$ROOT_DIR/app"
APP_BUNDLE="$ROOT_DIR/ModelBridge.app"

echo "[*] Building ModelBridge (Release)..."
cd "$APP_DIR"
swift build -c release

BIN_PATH="$(swift build -c release --show-bin-path)/ModelBridge"
if [[ ! -f "$BIN_PATH" ]]; then
    echo "[-] Could not locate compiled binary at $BIN_PATH"
    exit 1
fi

echo "[*] Packaging $APP_BUNDLE..."
rm -rf "$APP_BUNDLE"
mkdir -p "$APP_BUNDLE/Contents/MacOS"
mkdir -p "$APP_BUNDLE/Contents/Resources"

cp "$BIN_PATH" "$APP_BUNDLE/Contents/MacOS/ModelBridge"
chmod +x "$APP_BUNDLE/Contents/MacOS/ModelBridge"

cat > "$APP_BUNDLE/Contents/Info.plist" << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleExecutable</key>
    <string>ModelBridge</string>
    <key>CFBundleIdentifier</key>
    <string>com.languel.modelbridge</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>ModelBridge</string>
    <key>CFBundleDisplayName</key>
    <string>Model Bridge</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0.0</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>LSMinimumSystemVersion</key>
    <string>14.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
EOF

echo "[+] Built $APP_BUNDLE successfully!"
echo "    Launch via: open \"$APP_BUNDLE\""
