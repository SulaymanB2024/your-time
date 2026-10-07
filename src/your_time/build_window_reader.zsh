#!/bin/zsh
set -eu
umask 077

source_dir=/path/to/your-time/native
build_dir=/path/to/builds/your-time-window-reader-build
app="$build_dir/YourTimeWindowReader.app"
mkdir -p "$app/Contents/MacOS"
cp "$source_dir/WindowReader-Info.plist" "$app/Contents/Info.plist"
/usr/bin/clang -fobjc-arc -O2 -Wall -Wextra -framework Foundation -framework AppKit -framework ApplicationServices \
  "$source_dir/YourTimeWindowReader.m" -o "$app/Contents/MacOS/window-reader"
/usr/bin/codesign --force --sign - "$app"
/usr/bin/codesign --verify --deep --strict "$app"
# Use conventional permissions for this code-only application bundle.
# Captured activity files remain owner-only in the private state directory.
chmod 755 "$app" "$app/Contents" "$app/Contents/MacOS" "$app/Contents/_CodeSignature" "$app/Contents/MacOS/window-reader"
chmod 644 "$app/Contents/Info.plist" "$app/Contents/_CodeSignature/CodeResources"
print "$app"
