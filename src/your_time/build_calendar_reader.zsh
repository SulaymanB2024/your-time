#!/bin/zsh
set -eu
umask 077

source_dir=/path/to/your-time/native
build_dir=/path/to/builds/your-time-calendar-reader-build
app="$build_dir/YourTimeCalendarReader.app"
mkdir -p "$app/Contents/MacOS"
cp "$source_dir/CalendarReader-Info.plist" "$app/Contents/Info.plist"
/usr/bin/clang -fobjc-arc -O2 -framework Foundation -framework EventKit \
  "$source_dir/YourTimeCalendarReader.m" -o "$app/Contents/MacOS/calendar-reader"
/usr/bin/codesign --force --sign - "$app"
/usr/bin/codesign --verify "$app"
print "$app"
