#!/usr/bin/env bash

# Build the neural-network into a library JAR
# Output: dist/neural-network.jar
#
# Options:
#   --clean           wipe previous build before compiling
#   --target=<ver>    set Java target (default 8). Examples: --target=8
#
# Notes:
# - On JDK 9+, this uses:    javac --release <TARGET>
# - On JDK 8, falls back to: javac -source <TARGET> -target <TARGET>
# - If running JDK 8 and TARGET > 8, the script will error
# - The script excludes test/ directory from build

set -euo pipefail

# --- utils ---
die() { echo "ERROR: $*" >&2; exit 1; }
info() { echo -e "\033[1;34m[INFO]\033[0m $*"; }

# --- config / paths ---
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SRC_DIR="$ROOT/src"
BLD_DIR="$ROOT/build"
CLS_DIR="$BLD_DIR/classes"
DST_DIR="$ROOT/dist"

JAVAC="${JAVAC:-javac}"
JAR="${JAR:-jar}"

TARGET="8"
DO_CLEAN=false

# --- parse args ---
for arg in "$@"; do
    case "$arg" in
        --clean)     DO_CLEAN=true ;;
        --target=*)  TARGET="${arg#*=}" ;;
        *)           die "Unknown option: $arg" ;;
    esac
done

# --- normalize 1.8 to 8, etc ---
if [[ "$TARGET" =~ ^1\.([0-9]+)$ ]]; then
    TARGET="${BASH_REMATCH[1]}"
fi

# --- sanity checks ---
command -v "$JAVAC" >/dev/null || die "Command javac not found in PATH"
command -v "$JAR"   >/dev/null || die "Command jar not found in PATH"
[[ -d "$SRC_DIR" ]] || die "Source folder not found: $SRC_DIR"

# --- clean if requested ---
if $DO_CLEAN; then
    info "Cleaning previous build/dist"
    rm -rf "$BLD_DIR" "$DST_DIR"
fi

# --- prepare dirs ---
mkdir -p "$CLS_DIR" "$DST_DIR"

# --- gather sources, excluding test package ---
find "$SRC_DIR" -name '*.java' ! -path "*/test/*" | sort > "$BLD_DIR/sources.txt"
NUM_SOURCES=$(wc -l < "$BLD_DIR/sources.txt" | tr -d '[:space:]')
(( NUM_SOURCES > 0 )) || die "No .java files found under $SRC_DIR (excluding tests)"

# --- detect JDK and choose compile flags ---
if "$JAVAC" --help 2>/dev/null | grep -q -- '--release'; then
    USE_RELEASE=true
else
    USE_RELEASE=false
fi

# --- if JDK 8 (no --release) and target > 8 then error ---
if ! $USE_RELEASE && [[ "$TARGET" -gt 8 ]]; then
    die "JDK 8 detected (no --release). Use --target=8 or run with JDK 11+/17+"
fi

# --- compile ---
if $USE_RELEASE; then
    info "Compiling $NUM_SOURCES sources with: --release $TARGET"
    "$JAVAC" -encoding UTF-8 -g --release "$TARGET" -d "$CLS_DIR" @"$BLD_DIR/sources.txt"
else
    info "Compiling $NUM_SOURCES sources with: -source $TARGET -target $TARGET (JDK 8)"
    "$JAVAC" -encoding UTF-8 -g -source "$TARGET" -target "$TARGET" -d "$CLS_DIR" @"$BLD_DIR/sources.txt"
fi

# --- manifest ---
echo "Manifest-Version: 1.0" > "$BLD_DIR/MANIFEST.MF"

# --- jar and sources ---
OUT_JAR="$DST_DIR/neural-network.jar"
OUT_SRC_JAR="$DST_DIR/neural-network-sources.jar"

info "Packaging binary jar to $OUT_JAR"
"$JAR" cfm "$OUT_JAR" "$BLD_DIR/MANIFEST.MF" -C "$CLS_DIR" .

info "Packaging sources jar to $OUT_SRC_JAR"
"$JAR" cf "$OUT_SRC_JAR" -C "$SRC_DIR" .

info "Done. JAR at: $OUT_JAR"
