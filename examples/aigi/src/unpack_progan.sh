#!/usr/bin/env bash
# Unpack the CNNDetection ProGAN training set (progan_train.7z.001-007 -> progan_train.zip -> 20 class dirs)
# into DEST without touching the original archive.  Usage: bash src/unpack_progan.sh ARCHIVE_DIR DEST SEVENZIP_BIN
set -euo pipefail
SRC="$1"; DEST="$2"; SZ="$3"
mkdir -p "$DEST/_tmp"
if [ ! -f "$DEST/_tmp/progan_train.zip" ]; then
  "$SZ" x -y "$SRC/progan_train.7z.001" -o"$DEST/_tmp" > "$DEST/_tmp/7z.log"
fi
cd "$DEST"
if command -v unzip > /dev/null; then
  unzip -q -o "_tmp/progan_train.zip" -d "$DEST/_unzipped"
else
  "$PY" -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "_tmp/progan_train.zip" "$DEST/_unzipped"
fi
# Flatten: move the 20 class directories (those containing 0_real/1_fake) directly under DEST.
"$PY" - "$DEST" <<'EOF'
import sys
from pathlib import Path
dest = Path(sys.argv[1])
classes = sorted({p.parent for p in (dest / "_unzipped").rglob("0_real") if p.is_dir()})
assert len(classes) == 20, f"expected 20 ProGAN classes, found {len(classes)}"
for c in classes:
    c.rename(dest / c.name)
print("classes:", " ".join(c.name for c in classes))
EOF
rm -rf "$DEST/_unzipped" "$DEST/_tmp"
echo "UNPACKED $(find "$DEST" -type f | wc -l) files into $DEST"
