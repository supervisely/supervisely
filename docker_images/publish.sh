#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <image_dir> <image_ref>" >&2
  exit 2
fi

IMAGE_DIR="$1"
IMAGE_REF="$2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"

mkdir -p "$LOG_DIR"

docker build --platform linux/amd64 -t "$IMAGE_REF" "$IMAGE_DIR"

docker run --rm -v "$LOG_DIR:/work" "$IMAGE_REF" \
  sh -lc "python -m pip install -q pip-audit && pip-audit --format json > /work/audit_report.json"

trivy image --severity HIGH,CRITICAL \
  --format json --output "$LOG_DIR/trivy_report.json" "$IMAGE_REF"

docker push "$IMAGE_REF"
