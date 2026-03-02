#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <image_dir> <image_ref> <tag_ref>" >&2
  exit 2
fi

IMAGE_DIR="$1"
IMAGE_REF="$2"
TAG_REF="$3"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/vuln_logs"

mkdir -p "$LOG_DIR"

docker build --platform linux/amd64 -t "$IMAGE_REF" "$IMAGE_DIR" --build-arg tag_ref_name="$TAG_REF"

echo "Running pip-audit scan for vulnerabilities..."
if ! docker run --rm -v "$LOG_DIR:/work" "$IMAGE_REF" \
  sh -lc "/opt/venv/bin/python -m pip_audit --format json > /work/audit_report.json"; then
  echo "pip-audit found vulnerabilities" >&2
  echo "Check $LOG_DIR/audit_report.json for details." >&2
  exit 1
fi

echo "pip-audit clean: no vulnerabilities found."

echo "Running Trivy scan for HIGH and CRITICAL vulnerabilities..."
if ! trivy image --severity HIGH,CRITICAL --exit-code 1 \
  --format json --output "$LOG_DIR/trivy_report.json" "$IMAGE_REF"; then
  echo "Trivy found HIGH/CRITICAL vulnerabilities" >&2
  echo "Check $LOG_DIR/trivy_report.json for details." >&2
  exit 1
fi

echo "Trivy clean: no HIGH or CRITICAL vulnerabilities found."

docker push "$IMAGE_REF"
