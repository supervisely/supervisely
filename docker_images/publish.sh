#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <image_dir> <image_ref> <tag_ref>" >&2
  exit 2
fi

# bash docker_images/publish.sh docker_images/system_hardened supervisely/system-hardened:test 0.0.1

IMAGE_DIR="$1"
IMAGE_REF="$2"
TAG_REF="$3"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/vuln_logs"
# IMAGE_TAR="$LOG_DIR/image.tar"

mkdir -p "$LOG_DIR"

# echo "Building image with buildctl..."
# buildctl build \
#   --frontend dockerfile.v0 \
#   --local context="$IMAGE_DIR" \
#   --local dockerfile="$IMAGE_DIR" \
#   --opt build-arg:tag_ref_name="$TAG_REF" \
#   --output type=docker,name="$IMAGE_REF",dest="$IMAGE_TAR"
docker build --platform linux/amd64 -t "$IMAGE_REF" "$IMAGE_DIR" --build-arg tag_ref_name="$TAG_REF"

# docker load -i "$IMAGE_TAR"

# echo "Running pip-audit scan for vulnerabilities..."
# if ! docker run --rm -v "$LOG_DIR:/work" "$IMAGE_REF" \
#   sh -lc "/opt/venv/bin/python -m pip_audit --format json > /work/audit_report.json"; then
#   echo "pip-audit found vulnerabilities" >&2
#   echo "Check $LOG_DIR/audit_report.json for details." >&2
#   exit 1
# fi

# echo "pip-audit clean: no vulnerabilities found."

echo "Running Trivy scan for HIGH and CRITICAL vulnerabilities..."
if ! docker run --rm \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$LOG_DIR:/work" \
  aquasec/trivy:latest image --severity HIGH,CRITICAL --exit-code 1 \
  --ignore-unfixed \
  --format json --output /work/trivy_report.json "$IMAGE_REF"; then
  echo "Trivy found HIGH/CRITICAL vulnerabilities" >&2
  echo "Check $LOG_DIR/trivy_report.json for details." >&2
  exit 1
fi

echo "Trivy clean: no HIGH or CRITICAL vulnerabilities found."

echo "Running Dockle scan..."
if ! docker run --rm \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$LOG_DIR:/work" \
  goodwithtech/dockle:latest --exit-code 1 --exit-level FATAL \
  -af settings.py \
  --format json --output /work/dockle_report.json "$IMAGE_REF"; then
  echo "Dockle found FATAL findings" >&2
  echo "Check $LOG_DIR/dockle_report.json for details." >&2
  exit 1
fi

echo "Dockle clean: no FATAL findings."

echo "Generating SBOM (SPDX JSON) with Syft..."
docker run --rm \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$LOG_DIR:/work" \
  anchore/syft:latest \
  "$IMAGE_REF" \
  -o spdx-json=/work/sbom_spdx.json

docker push "$IMAGE_REF"
