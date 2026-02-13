set -euo pipefail

IMAGE_REF="supervisely/import-export-hardened:0.0.1"

docker build --platform linux/amd64 -t "$IMAGE_REF" .
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

docker run --rm -v "$LOG_DIR:/work" "$IMAGE_REF" \
	sh -lc "python -m pip install -q pip-audit && pip-audit --format json > /work/audit_report.json"
trivy image --severity HIGH,CRITICAL --exit-code 1 \
	--format json --output "$LOG_DIR/trivy_report.json" "$IMAGE_REF"
docker push "$IMAGE_REF"
