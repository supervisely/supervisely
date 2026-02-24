import json
import sys

def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: trivy_summary.py <trivy_report.json>", file=sys.stderr)
        return 2

    path = sys.argv[1]
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    high = 0
    critical = 0
    for result in data.get("Results", []):
        for vuln in result.get("Vulnerabilities", []) or []:
            severity = (vuln.get("Severity") or "").upper()
            if severity == "HIGH":
                high += 1
            elif severity == "CRITICAL":
                critical += 1

    print(f"{high} {critical}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
