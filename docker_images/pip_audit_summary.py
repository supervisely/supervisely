import json
import sys


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: pip_audit_summary.py <audit_report.json>", file=sys.stderr)
        return 2

    path = sys.argv[1]
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    vulns = data.get("vulnerabilities", [])
    print(str(len(vulns)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
