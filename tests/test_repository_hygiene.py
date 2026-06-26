from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_migration_report_contains_no_api_key_shaped_values():
    report = (PROJECT_ROOT / "note" / "migration_audit_report.md").read_text(encoding="utf-8")
    key_prefix = "s" + "k-"
    assert re.search(re.escape(key_prefix) + r"[A-Za-z0-9_-]{16,}", report) is None


def test_active_files_contain_no_private_server_root():
    private_root = "/" + "rhome/"
    candidates = [PROJECT_ROOT / "README.md"]
    candidates.extend((PROJECT_ROOT / "src").rglob("*.py"))
    candidates.extend(
        path
        for path in (PROJECT_ROOT / "scripts").rglob("*")
        if path.is_file() and path.suffix in {".py", ".ps1", ".sh"}
    )
    candidates.extend(
        path
        for path in (PROJECT_ROOT / "tests").rglob("*.py")
        if path.name != Path(__file__).name
    )

    offenders = []
    for path in candidates:
        if private_root in path.read_text(encoding="utf-8"):
            offenders.append(path.relative_to(PROJECT_ROOT).as_posix())

    assert offenders == []


def test_shell_scripts_keep_lf_line_endings():
    attributes = (PROJECT_ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "*.sh text eol=lf" in attributes.splitlines()
