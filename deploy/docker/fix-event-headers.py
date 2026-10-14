"""Image-only libevent/Folly macro namespacing; leave libev and Clipper intact.

Run after installing the pinned dependencies, before compiling Clipper. Renaming
header macros (not their integer values or exported functions) prevents libevent
EV_TIMEOUT/EV_READ/EV_WRITE/EV_SIGNAL from replacing libev enum declarations.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


FLAGS = {"EV_TIMEOUT": 0x01, "EV_READ": 0x02, "EV_WRITE": 0x04, "EV_SIGNAL": 0x08}
LIBEV_FLAGS = {"EV_READ": 0x01, "EV_WRITE": 0x02, "EV_TIMER": 0x100, "EV_SIGNAL": 0x400}
PREFIX = "CLIPPER_LIBEVENT_"
TOKENS = re.compile(r"\b(?:" + "|".join(sorted(FLAGS)) + r")\b")


def without_comments(text):
    return re.sub(r"/\*.*?\*/|//[^\n]*", "", text, flags=re.DOTALL)


def enum_names(text):
    names = set()
    for body in re.findall(r"\benum(?:\s+\w+)?\s*\{([^{}]*)\}", without_comments(text)):
        names.update(re.findall(r"\b(EV_[A-Z0-9_]+)\s*=", body))
    return names


def macro_names(text):
    return set(re.findall(r"^\s*#\s*define\s+(\w+)\b", without_comments(text), re.MULTILINE))


def define_value(text, name):
    match = re.search(r"^\s*#\s*define\s+" + re.escape(name) +
                      r"\s+(0[xX][0-9a-fA-F]+|[0-9]+)\b", text, re.MULTILINE)
    if not match:
        raise ValueError("Missing simple event constant: " + name)
    return int(match.group(1), 0)


def patch_headers(root=Path("/")):
    root = Path(root).resolve()
    event_root = root / "usr/include/event2"
    folly_root = root / "usr/local/include/folly"
    libev_header = root / "usr/local/include/ev.h"
    event_header = event_root / "event.h"
    for directory in (event_root, folly_root):
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError("Expected installed header directory: " + str(directory))
    libev = libev_header.read_bytes()
    libev_text = libev.decode("utf-8")
    for name, value in LIBEV_FLAGS.items():
        match = re.search(r"\b" + name + r"\s*=\s*(0[xX][0-9a-fA-F]+|[0-9]+)\b", libev_text)
        if not match or int(match.group(1), 0) != value:
            raise ValueError("Unexpected libev enum definition: " + name)
    if not re.search(r"\bEV_TIMEOUT\s*=\s*EV_TIMER\b", libev_text):
        raise ValueError("Unexpected libev enum definition: EV_TIMEOUT alias")
    libev_names = enum_names(libev_text)
    if not set(FLAGS).issubset(libev_names):
        raise ValueError("Unexpected libev enum layout")
    event_text = event_header.read_text(encoding="utf-8")
    defined = macro_names(event_text)
    namespaced = all(PREFIX + name in defined for name in FLAGS)
    for name, expected in FLAGS.items():
        # Also support upgrading a partially namespaced v1 header safely.
        if (name in defined) == (PREFIX + name in defined):
            raise ValueError("Missing or ambiguous libevent constant: " + name)
        actual = define_value(event_text, PREFIX + name if PREFIX + name in defined else name)
        if actual != expected:
            raise ValueError("Unexpected libevent value for %s: %s" % (name, actual))
    # Resolve every target before changing anything. Do not follow header symlinks
    # into unrelated installed files or operate on host/Clipper source paths.
    candidates = []
    for directory in (event_root, folly_root):
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix in (".h", ".hpp", ".inl", ".tcc"):
                if path.is_symlink():
                    continue  # Multiarch configuration headers may be symlinks.
                path.resolve().relative_to(directory.resolve())
                candidates.append(path)
    legacy_header = root / "usr/include/event.h"
    if legacy_header.exists():
        if not legacy_header.is_symlink():
            candidates.append(legacy_header)
    updates = []
    for path in candidates:
        before = path.read_bytes()
        text = before.decode("utf-8")
        collisions = macro_names(text) & libev_names
        unexpected = collisions - set(FLAGS)
        if unexpected:
            raise ValueError("Uncovered libevent/libev macro collision in %s: %s" %
                             (path, ", ".join(sorted(unexpected))))
        rewritten = TOKENS.sub(lambda match: PREFIX + match.group(0), text)
        remaining = macro_names(rewritten) & libev_names
        if remaining:
            raise ValueError("Event macro collision remains: " + ", ".join(sorted(remaining)))
        after = rewritten.encode("utf-8")
        if before != after:
            updates.append((path, before, after))
    if not namespaced and not any(folly_root in path.parents for path, _, _ in updates):
        raise ValueError("No Folly event flag references found; incompatible dependency layout")
    records = []
    for path, before, after in updates:
        path.write_bytes(after)
        records.append({"path": "/" + path.relative_to(root).as_posix(),
                        "before_sha256": hashlib.sha256(before).hexdigest(),
                        "after_sha256": hashlib.sha256(after).hexdigest()})
    for name, expected in FLAGS.items():
        if define_value(event_header.read_text(encoding="utf-8"), PREFIX + name) != expected:
            raise ValueError("Event constant changed after namespacing: " + name)
    if libev_header.read_bytes() != libev:
        raise ValueError("libev header must remain unchanged")
    report = {"compatibility": "libevent-prefix-v2", "libevent_values": FLAGS,
              "libev_values": dict(LIBEV_FLAGS, EV_TIMEOUT=LIBEV_FLAGS["EV_TIMER"]),
              "checked_libev_enum_names": sorted(libev_names), "changed_headers": records}
    report_path = root / "opt/clipper-base-sources/event-header-compat.json"
    if updates or not report_path.exists():
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/"))
    args = parser.parse_args()
    report = patch_headers(args.root)
    print("Event header compatibility: %s, changed_headers=%d" %
          (report["compatibility"], len(report["changed_headers"])))


if __name__ == "__main__":
    main()
