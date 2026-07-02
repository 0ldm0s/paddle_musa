#!/usr/bin/env python3

import argparse
import difflib
import json
import re
import subprocess
from pathlib import Path


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def normalize_paddle_path(path):
    path = path.strip()
    if path.startswith("./"):
        path = path[2:]
    return path


def rule_name_for_path(paddle_path):
    name = re.sub(r"[^A-Za-z0-9]+", "_", paddle_path).strip("_").lower()
    return f"{name}.hackdiff"


def replacement_rule_name(old, new):
    def slug(value):
        value = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
        return value[:80] or "empty"

    return f"{slug(old)}_to_{slug(new)}.json"


def diff_for_file(source_root, paddle_path):
    result = subprocess.run(
        ["git", "-C", str(source_root), "diff", "--", paddle_path],
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout


def parse_unified_diff(diff_text):
    hunks = []
    old_lines = []
    new_lines = []

    def flush_hunk():
        nonlocal old_lines, new_lines
        if old_lines or new_lines:
            hunks.append((old_lines, new_lines))
            old_lines = []
            new_lines = []

    for line in diff_text.splitlines(keepends=True):
        if line.startswith(("diff --git", "index ", "--- ", "+++ ")):
            continue
        if line.startswith("@@"):
            flush_hunk()
            continue
        if line.startswith("-"):
            old_lines.append(line[1:])
        elif line.startswith("+"):
            new_lines.append(line[1:])
        elif line.startswith(" "):
            old_lines.append(line[1:])
            new_lines.append(line[1:])
        elif line.startswith("\\ No newline"):
            continue

    flush_hunk()
    return hunks


def hackdiff_block(old_lines, new_lines):
    lines = []
    for line in old_lines:
        lines.append("-" + line)
    for line in new_lines:
        lines.append("+" + line)
    text = "".join(lines)
    if text and not text.endswith("\n"):
        text += "\n"
    return text


def load_mapping(mapping_path):
    mapping = load_json(mapping_path)
    files = mapping.get("files")
    if not isinstance(files, list):
        raise ValueError(f"Mapping file must contain a 'files' list: {mapping_path}")
    return mapping


def find_mapping_item(mapping, paddle_path):
    for item in mapping["files"]:
        if item.get("path") == paddle_path:
            return item
    return None


def ensure_mapping_item(mapping, paddle_path):
    item = find_mapping_item(mapping, paddle_path)
    if item is not None:
        return item, False
    item = {"path": paddle_path, "rules": []}
    mapping["files"].append(item)
    mapping["files"].sort(key=lambda entry: entry["path"])
    return item, True


def existing_json_rule(repo_root, old, new):
    replacements_dir = repo_root / "hack" / "rules" / "replacements"
    for path in sorted(replacements_dir.glob("*.json")):
        try:
            rules = load_json(path)
        except Exception:
            continue
        if not isinstance(rules, list):
            continue
        for rule in rules:
            if isinstance(rule, dict) and rule.get("from") == old and rule.get("to") == new:
                return path
    return None


def append_rule_once(item, rule_path):
    if rule_path not in item["rules"]:
        item["rules"].append(rule_path)
        return True
    return False


def update_hackdiff_rule(repo_root, item, paddle_path, hunks):
    custom_dir = repo_root / "hack" / "rules" / "custom_patches"
    custom_dir.mkdir(parents=True, exist_ok=True)

    existing = [rule for rule in item["rules"] if rule.endswith(".hackdiff")]
    if existing:
        rel_rule = existing[0]
        rule_path = repo_root / "hack" / rel_rule
    else:
        rule_path = custom_dir / rule_name_for_path(paddle_path)
        rel_rule = str(rule_path.relative_to(repo_root / "hack"))
        append_rule_once(item, rel_rule)

    blocks = []
    for old_lines, new_lines in hunks:
        if not old_lines:
            raise ValueError("Cannot create .hackdiff rule without old lines in a git diff hunk.")
        blocks.append(hackdiff_block(old_lines, new_lines).rstrip("\n"))
    rule_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    return rel_rule


def add_or_reuse_json_rule(repo_root, item, old, new):
    existing = existing_json_rule(repo_root, old, new)
    if existing is not None:
        rel_rule = str(existing.relative_to(repo_root / "hack"))
        append_rule_once(item, rel_rule)
        return rel_rule, False

    replacements_dir = repo_root / "hack" / "rules" / "replacements"
    replacements_dir.mkdir(parents=True, exist_ok=True)
    rule_path = replacements_dir / replacement_rule_name(old, new)
    suffix = 2
    while rule_path.exists():
        rule_path = replacements_dir / f"{rule_path.stem}_{suffix}.json"
        suffix += 1
    write_json(rule_path, [{"from": old, "to": new}])
    rel_rule = str(rule_path.relative_to(repo_root / "hack"))
    append_rule_once(item, rel_rule)
    return rel_rule, True


def main():
    parser = argparse.ArgumentParser(
        description="Create or update hack rules from a git diff for one Paddle source file."
    )
    parser.add_argument(
        "path",
        help="Paddle-relative file path, for example paddle/phi/kernels/foo.h.",
    )
    parser.add_argument(
        "--repo-root",
        default=Path(__file__).resolve().parents[1],
        type=Path,
        help="Repository root. Defaults to the parent directory of tools/.",
    )
    parser.add_argument(
        "--source-root",
        default="/home/paddle_musa/Paddle",
        type=Path,
        help="Paddle source tree root. Defaults to /home/paddle_musa/Paddle.",
    )
    parser.add_argument(
        "--mapping",
        default="hack/hack_file_rules.json",
        help="File-rule mapping JSON path relative to repo root.",
    )
    parser.add_argument(
        "--json-rule",
        nargs=2,
        metavar=("FROM", "TO"),
        action="append",
        default=[],
        help="Add or reuse a JSON replacement rule and attach it to the mapping. Can be used multiple times.",
    )
    parser.add_argument(
        "--no-hackdiff",
        action="store_true",
        help="Do not create or update a .hackdiff rule from git diff; only process --json-rule entries.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned changes without writing rule or mapping files.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    source_root = args.source_root.resolve()
    paddle_path = normalize_paddle_path(args.path)
    mapping_path = repo_root / args.mapping

    target_path = source_root / paddle_path
    if not target_path.exists():
        raise FileNotFoundError(f"Paddle target file does not exist: {target_path}")

    mapping = load_mapping(mapping_path)
    item = find_mapping_item(mapping, paddle_path)
    created_mapping = False

    planned = []

    if not args.no_hackdiff:
        diff_text = diff_for_file(source_root, paddle_path)
        if diff_text.strip():
            if item is None:
                item, created_mapping = ensure_mapping_item(mapping, paddle_path)
                planned.append(f"add mapping entry: {paddle_path}")
            hunks = parse_unified_diff(diff_text)
            if not hunks:
                raise ValueError("Cannot create .hackdiff rule without diff hunks.")
            if not args.dry_run:
                rel_rule = update_hackdiff_rule(repo_root, item, paddle_path, hunks)
            else:
                existing = [rule for rule in item["rules"] if rule.endswith(".hackdiff")]
                rel_rule = existing[0] if existing else f"rules/custom_patches/{rule_name_for_path(paddle_path)}"
            planned.append(f"update hackdiff rule: {rel_rule}")
        else:
            planned.append(f"no git diff for: {paddle_path}")

    for old, new in args.json_rule:
        if item is None:
            item, created_mapping = ensure_mapping_item(mapping, paddle_path)
            planned.append(f"add mapping entry: {paddle_path}")
        if args.dry_run:
            existing = existing_json_rule(repo_root, old, new)
            rel_rule = (
                str(existing.relative_to(repo_root / "hack"))
                if existing is not None
                else f"rules/replacements/{replacement_rule_name(old, new)}"
            )
            planned.append(f"add or reuse JSON rule: {rel_rule}")
        else:
            rel_rule, created = add_or_reuse_json_rule(repo_root, item, old, new)
            action = "add JSON rule" if created else "reuse JSON rule"
            planned.append(f"{action}: {rel_rule}")

    if not planned:
        print(f"no rule changes planned for: {paddle_path}")
        return

    if not args.dry_run:
        write_json(mapping_path, mapping)

    for line in planned:
        print(line)


if __name__ == "__main__":
    main()
