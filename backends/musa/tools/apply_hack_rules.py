#!/usr/bin/env python3

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HackDiffRule:
    old_lines: list[str]
    new_lines: list[str]
    relaxed_old_lines: list[str]
    relaxed_new_lines: list[str]


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_lines(path):
    return path.read_text(encoding="utf-8").splitlines(keepends=True)


def split_hackdiff_rule_blocks(lines):
    blocks = []
    current = []

    for line in lines:
        if line.strip() == "":
            if current:
                blocks.append(current)
                current = []
            continue
        current.append(line)

    if current:
        blocks.append(current)

    return blocks


def strip_hackdiff_prefix(line):
    return line[1:]


def strip_one_space_after_prefix(line):
    content = line[1:]
    if content.startswith(" "):
        return content[1:]
    return content


def strip_single_separator_space(line):
    content = line[1:]
    if content.startswith(" ") and not content.startswith("  "):
        return content[1:]
    return content


def parse_hackdiff_rule_block(block, block_index):
    old_lines = []
    new_lines = []
    relaxed_old_lines = []
    relaxed_new_lines = []
    seen_new_line = False

    if all(line.rstrip("\n") == "-" for line in block):
        return None

    for line_index, line in enumerate(block, start=1):
        if line.startswith("-"):
            if seen_new_line:
                raise ValueError(
                    f"Invalid rule block {block_index}: '-' line appears after '+' line "
                    f"at line {line_index}."
                )
            old_lines.append(strip_hackdiff_prefix(line))
            relaxed_old_lines.append(strip_one_space_after_prefix(line))
        elif line.startswith("+"):
            if not old_lines:
                raise ValueError(
                    f"Invalid rule block {block_index}: '+' line must follow at least one '-' line."
                )
            seen_new_line = True
            new_lines.append(strip_single_separator_space(line))
            relaxed_new_lines.append(strip_one_space_after_prefix(line))
        else:
            raise ValueError(
                f"Invalid rule block {block_index}: line must start with '-' or '+': {line.rstrip()}"
            )

    if not old_lines:
        raise ValueError(f"Invalid rule block {block_index}: rule must contain '-' lines.")

    return HackDiffRule(
        old_lines=old_lines,
        new_lines=new_lines,
        relaxed_old_lines=relaxed_old_lines,
        relaxed_new_lines=relaxed_new_lines,
    )


def load_hackdiff_rules(path):
    blocks = split_hackdiff_rule_blocks(read_lines(path))
    if not blocks:
        raise ValueError(f"No replacement rules found: {path}")
    rules = []
    for index, block in enumerate(blocks, start=1):
        rule = parse_hackdiff_rule_block(block, index)
        if rule is not None:
            rules.append(rule)
    if not rules:
        raise ValueError(f"No replacement rules found: {path}")
    return rules


def find_subsequence(lines, pattern):
    if not pattern:
        raise ValueError("Replacement pattern must not be empty.")

    last_start = len(lines) - len(pattern)
    for index in range(last_start + 1):
        if lines[index : index + len(pattern)] == pattern:
            return index
    return -1


def has_partial_subsequence(lines, pattern):
    if not pattern:
        return False

    for index in range(len(lines)):
        if lines[index] != pattern[0]:
            continue
        matched = 0
        while (
            matched < len(pattern)
            and index + matched < len(lines)
            and lines[index + matched] == pattern[matched]
        ):
            matched += 1
        if 0 < matched < len(pattern):
            return True
    return False


def replace_all_subsequences(lines, old_lines, new_lines):
    if not old_lines:
        raise ValueError("Replacement pattern must not be empty.")

    updated_lines = []
    replacements = 0
    index = 0
    while index < len(lines):
        if lines[index : index + len(old_lines)] == old_lines:
            updated_lines.extend(new_lines)
            index += len(old_lines)
            replacements += 1
        else:
            updated_lines.append(lines[index])
            index += 1
    return updated_lines, replacements


def replace_first_subsequence(lines, old_lines, new_lines):
    match_index = find_subsequence(lines, old_lines)
    if match_index == -1:
        return list(lines), 0
    return (
        lines[:match_index] + new_lines + lines[match_index + len(old_lines) :],
        1,
    )


def hackdiff_candidate_replacements(rule):
    candidates = [(rule.old_lines, rule.new_lines)]
    if rule.relaxed_old_lines != rule.old_lines or rule.relaxed_new_lines != rule.new_lines:
        candidates.append((rule.relaxed_old_lines, rule.relaxed_new_lines))
    return candidates


def hackdiff_rule_already_applied(lines, rule):
    for old_lines, new_lines in hackdiff_candidate_replacements(rule):
        if not new_lines:
            continue
        if find_subsequence(lines, old_lines) != -1:
            return False
        if find_subsequence(lines, new_lines) != -1:
            return True
    return False


def apply_hackdiff_rules(lines, rules):
    updated_lines = list(lines)
    replacements = 0

    for rule_index, rule in enumerate(rules, start=1):
        if hackdiff_rule_already_applied(updated_lines, rule):
            continue

        matched = False
        delete_only_partial_match = False
        for candidate_old_lines, candidate_new_lines in hackdiff_candidate_replacements(rule):
            if candidate_new_lines:
                candidate_updated_lines, candidate_replacements = replace_first_subsequence(
                    updated_lines, candidate_old_lines, candidate_new_lines
                )
            else:
                candidate_updated_lines, candidate_replacements = replace_all_subsequences(
                    updated_lines, candidate_old_lines, candidate_new_lines
                )
            if candidate_replacements:
                updated_lines = candidate_updated_lines
                replacements += candidate_replacements
                if not candidate_new_lines and has_partial_subsequence(
                    updated_lines, candidate_old_lines
                ):
                    raise ValueError(
                        f"Delete-only rule {rule_index} partially matched the target file."
                    )
                matched = True
                break
            if not candidate_new_lines and has_partial_subsequence(
                updated_lines, candidate_old_lines
            ):
                delete_only_partial_match = True

        if matched:
            continue
        if not rule.new_lines:
            if delete_only_partial_match:
                raise ValueError(
                    f"Delete-only rule {rule_index} partially matched the target file."
                )
            continue
        raise ValueError(f"Rule {rule_index} did not match the target file.")

    return updated_lines, replacements


def apply_hackdiff_rule_file(path, target_path, dry_run=False):
    original_lines = read_lines(target_path)
    rules = load_hackdiff_rules(path)

    updated_lines, replacements = apply_hackdiff_rules(original_lines, rules)

    if updated_lines != original_lines and not dry_run:
        target_path.write_text("".join(updated_lines), encoding="utf-8")

    return replacements


def load_json_rules(path):
    rule_data = load_json(path)
    if not isinstance(rule_data, list):
        raise ValueError(f"Rule file must be a list: {path}")

    rules = []
    for rule in rule_data:
        if not isinstance(rule, dict) or "from" not in rule or "to" not in rule:
            raise ValueError(f"Invalid rule in {path}: {rule}")
        max_replacements = rule.get("max_replacements")
        if max_replacements is not None and (
            not isinstance(max_replacements, int) or max_replacements < 1
        ):
            raise ValueError(f"Invalid max_replacements in {path}: {rule}")
        rules.append(
            {
                "from": rule["from"],
                "to": rule["to"],
                "skip_include": rule.get("skip_include", True),
                "multiline": rule.get("multiline", False),
                "max_replacements": max_replacements,
            }
        )
    return rules


def apply_json_rule_file(path, target_path, dry_run=False):
    original_lines = read_lines(target_path)
    updated_lines = list(original_lines)
    replacements = 0

    for rule in load_json_rules(path):
        old = rule["from"]
        new = rule["to"]
        max_replacements = rule["max_replacements"]
        if rule["multiline"]:
            updated_text = "".join(updated_lines)
            rule_count = updated_text.count(old)
            if max_replacements is not None:
                rule_count = min(rule_count, max_replacements)
            if rule_count:
                updated_text = updated_text.replace(old, new, rule_count)
                updated_lines = updated_text.splitlines(keepends=True)
                replacements += rule_count
            continue

        remaining = max_replacements
        for index, line in enumerate(updated_lines):
            if rule["skip_include"] and "include" in line:
                continue
            rule_count = line.count(old)
            if remaining is not None:
                rule_count = min(rule_count, remaining)
            if rule_count:
                updated_lines[index] = line.replace(old, new, rule_count)
                replacements += rule_count
                if remaining is not None:
                    remaining -= rule_count
                    if remaining == 0:
                        break

    if updated_lines != original_lines and not dry_run:
        target_path.write_text("".join(updated_lines), encoding="utf-8")

    return replacements


def apply_rule_file(repo_root, rule_path, target_path, dry_run=False):
    path = repo_root / "hack" / rule_path
    if not path.is_file():
        raise FileNotFoundError(f"Rule file does not exist: {path}")

    if path.suffix == ".json":
        return apply_json_rule_file(path, target_path, dry_run=dry_run)
    if path.suffix == ".hackdiff":
        return apply_hackdiff_rule_file(path, target_path, dry_run=dry_run)

    raise ValueError(f"Unsupported rule file suffix: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Replace strings according to hack/hack_file_rules.json and hack/rules/*.json|*.hackdiff."
    )
    parser.add_argument(
        "--repo-root",
        default=Path(__file__).resolve().parents[1],
        type=Path,
        help="Repository root. Defaults to the parent directory of tools/.",
    )
    parser.add_argument(
        "--mapping",
        default="hack/hack_file_rules.json",
        help="File-rule mapping JSON path relative to repo root.",
    )
    parser.add_argument(
        "--source-root",
        default="/home/paddle_musa/Paddle",
        type=Path,
        help="Source tree root for target files. Defaults to /home/paddle_musa/Paddle.",
    )
    parser.add_argument(
        "--path-prefix",
        default=None,
        help="Only apply mapping entries whose target path starts with this prefix.",
    )
    parser.add_argument(
        "--path",
        default=None,
        help="Only apply the mapping entry whose target path exactly matches this Paddle-relative file path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned replacements without modifying files.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    source_root = args.source_root.resolve()
    if args.path and args.path_prefix:
        raise ValueError("--path and --path-prefix cannot be used together.")
    mapping_path = repo_root / args.mapping
    mapping = load_json(mapping_path)

    files = mapping.get("files")
    if not isinstance(files, list):
        raise ValueError(f"Mapping file must contain a 'files' list: {mapping_path}")

    changed_files = set()
    total_replacements = 0

    work_items = []
    for item in files:
        if not isinstance(item, dict) or "path" not in item or "rules" not in item:
            raise ValueError(f"Invalid mapping item: {item}")

        item_path = item["path"]
        if args.path and item_path != args.path:
            continue
        if args.path_prefix and not item_path.startswith(args.path_prefix):
            continue

        target_path = source_root / item_path
        if not target_path.is_file():
            raise FileNotFoundError(f"Target file does not exist: {target_path}")

        work_items.append((item_path, target_path, item["rules"]))

    for suffix_match in (".hackdiff", None):
        for item_path, target_path, rule_paths in work_items:
            if suffix_match == ".hackdiff":
                current_rule_paths = [
                    rule_path for rule_path in rule_paths if rule_path.endswith(".hackdiff")
                ]
            else:
                current_rule_paths = [
                    rule_path for rule_path in rule_paths if not rule_path.endswith(".hackdiff")
                ]

            file_replacements = 0
            for rule_path in current_rule_paths:
                file_replacements += apply_rule_file(
                    repo_root, rule_path, target_path, dry_run=args.dry_run
                )

            if file_replacements:
                changed_files.add(item_path)
                total_replacements += file_replacements
                action = "would replace" if args.dry_run else "replaced"
                print(f"{action} {file_replacements}: {item_path}")

    print(f"files changed: {len(changed_files)}")
    print(f"total replacements: {total_replacements}")


if __name__ == "__main__":
    main()
