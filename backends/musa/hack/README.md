# Hack rule replacement guide

This directory stores the local MUSA adaptations that are applied on top of the
PaddlePaddle source tree. The preferred workflow is to keep Paddle source changes
as small, reviewable rules instead of copying full Paddle files into this
repository.

## Why use hack rules instead of `git patch` or full-file copies?

PaddlePaddle is upgraded frequently. Many local MUSA changes are small and stable,
but the surrounding upstream files may change because of unrelated commits. The
rule-based workflow is designed to make those upgrades cheaper and safer.

Compared with applying normal `git patch` files:

- Rules are less sensitive to unrelated upstream commit changes. A normal patch
  often fails when line numbers or nearby context change, even if the actual code
  to replace is unchanged and unrelated to the patch.
- Rules describe the intended replacement directly. For common CUDA-to-MUSA
  substitutions, we do not need to preserve a large patch hunk just to change one
  token.
- Rules can be replayed across Paddle upgrades and fail only when the actual
  matched text is missing, stale, or ambiguous.
- JSON replacement rules can be reused by many files, instead of maintaining many
  nearly identical patch hunks.
- `.hackdiff` rules can be made as small and targeted as needed for structural
  changes, while still avoiding broad global replacements.

Compared with historical full-file Paddle overrides:

- Rules avoid carrying and replaying a full forked copy of upstream Paddle files.
- Paddle upgrades require less manual work because only the small MUSA-specific
  differences need to be checked and updated.
- Duplicate replacements are reduced. One semantic JSON rule can cover many
  files, and one targeted `.hackdiff` block can express only the changed logic.
- Code review is easier because reviewers see the intended MUSA delta instead of
  re-reading large copied files.
- Merge risk is lower: upstream bug fixes and refactors remain in the Paddle tree
  unless a rule intentionally changes the relevant lines.
- Exact replay is possible: rules can be applied to a clean Paddle tree and the
  generated files can be compared byte-for-byte with the expected Paddle diff.
- Stale adaptations are easier to find. If upstream removes or changes the code a
  rule depends on, `tools/apply_hack_rules.py` fails loudly instead of silently
  keeping an old full-file override.

## Directory overview

- `hack_file_rules.json`
  - The main mapping file used by `tools/apply_hack_rules.py`.
  - It tells the tool which Paddle source file should receive which rule files.
  - Each item has:
    - `path`: target file path relative to the Paddle source root.
    - `rules`: ordered rule files under this `hack/` directory.

- `rules/replacements/*.json`
  - Simple text replacement rules.
  - Best for repeated, mechanical substitutions, such as CUDA/MUSA symbol names.
  - Example use cases:
    - `PADDLE_WITH_CUDA` -> `PADDLE_WITH_MUSA`
    - `cublas_handle` -> `mublas_handle`
    - `curand` -> `murand`

- `rules/custom_patches/*.hackdiff`
  - Structured line-block replacement rules.
  - Best for context-sensitive or multi-line changes that are unsafe to express
    as broad string replacements.
  - Despite the name, these files are not full `git apply` patches. They are
    rule files parsed by `tools/apply_hack_rules.py`.

- `cuda_hack/`
  - Compatibility headers that make CUDA-style includes or APIs route to MUSA
    equivalents where needed.

- `paddle_backend_dyload_hack/` and `paddle_platform_dyload_hack/`
  - Dynamic loading shims for MUSA libraries and related Paddle backend/platform
    integration.

- `thrust_hack/`
  - Compatibility code for thrust-related usage.

## How to apply rules

From the repository root:

```bash
python tools/apply_hack_rules.py \
  --repo-root /home/paddle_musa/backends/musa \
  --source-root /home/paddle_musa/Paddle
```

Useful options:

```bash
# Show what would be replaced without modifying Paddle files
python tools/apply_hack_rules.py --dry-run

# Use a custom mapping file
python tools/apply_hack_rules.py \
  --mapping hack/hack_file_rules.json \
  --source-root /home/paddle_musa/Paddle
```

`--source-root` points to the PaddlePaddle source tree. The script modifies files
under that tree unless `--dry-run` is used.

## Execution order

For every target file listed in `hack_file_rules.json`, the tool executes rules in
two passes:

1. All `.hackdiff` rules.
2. All non-`.hackdiff` rules, currently JSON replacement rules.

This means that if one file has both `.hackdiff` and `.json` rules, both types are
applied. A `.hackdiff` rule should normally match upstream Paddle text, while JSON
rules can do later broad symbol substitutions.

## JSON replacement rule format

A JSON rule file is a list of replacement objects:

```json
[
  {
    "from": "PADDLE_WITH_CUDA",
    "to": "PADDLE_WITH_MUSA"
  }
]
```

Behavior:

- Replacements are plain substring replacements, not regular expressions.
- Each rule is applied line by line.
- Lines containing the text `include` are skipped by JSON rules. Use a
  `.hackdiff` rule for include changes.
- All occurrences on a matching line are replaced.
- Rules should be semantic and reusable when possible.

Good JSON rule candidates:

- Stable API or macro renames.
- Repeated token substitutions.
- Changes that are safe everywhere the token appears outside include lines.

Avoid JSON rules for:

- Include block changes.
- Context-sensitive logic changes.
- Broad strings that may appear in comments or unrelated code.

## `.hackdiff` rule format

A `.hackdiff` rule is split into blocks by blank lines. Each block contains:

- one or more `-` lines: the old Paddle text to match.
- zero or more `+` lines: the replacement text.

Example:

```diff
-#ifdef PADDLE_WITH_CUDA
-#include <cuda.h>
-#endif
+#ifdef PADDLE_WITH_MUSA
+#include <musa.h>
+#endif
```

Delete-only example:

```diff
-  // obsolete CUDA-only workaround
-  DoOldThing();
```

Important syntax rules:

- Every non-empty line in a `.hackdiff` rule must start with `-` or `+`.
- All `-` lines must come before all `+` lines in the same block.
- A block must contain at least one `-` line.
- Blank lines separate blocks.
- To represent an empty line inside old or new text, use a line containing only
  `-` or `+`.

## `.hackdiff` matching behavior

For each block, the tool tries to match the old `-` lines exactly. It also tries a
relaxed form that removes one separator space after the `-`/`+` prefix when
needed, which makes common diff-style indentation easier to write.

Current behavior:

- If multiple locations match a non-delete block, only the first complete match is
  replaced. Repeated mechanical substitutions should be handled by JSON rules
  instead of `.hackdiff` blocks.
- If a non-delete block is already applied, it is skipped.
- If a non-delete block is neither matched nor already applied, the script fails.
- Delete-only blocks delete all complete matches.
- If a delete-only block has no match at all, it is skipped.
- If a delete-only block partially matches, the script fails.

Because non-delete `.hackdiff` blocks replace only one match, use them for targeted
structural edits. Use JSON replacement rules for repeated token or symbol
substitutions across a file.

## Difference from normal `git diff` / patch files

The `.hackdiff` format is designed specifically for the current PaddlePaddle-to-MUSA adaptation workflow. It is not a normal unified diff generated by `git diff`, and it cannot be applied by `git apply`.

Normal patch files describe a change in git's unified-diff format. They usually contain file headers, hunk headers, and unchanged context lines. A `.hackdiff` rule file only describes the old Paddle text to match and the replacement text to write.

Do not include these normal patch elements in `.hackdiff` rule files:

- `diff --git ...`
- `index ...`
- `--- a/file`
- `+++ b/file`
- `@@ ... @@`
- unchanged context lines that do not start with `-` or `+`

A `.hackdiff` rule file only contains replacement blocks:

- `-` lines are the exact old Paddle text to match.
- `+` lines are the replacement text to write.
- Blank lines separate independent replacement blocks.

This design is useful for the current PaddlePaddle MUSA adaptation scenario because many local changes are small MUSA-specific deltas inside large upstream Paddle files. Compared with normal patch files, `.hackdiff` rules make it easier to:

- keep only the MUSA-specific replacement instead of carrying unrelated upstream context;
- survive unrelated upstream line-number or nearby-context changes during Paddle upgrades;
- review the intended MUSA adaptation directly;
- replay the adaptation on a clean Paddle tree and compare the generated result exactly;
- combine targeted structural edits with reusable JSON CUDA-to-MUSA substitutions.

## Mapping file example

```json
{
  "path": "paddle/phi/kernels/primitive/compute_primitives.h",
  "rules": [
    "rules/custom_patches/paddle_phi_kernels_primitive_compute_primitives_h_legacy_patch.hackdiff",
    "rules/replacements/paddle_with_cuda_to_musa.json"
  ]
}
```

This means:

1. Apply the custom `.hackdiff` patch to
   `paddle/phi/kernels/primitive/compute_primitives.h`.
2. Apply the JSON replacement rule to the same file.

The paths in `rules` are relative to `hack/`.

## How to add or update a rule

1. Start from the current Paddle diff for the target file.
2. Prefer existing JSON rules if they can safely reproduce the change.
3. Add a new JSON rule only for safe, reusable token replacements.
4. Use a minimal `.hackdiff` rule for structural or context-sensitive changes.
5. Register the target file and rule path in `hack_file_rules.json`.
6. Apply the rules to a clean Paddle file or a temporary replay tree.
7. Compare the generated file against the intended Paddle working tree diff.

## Practical migration principles

When adapting new Paddle code for MUSA, first make and verify the required change directly in the Paddle source tree. Then use `git diff` in the Paddle tree to inspect the exact Paddle-side delta. Convert that diff into hack rules by following the principles below.

A helper tool is available for the common one-file workflow:

```bash
python tools/update_hack_rule_from_diff.py paddle/path/to/file.cc
```

The tool reads `git diff -- paddle/path/to/file.cc` from the Paddle source tree and then:

- updates the existing `.hackdiff` rule if the file already has one in `hack_file_rules.json`;
- creates a new `.hackdiff` rule and mapping entry if the file is not mapped yet;
- reuses existing JSON replacement rules when explicitly requested and already present;
- creates new JSON replacement rules when explicitly requested and no equivalent rule exists.

For reusable JSON substitutions, pass explicit replacement pairs:

```bash
python tools/update_hack_rule_from_diff.py paddle/path/to/file.cc \
  --json-rule PADDLE_WITH_CUDA PADDLE_WITH_MUSA \
  --json-rule cublas_handle mublas_handle
```

To apply rules for one mapped Paddle file only, use `apply_hack_rules.py --path`:

```bash
python tools/apply_hack_rules.py \
  --source-root /home/paddle_musa/Paddle \
  --path paddle/path/to/file.cc
```

Use `--dry-run` with either tool when you want to inspect the planned changes first.

Note: `tools/update_hack_rule_from_diff.py` is a helper, not a correctness guarantee. It cannot always infer the most reviewable or safest rule structure from a raw diff. If the generated rule is too broad, too narrow, hard to review, or does not replay to the expected Paddle-side result, update the rules manually according to the principles below.

### 1. Identify the real MUSA-specific delta first

Do not convert historical full-file overrides mechanically. Old full-file overrides usually contain three kinds of content:

- Original Paddle upstream code.
- Stale code from an older Paddle version.
- The actual MUSA-specific adaptation that still needs to be preserved.

Use the current upstream Paddle file as the baseline, and keep only the third category. Do not preserve unrelated old code just to make a replayed file look like a historical override.

### 2. Choose the rule type by priority

Handle each difference in this order:

1. **Reuse an existing JSON rule first.**
   - Use this for repeated semantic substitutions.
   - Examples: `PADDLE_WITH_CUDA` -> `PADDLE_WITH_MUSA`, `cublas_handle` -> `mublas_handle`.
2. **Add a new JSON rule only when it is safe and reusable.**
   - A new JSON rule should serve multiple files or a stable repeated pattern.
   - Do not add broad JSON rules for one-off file-specific changes.
3. **Use a `.hackdiff` rule for structural, context-sensitive, or include-related changes.**
   - Include block changes, conditional compilation blocks, function logic changes, and synchronization workarounds should use `.hackdiff`.

### 3. Keep JSON rules semantic and explicit

JSON rules are line-by-line substring replacements. Even though lines containing `include` are skipped, broad replacements can still affect comments, string literals, or unrelated identifiers.

Good JSON candidates:

```text
PADDLE_WITH_CUDA -> PADDLE_WITH_MUSA
cublas_handle -> mublas_handle
curand -> murand
```

Poor JSON candidates:

```text
large one-off code blocks
whitespace-only changes
comment-only changes
single-file workarounds
very short or ambiguous tokens
```

If you are not sure whether a JSON rule is safe, prefer a smaller `.hackdiff` rule with enough local context.

### 4. Keep `.hackdiff` rules small and targeted

A `.hackdiff` rule is not a normal `git apply` patch. It is an old/new text replacement block parsed by `tools/apply_hack_rules.py`.

When writing `.hackdiff` rules:

- Each block should express one clear intent.
- Use an appropriate amount of surrounding context: prefer the shortest old-code block that still uniquely matches the intended location and can be replaced with the desired MUSA adaptation block.
- Do not copy large unrelated context blocks.
- Do not encode repeated mechanical substitutions as many `.hackdiff` blocks; prefer JSON when the replacement is semantic and safe.
- Include changes must use `.hackdiff`, not JSON.

### 5. Respect the rule execution model

For each target file, the tool currently runs rules in two phases:

1. Apply all `.hackdiff` rules.
2. Apply all JSON replacement rules.

Therefore:

- `.hackdiff` rules should normally match current upstream Paddle text.
- JSON rules should normally act as later broad CUDA-to-MUSA substitutions.
- Avoid writing `.hackdiff` rules that only match after JSON replacements unless you have explicitly verified the ordering and replay result.

### 6. Do not use replay compatibility rules as a dumping ground

A small number of replay compatibility rules may be useful, but `*_replay_compat.json` rules should not be overused.

Avoid:

- Adding replay JSON rules for whitespace-only differences.
- Adding replay JSON rules for comment-only differences.
- Using JSON to carry large one-off file-specific changes.
- Preserving old code that has no real MUSA-specific meaning only to make a diff disappear.

If a difference is structural, put it in the corresponding `.hackdiff` rule. If a difference has no value, remove it instead of creating a rule for it.

### 7. Every rule change must be replay-verifiable

A rule change is not complete when the file is written. The correctness standard is:

1. Apply the rules to a clean Paddle tree.
2. Confirm the generated target file matches the expected Paddle working tree byte-for-byte.
3. If a rule fails, fix the rule or confirm that Paddle upstream changed. Do not silently skip failures.

See `Recommended verification` below for the suggested validation flow.

## Recommended verification

For quick validation:

```bash
python tools/apply_hack_rules.py --dry-run
```

For exact replay validation, use a clean copy of the target Paddle files, apply
rules there, then compare against the current Paddle working tree:

```bash
python tools/apply_hack_rules.py \
  --repo-root /home/paddle_musa/backends/musa \
  --source-root /tmp/paddle_rule_replay \
  --mapping hack/hack_file_rules.json
```

Then compare:

```bash
diff -u /home/paddle_musa/Paddle/<target-file> \
        /tmp/paddle_rule_replay/<target-file>
```

A rule change is considered correct only when the generated Paddle file exactly
matches the intended Paddle-side change.

## Common pitfalls

- Do not use JSON rules for include lines; the tool intentionally skips lines
  containing `include`.
- Do not make `.hackdiff` blocks too short. Short blocks may match the wrong
  location or become ambiguous when Paddle upstream changes.
- Do not put normal context lines in `.hackdiff` rules. Every non-empty line must
  start with `-` or `+`.
- Do not silently ignore rule failures. A failed match usually means Paddle
  upstream changed or the rule is too broad/stale.
- Do not carry stale full-file hacks forward when a minimal rule can reproduce the
  required MUSA change.
