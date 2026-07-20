---
name: minivllm-release-maintainer
description: Maintain this repository's chronological, bilingual release history. Use whenever the user asks to analyze or split a commit range or PR into releases, create or revise release Markdown, explain bugs and fixes with exact code links and examples, update the README Recent Releases tables, or create, verify, and push release tags. Also use for requests such as “整理 release”, “拆分 commit 区间”, “补原始代码链接”, “更新 release 表格”, or “打 tag”. Do not use for ordinary implementation work that does not involve release documentation or version tags.
compatibility: Requires git and a GitHub remote; GitHub CLI is useful when release content comes from a PR.
---

# MiniVLLM Release Maintainer

Maintain a cumulative, learnable release history for this repository. A learner should be able to follow releases chronologically, inspect the buggy implementation, understand when a bug appears, study the fix, and check out the matching code with a Git tag.

Before editing release documentation, read [references/release-format.md](references/release-format.md). It defines the repository's required bilingual document and README table structure.

## Core principles

1. Treat `main` as cumulative product support. A new release should not silently remove behavior supported by an older release.
2. Treat tags as immutable historical code checkpoints. Never move, replace, or delete an existing release tag unless the user explicitly requests that destructive change.
3. Build every factual claim from the repository history. Read the actual diff, the buggy parent revision, the fixed revision, and relevant tests.
4. Keep English and Simplified Chinese documents structurally equivalent. Their prose may be idiomatic, but their sections and link targets should match.
5. Make the documentation useful for learning. Explain non-obvious execution order, state transitions, tensor shapes, Block/KV mappings, and scheduler behavior instead of merely restating commit messages.

## 1. Inspect repository state safely

Start with read-only checks:

```bash
git status --short --branch
git branch --all --verbose --no-abbrev
git tag --list --sort=version:refname
git log --oneline --decorate --graph --all -n 50
```

Preserve unrelated modified or untracked files. Do not clean, reset, stash, commit, merge, or push work outside the user's requested scope.

Resolve the exact commit topology rather than inferring it from displayed log order:

```bash
git rev-parse <fix-commit>^
git merge-base <start> <end>
git log --reverse --format='%H %s' <base>..<end>
```

The parent of a fixing commit is normally the correct revision for an “Original code” link.

## 2. Analyze and group commits

Inspect every candidate commit with both statistics and diffs:

```bash
git show --stat --summary <commit>
git show --unified=5 <commit> -- <relevant-files>
git show <revision>:<file> | nl -ba
```

Group commits chronologically using these rules:

- Keep a large, self-contained change as its own release.
- Combine small consecutive commits only when they implement or refine the same feature.
- Do not combine unrelated correctness, model-loading, scheduling, or Kernel work merely to reduce the release count.
- Make each release endpoint runnable and conceptually teachable.
- Cover the requested commit interval exactly once: no missing commits and no overlapping release ranges.

When the user already specifies release boundaries, preserve them unless repository evidence shows they are invalid.

## 3. Find the actual buggy code

For every Bug item:

1. Identify the commit that fixes it.
2. Inspect `<fix-commit>^` to find the last revision containing the bug.
3. Locate the smallest useful line range that demonstrates the faulty behavior.
4. Add an individual GitHub permalink at the end of that Bug item.

Use a full immutable commit SHA in code links:

```text
https://github.com/<owner>/<repo>/blob/<full-sha>/<path>#L<start>-L<end>
```

Do not use `main`, a branch name, or a release tag for Original/Fix code links because their line numbers may change or the tag may point to a different documentation state.

Each Bug bullet or standalone Bug paragraph must contain at least one original-code link. Do not place one shared link after a group of unrelated Bug bullets.

For a release containing multiple commits, a Bug fixed by a later commit should link to that fixing commit's parent, not automatically to the previous release endpoint.

If the bug comes from an omitted validation or missing branch, link to the original function that accepted or processed the invalid state. Make the prose explicit that the absence of a check is the defect.

## 4. Explain when a bug occurs

Add an `Example` / `示例` block for bugs whose trigger is not obvious. Prefer concrete values taken from the code's abstractions:

- Token and Block examples: `block_size`, logical Block IDs, physical Block IDs, `ref_count`, cached Tokens.
- Scheduler examples: contents of `waiting` and `running`, token budgets, and the order of Prefill/Decode passes.
- Attention examples: Q length, full KV length, cached prefix length, causal positions, and Block Table translation.
- Tensor-parallel examples: global source Shape, TP size, rank-local Shape, and packed destination regions.
- CUDA Graph examples: captured and runtime Shapes, Dtypes, and Batch sizes.
- Multi-GPU examples: Rank state, rendezvous resources, serialization, and where a failure appears as a hang.

Skip examples for self-evident validation bugs such as rejecting a non-positive scalar unless an example materially clarifies the consequence.

An example should answer:

1. What inputs or runtime state trigger the bug?
2. Which old condition or data mapping behaves incorrectly?
3. What incorrect result, exception, stall, or resource error follows?

## 5. Explain the fix in implementation order

Describe the fix more deeply than the commit message. Follow the actual data flow:

- State what is computed first.
- Explain how state, Shapes, offsets, queues, or ownership change.
- Explain the boundary condition or invariant enforced.
- Explain why the old failure can no longer occur.
- Link each Fix item to the fixed code at an immutable commit SHA.

For complex fixes, separate mechanisms into multiple bullets. For example, distinguish cache-hit discovery, capacity admission, physical allocation, Block Table updates, and reference-count changes.

Do not claim runtime validation that was not performed. Distinguish committed tests, tests run in the current environment, and checks that still require CUDA or multiple GPUs.

## 6. Write bilingual release documents

Create one pair per release:

```text
releases/release-N.md
releases/release-N_zh.md
```

Follow the exact structure in the format reference. At minimum include:

- Summary heading with Commit ID range.
- Corresponding Tag and a link to the remote tag tree.
- Concise release purpose.
- Major Fixes grouped by subsystem.
- Files, Bugs, Examples where useful, and detailed Fixes.
- Validation or Result section supported by evidence.

For a single-commit release, display that one commit as the Commit ID range value rather than inventing a duplicated `A → A` range.

Keep technical identifiers, filenames, formulas, Tensor Shapes, and code snippets unchanged between languages. Translate link labels but preserve URL targets and ordering.

## 7. Update both README tables

Update `README.md` and `README_zh.md`. Use a Markdown table, never a long bullet list.

Keep releases in chronological learning order unless the user explicitly asks for newest-first ordering. The English and Chinese tables must contain the same releases, commit ranges, tag links, and document links.

Do not overwrite unrelated README edits.

## 8. Create and verify tags

Use the existing naming convention:

```text
release-1
release-2
...
```

Default each annotated tag to the final code commit in its release range:

```bash
git tag -a release-N <release-end-commit> -m 'Release N: <short description>'
```

Before creating it, confirm the name does not already exist. If it exists and resolves to the expected commit, keep it. If it points elsewhere, stop and ask the user; do not force-update it.

Verify local annotated tags:

```bash
git cat-file -t release-N
git rev-parse 'release-N^{commit}'
git tag -n1 --list 'release-*' --sort=version:refname
```

Creating a tag does not imply permission to push it. Push only when the user asks:

```bash
git push origin release-N
```

For multiple requested tags, name them explicitly rather than using `--tags`, which could publish unrelated local tags.

Verify remote tags after pushing:

```bash
git ls-remote --tags origin 'refs/tags/release-*' 'refs/tags/release-*^{}'
```

If Git reports that the repository moved, update only release Tag links to the canonical remote repository. Preserve historical code permalinks unless the user asks to migrate them or they are broken.

## 9. Validate the finished work

Run deterministic documentation checks:

```bash
git diff --check
```

Also verify:

- Every English release has a Chinese partner.
- Every Bug item contains an `https://` original-code link.
- English and Chinese partner files contain identical URL targets in identical order.
- Every README local link points to an existing release document.
- Every README Tag link matches the Tag declared inside its release documents.
- Release ranges cover the requested commits exactly once.
- Each local or remote Tag resolves to the intended release endpoint.

Use `rg` and `diff` for these checks. Report platform limitations when runtime tests cannot run, but do not conflate documentation validation with model/Kernel runtime validation.

## Handoff

Lead with the outcome. Report:

- Release boundaries and themes.
- Files created or updated.
- Tag-to-commit mapping.
- Whether tags are local or pushed and how remote verification was performed.
- Documentation and runtime checks performed.
- Any work intentionally left uncommitted or unpushed.

Never claim that README or release documents were pushed merely because tags were pushed; tags and branch content are separate Git operations.
