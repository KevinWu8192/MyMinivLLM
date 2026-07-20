# MiniVLLM release format

Read this reference before creating or revising release documentation.

## English release document

```markdown
# Summary — Commit ID Range: [`<short-start>`](<start-commit-url>) → [`<short-end>`](<end-commit-url>)

**Tag:** [`release-N`](https://github.com/<owner>/<repo>/tree/release-N)

<One concise paragraph describing the release outcome.>

## Major Fixes

### 1. <Subsystem or feature>

**Files**

* `path/to/file.py`

**Bugs**

* <One independently understandable bug.> ([Original code](<immutable-parent-code-url>))
* <Another bug.> ([Original code](<immutable-parent-code-url>))

**Example**

<Concrete trigger, old behavior, and consequence. Omit when the bug is obvious.>

**Fixes**

* <Detailed implementation step and why it works.> ([Fix](<immutable-fixed-code-url>))

## Validation

<Committed tests, current-run verification, remaining hardware-dependent checks.>
```

For a single-commit release, use:

```markdown
# Summary — Commit ID Range: [`<short-sha>`](<commit-url>)
```

## Chinese release document

Use the same structure and URL order:

```markdown
# 摘要 — Commit ID 区间：[`<short-start>`](<start-commit-url>) → [`<short-end>`](<end-commit-url>)

**Tag：** [`release-N`](https://github.com/<owner>/<repo>/tree/release-N)

<一句简洁的 Release 结果说明。>

## 主要修复

### 1. <子系统或功能>

**涉及文件**

* `path/to/file.py`

**问题**

* <一条能够独立理解的问题描述。> ([原始代码](<immutable-parent-code-url>))
* <另一条问题描述。> ([原始代码](<immutable-parent-code-url>))

**示例**

<具体触发条件、旧行为和后果。问题很直观时省略。>

**修复**

* <详细实现步骤及其正确性原因。> ([修复](<immutable-fixed-code-url>))

## 验证

<已提交测试、本次实际验证、仍依赖硬件的检查。>
```

## English README table

```markdown
## Recent Releases

| Release | Commit range | Tag | Highlights | Documentation |
|---|---|---|---|---|
| Release N | `<range>` | [`release-N`](<tag-url>) | <short highlights> | [English](releases/release-N.md) · [简体中文](releases/release-N_zh.md) |
```

## Chinese README table

```markdown
## 最近发布

| Release | Commit 区间 | Tag | 主要内容 | 文档 |
|---|---|---|---|---|
| Release N | `<range>` | [`release-N`](<tag-url>) | <简短主要内容> | [简体中文](releases/release-N_zh.md) · [English](releases/release-N.md) |
```

## Link rules

- Use full immutable SHAs for code permalinks.
- Give every Bug item its own Original-code link.
- Link a Bug to the last revision where it exists, normally `<fix-commit>^`.
- Use the canonical remote repository for Tag tree links.
- Keep English and Chinese URL sequences identical.
