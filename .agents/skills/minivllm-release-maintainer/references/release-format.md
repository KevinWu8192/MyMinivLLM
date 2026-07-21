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
```

When a release adds a benchmark or another measurable feature, put its `Results` subsection immediately below that feature. Include the test hardware, workload shape, key metrics, interpretation, and links to committed raw results there.

A standalone `## Validation` section is optional. Use it only for release-relevant evidence that does not fit naturally beside a feature or fix. Do not add routine boilerplate about the current machine architecture, missing GPU, unavailable dependencies, or tests that could not be rerun.

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
```

当 Release 新增 Benchmark 或其他可测量功能时，将其 `结果` 子节直接放在对应功能下面，并在同一处说明测试硬件、负载 Shape、关键指标、结果分析及已提交的原始结果链接。

独立的 `## 验证` 章节是可选项。仅当存在无法自然放在功能或修复旁边、且对 Release 有独立价值的证据时使用。不要例行写入当前机器架构、没有 GPU、依赖缺失或无法重复运行测试等环境说明。

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
- Focus release content on user-visible features, fixed defects, and meaningful results. Exclude file moves, naming cleanup, and directory organization unless they change a supported workflow or compatibility contract.
