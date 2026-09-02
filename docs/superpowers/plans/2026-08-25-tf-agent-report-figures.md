# TF-agent Report Figures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate, audit, and embed 13 Nature-style Python figures into the complete TF-agent paper experiment report without altering application code or inventing experimental results.

**Architecture:** A single Python figure builder reads the reviewed canonical artifact as its source of truth, writes versioned SVG/PDF/PNG outputs plus a figure manifest and QA notes, and keeps every figure tagged by evidence status. The existing JavaScript artifact generator reads the figure manifest, embeds the SVGs as data URIs in canonical `html` blocks, preserves all existing report content, and hands the complete artifact to the packaged portable HTML builder.

**Tech Stack:** Python 3.10, matplotlib, NumPy, Pillow, Nature-figure QA scripts, Node.js ES modules, canonical Data Analytics report artifact builder.

**Spec:** `docs/research/tf_agent_report_figure_design_2026-08-25.md`

## Global Constraints

- Use Python exclusively for drawing, previewing, and exporting all 13 figures.
- Do not modify `TF-agent/app.py`, `TF-agent/conversation_store.py`, `tests/unit/test_chat_ui_contract.py`, or `tests/unit/test_conversation_store.py`.
- Do not present proposed experiments, planning estimates, or illustrative mechanisms as measured results.
- Preserve the complete existing report artifact; additions are limited to 13 figure blocks, 8 adjacent explanatory blocks, figure metadata, and directly dependent figure-registry text.
- Every figure exports SVG, PDF, and PNG; every rendered glyph must be at least 5 pt.
- Do not commit or push from the shared dirty checkout; verify the scoped diff instead.

---

### Task 1: Freeze figure contracts and source rows

**Files:**
- Create: `docs/research/figures/build_report_figures.py`
- Create: `docs/research/figures/figure_source_data.json`
- Create: `docs/research/figures/figure_manifest.json`

**Interfaces:**
- Consumes: `docs/research/tf_agent_paper_experiment_report_nature_reviewed_2026-08-25.artifact.json`.
- Produces: exactly 13 manifest entries with `id`, `section`, `title`, `claim`, `status`, `archetype`, `source_ids`, `alt`, `caption`, `width_mm`, `height_mm`, and output paths.

- [ ] Parse the reviewed artifact and assert that all required datasets and all 14 related-work rows exist.
- [ ] Write the 13 figure contracts and the reviewed rows used by R2, R3, and R6 to `figure_source_data.json`.
- [ ] Validate status labels so only `verified literature`, `implemented baseline`, `proposed method`, `planned experiment`, and `planning estimate` are allowed.
- [ ] Run the builder in metadata-only mode and assert that exactly 13 complete contracts are emitted.

### Task 2: Implement the publication figure system and render 13 figures

**Files:**
- Modify: `docs/research/figures/build_report_figures.py`
- Create: `docs/research/figures/R01_*` through `R13_*` in SVG/PDF/PNG form.

**Interfaces:**
- Consumes: figure contracts and reviewed artifact rows from Task 1.
- Produces: `draw_r01(...)` through `draw_r13(...)` and a shared `save_figure(...)` export helper.

- [ ] Define the 180–183 mm publication canvas, 5 pt glyph floor, editable SVG/PDF settings, restrained palette, panel-label helper, and direct-label helpers.
- [ ] Implement R1–R6 with explicit status banners and no fabricated outcome values.
- [ ] Render R1–R6 and inspect their PNGs at final aspect ratio.
- [ ] Implement R7–R13 with explicit status banners and no fabricated outcome values.
- [ ] Render R7–R13 and inspect their PNGs at final aspect ratio.
- [ ] Re-run all 13 exports from a clean output list and assert every SVG/PDF/PNG exists and is non-empty.

### Task 3: Run Nature-figure source, text, collision, and panel QA

**Files:**
- Create: `docs/research/figures/qa/*.text-audit.json`
- Create: `docs/research/figures/qa/*.collision-audit.json`
- Create: `docs/research/figures/qa/*.collision-overlay.pdf`
- Create: `docs/research/figures/figure_qa_notes_2026-08-25.md`

**Interfaces:**
- Consumes: final plotting source and 13 exported PDFs/PNGs.
- Produces: one consolidated QA ledger with source-preflight, glyph-floor, collision, per-panel, and full-figure decisions.

- [ ] Run `validate_figure.py` on the plotting source and resolve every FAIL.
- [ ] Run `audit_pdf_text.py --min-pt 5 --json` for every PDF and resolve every failure.
- [ ] Run `audit_figure_collisions.py` for every PDF with JSON and overlay outputs; fix all blocking findings.
- [ ] Inspect every PNG panel and complete figure at final physical size; record why any remaining WARN is intentional and legible.
- [ ] Verify all SVGs contain editable `<text>` elements and all images use the declared palette.

### Task 4: Embed figures in the canonical report artifact

**Files:**
- Modify: `docs/research/build_tf_agent_paper_report_nature_reviewed.mjs`
- Regenerate: `docs/research/tf_agent_paper_experiment_report_nature_reviewed_2026-08-25.artifact.json`

**Interfaces:**
- Consumes: `docs/research/figures/figure_manifest.json` and the 13 SVG exports.
- Produces: 13 canonical `html` blocks using `body` with embedded `data:image/svg+xml;base64,...`, plus 8 adjacent explanatory markdown blocks.

- [ ] Add deterministic manifest loading, SVG base64 encoding, semantic `<figure>` HTML, captions, status, and alt text.
- [ ] Insert the eight figure groups at stable existing block anchors without deleting or reordering unrelated blocks.
- [ ] Add scoped structural assertions: original block IDs remain, 13 figure IDs are unique, all data URIs are embedded, and external URLs are absent from figure bodies.
- [ ] Regenerate the complete artifact and compare old/new block, chart, table, dataset, and source counts.

### Task 5: Package and verify the complete self-contained HTML

**Files:**
- Regenerate: `docs/research/tf_agent_paper_experiment_report_nature_reviewed_2026-08-25.html`
- Create: `docs/research/figures/verify_report_figures.py`

**Interfaces:**
- Consumes: full revised artifact.
- Produces: portable HTML plus a deterministic verification receipt from the packaged builder.

- [ ] Run `verify_report_figures.py` to assert 13 figure blocks, 13 embedded SVGs, 8 adjacent notes, unchanged native chart/table counts, and complete output bundles.
- [ ] Run the canonical `report:deliver` command once from the Data Analytics plugin root.
- [ ] Confirm the receipt reports `verification: passed`, expected block/chart/table/html counts, desktop and 390 px checks, no network calls, and no browser errors.
- [ ] Check final file size, self-contained CSP, source dialog behavior, and scoped Git status.
- [ ] Re-read the design spec line by line and record any unmet requirement before handoff.

