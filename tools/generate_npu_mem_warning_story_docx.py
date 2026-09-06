#!/usr/bin/env python3
"""Generate the NPU memory anomaly warning Story design Word document."""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

# Huawei-style palette
RED = RGBColor(0xC8, 0x10, 0x2E)
NAVY = RGBColor(0x00, 0x2A, 0x5C)
HEADER_FILL = "1F4E79"
ALT_FILL = "F2F2F2"
STEP_FILL = "1F4E79"
WARN_FILL = "C65911"
SYNC_FILL = "2E75B6"
ASYNC_FILL = "548235"
CHECK_FILL = "7030A0"
CN_FONT = "WenQuanYi Micro Hei"
EN_FONT = "Times New Roman"
CODE_FONT = "Noto Sans Mono"


def set_run_font(run, size=11, bold=False, color=None, font=EN_FONT, east_asia=CN_FONT):
    run.bold = bold
    run.font.size = Pt(size)
    run.font.name = font
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.get_or_add_rFonts()
    rfonts.set(qn("w:ascii"), font)
    rfonts.set(qn("w:hAnsi"), font)
    rfonts.set(qn("w:eastAsia"), east_asia)
    rfonts.set(qn("w:cs"), font)
    if color is not None:
        run.font.color.rgb = color


def set_cell_shading(cell, hex_color: str):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    for child in list(tc_pr):
        if child.tag == qn("w:shd"):
            tc_pr.remove(child)
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), hex_color)
    shd.set(qn("w:val"), "clear")
    tc_pr.append(shd)


def set_cell_border(cell, **kwargs):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_borders = OxmlElement("w:tcBorders")
    for edge in ("top", "left", "bottom", "right"):
        if edge in kwargs:
            element = OxmlElement(f"w:{edge}")
            for key, value in kwargs[edge].items():
                element.set(qn(f"w:{key}"), str(value))
            tc_borders.append(element)
    tc_pr.append(tc_borders)


def clear_cell(cell):
    for p in list(cell.paragraphs):
        p.clear()
    if not cell.paragraphs:
        cell.add_paragraph()


def add_cell_text(cell, text, *, size=10.5, bold=False, color=None, align="left", fill=None, font=EN_FONT):
    if fill:
        set_cell_shading(cell, fill)
    p = cell.paragraphs[0]
    p.clear()
    p.alignment = {
        "left": WD_ALIGN_PARAGRAPH.LEFT,
        "center": WD_ALIGN_PARAGRAPH.CENTER,
        "right": WD_ALIGN_PARAGRAPH.RIGHT,
    }[align]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.15
    run = p.add_run(text)
    set_run_font(run, size=size, bold=bold, color=color, font=font)
    return p


def set_narrow_margins(section):
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.left_margin = Cm(2.0)
    section.right_margin = Cm(2.0)
    section.top_margin = Cm(2.2)
    section.bottom_margin = Cm(2.2)


def add_paragraph(doc, text, *, size=11, bold=False, color=None, align="left", space_after=8, space_before=0, first_line=False):
    p = doc.add_paragraph()
    p.alignment = {
        "left": WD_ALIGN_PARAGRAPH.LEFT,
        "center": WD_ALIGN_PARAGRAPH.CENTER,
        "right": WD_ALIGN_PARAGRAPH.RIGHT,
        "justify": WD_ALIGN_PARAGRAPH.JUSTIFY,
    }[align]
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    if first_line:
        p.paragraph_format.first_line_indent = Cm(0.74)
    run = p.add_run(text)
    set_run_font(run, size=size, bold=bold, color=color)
    return p


def add_mixed_paragraph(doc, parts, *, align="left", space_after=8, space_before=0, first_line=False):
    p = doc.add_paragraph()
    p.alignment = {
        "left": WD_ALIGN_PARAGRAPH.LEFT,
        "center": WD_ALIGN_PARAGRAPH.CENTER,
        "right": WD_ALIGN_PARAGRAPH.RIGHT,
        "justify": WD_ALIGN_PARAGRAPH.JUSTIFY,
    }[align]
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    if first_line:
        p.paragraph_format.first_line_indent = Cm(0.74)
    for text, kwargs in parts:
        run = p.add_run(text)
        set_run_font(run, **kwargs)
    return p


def add_heading_cn(doc, text, level=1):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(16 if level == 1 else 12)
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.keep_with_next = True
    size = {1: 16, 2: 14, 3: 12}.get(level, 12)
    color = RED if level == 1 else NAVY
    run = p.add_run(text)
    set_run_font(run, size=size, bold=True, color=color)
    return p


def set_table_full_width(table, widths):
    table.autofit = False
    table.allow_autofit = False
    for row in table.rows:
        for idx, width in enumerate(widths):
            row.cells[idx].width = width


def shade_header_row(table, fill=HEADER_FILL):
    for cell in table.rows[0].cells:
        set_cell_shading(cell, fill)
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in p.runs:
                run.bold = True
                run.font.color.rgb = RGBColor(255, 255, 255)
                set_run_font(run, size=10.5, bold=True, color=RGBColor(255, 255, 255))


def add_table(doc, headers, rows, col_widths):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_full_width(table, col_widths)
    for i, h in enumerate(headers):
        add_cell_text(table.rows[0].cells[i], h, size=10.5, bold=True, color=RGBColor(255, 255, 255), align="center", fill=HEADER_FILL)
    for r_idx, row in enumerate(rows):
        fill = ALT_FILL if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row):
            add_cell_text(table.rows[r_idx + 1].cells[c_idx], val, size=10, fill=fill, align="left" if c_idx else "center")
    doc.add_paragraph().paragraph_format.space_after = Pt(4)
    return table


def add_code_block(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.left_indent = Cm(0.4)
    p.paragraph_format.line_spacing = 1.15
    run = p.add_run(text)
    set_run_font(run, size=9, font=CODE_FONT, east_asia=CN_FONT)
    run.font.color.rgb = RGBColor(0x1A, 0x1A, 0x1A)
    return p


def add_flow_box(doc, text, fill, color=RGBColor(255, 255, 255), arrow=True):
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = table.rows[0].cells[0]
    cell.width = Cm(14.5)
    add_cell_text(cell, text, size=10.5, bold=True, color=color, align="center", fill=fill)
    set_cell_border(
        cell,
        top={"val": "single", "sz": "8", "color": fill},
        left={"val": "single", "sz": "8", "color": fill},
        bottom={"val": "single", "sz": "8", "color": fill},
        right={"val": "single", "sz": "8", "color": fill},
    )
    if not arrow:
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run("↓")
    set_run_font(run, size=14, bold=True, color=NAVY)


def add_branch_row(doc, left_text, right_text, left_fill, right_fill):
    table = doc.add_table(rows=1, cols=3)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    widths = [Cm(7.4), Cm(1.6), Cm(7.4)]
    set_table_full_width(table, widths)
    add_cell_text(table.rows[0].cells[0], left_text, size=10, bold=True, color=RGBColor(255, 255, 255), align="center", fill=left_fill)
    add_cell_text(table.rows[0].cells[1], "或", size=10, bold=True, color=NAVY, align="center", fill="FFFFFF")
    add_cell_text(table.rows[0].cells[2], right_text, size=10, bold=True, color=RGBColor(255, 255, 255), align="center", fill=right_fill)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run("↓")
    set_run_font(run, size=14, bold=True, color=NAVY)


def build_document() -> Document:
    doc = Document()
    section = doc.sections[0]
    set_narrow_margins(section)

    # Default style
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = EN_FONT
    normal.font.size = Pt(11)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), CN_FONT)

    # Header / footer
    header = section.header
    header.is_linked_to_previous = False
    hp = header.paragraphs[0]
    hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = hp.add_run("vLLM Ascend  |  仅供内部使用  |  For internal use only")
    set_run_font(run, size=8.5, color=RGBColor(0x66, 0x66, 0x66))

    footer = section.footer
    footer.is_linked_to_previous = False
    fp = footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = fp.add_run("华为技术有限公司  Huawei Technologies Co., Ltd.  —  版权所有 侵权必究")
    set_run_font(run, size=8, color=RGBColor(0x66, 0x66, 0x66))
    # page number field
    fp2 = footer.add_paragraph()
    fp2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = fp2.add_run("第 ")
    set_run_font(run, size=8, color=RGBColor(0x66, 0x66, 0x66))
    fld1 = OxmlElement("w:fldSimple")
    fld1.set(qn("w:instr"), "PAGE")
    r1 = OxmlElement("w:r")
    t1 = OxmlElement("w:t")
    t1.text = "1"
    r1.append(t1)
    fld1.append(r1)
    fp2._p.append(fld1)
    run = fp2.add_run(" 页")
    set_run_font(run, size=8, color=RGBColor(0x66, 0x66, 0x66))

    # ===== Cover =====
    add_paragraph(doc, "vLLM Ascend", size=12, bold=True, color=RED, align="center", space_after=2)
    add_paragraph(doc, "NPU Memory Anomaly Warning  ·  Story Implementation Design", size=11, color=NAVY, align="center", space_after=10)

    info = doc.add_table(rows=4, cols=4)
    info.style = "Table Grid"
    info.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_full_width(info, [Cm(4.2), Cm(4.5), Cm(4.2), Cm(4.5)])
    cover_rows = [
        ("产品名称 Product name", "vLLM Ascend", "密级 Confidentiality", "内部公开"),
        ("产品版本 Product version", "dsv4-sparse-flash-mla-bf16", "文档状态 Status", "试用稿"),
        ("基线提交 Baseline commit", "3f0d4b4b1", "文档版本 Version", "V1.0"),
        ("硬件形态 Hardware", "Atlas 800T/I A2/A3、Atlas 900 A3、A5", "日期 Date", "2026-09-06"),
    ]
    for i, (k1, v1, k2, v2) in enumerate(cover_rows):
        add_cell_text(info.rows[i].cells[0], k1, size=9, bold=True, color=RGBColor(255, 255, 255), fill=HEADER_FILL)
        add_cell_text(info.rows[i].cells[1], v1, size=9.5)
        add_cell_text(info.rows[i].cells[2], k2, size=9, bold=True, color=RGBColor(255, 255, 255), fill=HEADER_FILL)
        add_cell_text(info.rows[i].cells[3], v2, size=9.5)

    add_paragraph(doc, "", space_after=18)
    add_paragraph(doc, "支持显存异常预警", size=26, bold=True, color=RED, align="center", space_after=4)
    add_paragraph(doc, "Story（AR）实现设计说明书", size=20, bold=True, color=NAVY, align="center", space_after=4)
    add_paragraph(doc, "PrefixCache / Chunked Prefill / 并行解码 / MTP / SparseFlashMLA", size=12, color=NAVY, align="center", space_after=2)
    add_paragraph(doc, "（仅供内部使用）  For internal use only", size=11, color=RGBColor(0x88, 0x00, 0x00), align="center", space_after=22)

    sign = doc.add_table(rows=5, cols=4)
    sign.style = "Table Grid"
    sign.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_full_width(sign, [Cm(4.2), Cm(4.5), Cm(4.2), Cm(4.5)])
    sign_rows = [
        ("拟制 Prepared by", "杨锦阳", "日期 Date", "2026-09-06"),
        ("适配 Adapted for", "vLLM Ascend 当前分支", "基线分支 Branch", "dsv4-sparse-flash-mla-bf16"),
        ("审核 Reviewed by", "", "日期 Date", "yyyy-mm-dd"),
        ("审核 Reviewed by", "", "日期 Date", "yyyy-mm-dd"),
        ("批准 Granted by", "", "日期 Date", "yyyy-mm-dd"),
    ]
    for i, (k1, v1, k2, v2) in enumerate(sign_rows):
        add_cell_text(sign.rows[i].cells[0], k1, size=10, bold=True, color=RGBColor(255, 255, 255), fill="2E75B6")
        add_cell_text(sign.rows[i].cells[1], v1, size=10.5)
        add_cell_text(sign.rows[i].cells[2], k2, size=10, bold=True, color=RGBColor(255, 255, 255), fill="2E75B6")
        add_cell_text(sign.rows[i].cells[3], v2, size=10.5)

    add_paragraph(doc, "", space_after=28)
    add_paragraph(doc, "华为技术有限公司", size=14, bold=True, color=RED, align="center", space_after=2)
    add_paragraph(doc, "Huawei Technologies Co., Ltd.", size=12, color=NAVY, align="center", space_after=6)
    add_paragraph(doc, "版权所有  侵权必究    All rights reserved", size=10, align="center", space_after=0)

    doc.add_page_break()

    # ===== Version control =====
    add_heading_cn(doc, "修订记录 Version Control", 1)
    add_table(
        doc,
        ["日期 Date", "修订版本 Version", "描述 Description", "作者 Prepared by"],
        [
            [
                "2022-12-01",
                "模板 V1.0",
                "软件开发流程中 Story 设计说明书模板试用稿，用于产品 Story 设计输出参考。",
                "徐林、孙鹤、汪功俊、白嗣建、陈鹏、张辉、吴昊、王彬",
            ],
            [
                "2026-09-06",
                "V1.0",
                "以 MindIE 显存异常预警 Story 为模板，针对 vLLM Ascend 当前分支 dsv4-sparse-flash-mla-bf16 映射实现路径：将 generate_token / text_generator 替换为 NPUWorker.execute_model、NPUModelRunner 的 prepare_inputs / forward / sample / postprocess_sampled；补充 PrefixCache、Chunked Prefill、MTP、SparseFlashMLA 叠加场景下的 warmup 基线与 5% 剩余显存告警设计。",
                "杨锦阳",
            ],
        ],
        [Cm(3.0), Cm(2.8), Cm(8.2), Cm(3.0)],
    )

    add_paragraph(
        doc,
        "本 Story 设计说明书用于承载软件实现设计阶段流程规范要求，将项目在 Story 实现设计过程中对需求的功能点分析、实现设计、测试建议等内容落地。旨在提升设计质量，为代码开发提供高质量输入。本文档在 MindIE 原模板结构上，按 vLLM Ascend 当前分支真实代码路径改写，避免直接复用 MindIE 的 generate_token / text_generator 接口名称。",
        align="justify",
        first_line=True,
    )

    # ===== 1 Story overview =====
    add_heading_cn(doc, "1  Story概述（必要）", 1)
    add_table(
        doc,
        ["包需求名称", "设计需求名称", "Story 名称", "Story 描述", "是否做 Story 设计"],
        [
            [
                "可靠性 / OOM 防护",
                "NPU 显存增长预警与请求规格校验",
                "PrefixCache / Chunked Prefill / 并行解码 / MTP / SparseFlashMLA 插件特性叠加避免出现 OOM 故障",
                "显存增长预警：warmup 与推理 prepare_inputs、forward、sample、postprocess 后打印显存占用；推理中剩余显存与 warmup 后剩余显存相比每减少 5%，打印告警。请求 shape 校验：execute_model 入口检查 max_model_len 与 max_num_seqs，异常则打印告警。",
                "是",
            ],
        ],
        [Cm(2.6), Cm(3.4), Cm(3.8), Cm(5.2), Cm(2.0)],
    )

    add_paragraph(doc, "【需求来源】现网 OOM 故障问题；当前分支仅在 warmup 结束与 execute_model 入口 DEBUG 日志中打印显存，推理阶段关键步骤无占用快照，也无法在剩余显存相对 warmup 基线下滑时提前告警。", first_line=True, align="justify")
    add_paragraph(doc, "【需求场景】PrefixCache、Chunked Prefill（SplitFuse / chunked prefill）、并行解码、MTP 投机解码，以及当前分支新增的 A5 BF16 SparseFlashMLA KV 路径叠加时，不出现 OOM 故障。", first_line=True, align="justify")
    add_paragraph(doc, "【需求收益】提升可靠性：现网出现显存异常增长时可在日志中定位阶段（prepare / forward / sample / postprocess），辅助优化配置、定位泄漏，保障推理稳定运行。", first_line=True, align="justify")
    add_paragraph(doc, "【硬件产品形态】Atlas 800T/I A2，Atlas 800T/I A3，Atlas 900 A3；当前分支同时覆盖 Ascend A5 上 DeepSeek-V4 BF16 SparseFlashMLA KV 路径。", first_line=True, align="justify")
    add_paragraph(doc, "【需求内容描述】PrefixCache / Chunked Prefill / 并行解码 / MTP / SparseFlashMLA 等插件的初始化与 workspace 预留放到 Warmup 前完成，确保 Warmup 的内存请求包含这些插件的需求。当前分支已将 DSA 注意力 KV 计划在 get_dsa_attn_kv_plan() 中于模型运行前选定 SparseFlashMLA 或 FP8 量化路径，避免推理期首次加载算子导致额外显存。", first_line=True, align="justify")

    add_heading_cn(doc, "【落地方案】", 3)
    add_paragraph(doc, "1. 显存增长预警：新增关键调用点显存统计。warmup（compile_or_warm_up_model / profile_run / _dummy_run）走同步分支，仅打印占用并记录剩余显存基线；推理 prepare_inputs、forward、sample、postprocess 后走异步分支，相比 warmup 阶段剩余显存每减少 >5%（相对比例），打印当前占用，接着打印显存告警。", first_line=True, align="justify")
    add_paragraph(doc, "2. 推理请求 shape 校验：NPUModelRunner.execute_model 收到 SchedulerOutput / InputBatch 后，检查本次请求规格。若 seq_len 超过 max_model_len，或 batch 超过 max_num_seqs，打印异常，不中断主路径（告警而非抛异常，避免把可观测性做成新的故障点）。", first_line=True, align="justify")

    add_heading_cn(doc, "1.1  当前分支现状与缺口", 2)
    add_table(
        doc,
        ["现有能力", "代码位置", "缺口"],
        [
            [
                "warmup 结束后打印 reserved / allocated",
                "vllm_ascend/worker/worker.py::compile_or_warm_up_model",
                "只打一次总结，不作为推理期对比基线；无 5% 阶梯告警。",
            ],
            [
                "execute_model 入口 DEBUG 显存日志",
                "NPUWorker.log_memory_stats()",
                "仅 DEBUG 级别，且只在入口，不覆盖 prepare / forward / sample / postprocess。",
            ],
            [
                "KV cache 预算 profiling",
                "NPUWorker.determine_available_memory / profile_run",
                "用于计算 KV 容量，不是推理中增长预警。",
            ],
            [
                "sampler warmup workspace 注释",
                "vllm_ascend/worker/v2/sample/apply_top_k_top_p.py",
                "已知 warmup 未预留 top-k/top-p workspace 可能导致后续 OOM，但无运行期告警。",
            ],
            [
                "部分 assert 校验",
                "model_runner_v1.py 中 end_idx <= max_model_len",
                "assert 会直接失败；缺少统一的异常规格打印，且 V2 prepare_inputs 入口无对等告警。",
            ],
        ],
        [Cm(4.2), Cm(6.4), Cm(6.4)],
    )

    # ===== 2 Context =====
    add_heading_cn(doc, "2  Story上下文（必要）", 1)
    add_paragraph(doc, "MindIE 原设计在 generate_token 中按 warmup 参数分流。vLLM Ascend 当前分支没有同名 generate_token，对应关系如下。", first_line=True, align="justify")

    add_table(
        doc,
        ["MindIE 概念（模板）", "vLLM Ascend 当前分支映射", "说明"],
        [
            ["generate_token(warmup=...)", "NPUWorker.execute_model + NPUModelRunner.execute_model / sample_tokens", "推理主循环；warmup 不走该入口的业务请求路径。"],
            ["warmup 阶段", "compile_or_warm_up_model、profile_run、_dummy_run、kernel_warmup、capture_model", "图编译、Triton kernel、ACLGraph 捕获均须计入基线。"],
            ["preprocess", "V1: _prepare_inputs；V2: prepare_inputs", "构建 InputBatch、attn metadata、slot mapping。"],
            ["forward", "model forward（execute_model 内）", "含 MLA / DSA / SparseFlashMLA、MTP draft 前向。"],
            ["sample", "V1: sample_tokens / _sample；V2: sample / sample_tokens", "含 rejection sampler、lmhead TP 对齐。"],
            ["postprocess", "V2: postprocess_sampled；V1: _update_states_after_model_execute 等", "写回 token、同步 CPU seq_lens。"],
            ["text_generator + input_metadata", "SchedulerOutput + InputBatch / AscendInputBatch", "规格字段为 max_model_len、max_num_seqs。"],
            ["watch_npu_mem", "拟新增 vllm_ascend/utils/npu_memory_watch.py::watch_npu_mem", "warmup 同步、推理异步。"],
            ["max_seq_len / max_batch_size", "model_config.max_model_len / scheduler_config.max_num_seqs", "与 vLLM 调度器配置对齐。"],
        ],
        [Cm(4.2), Cm(6.8), Cm(6.0)],
    )

    add_paragraph(doc, "1. 在 prepare_inputs、forward、sample、postprocess 模块后调用 watch_npu_mem 打印显存占用和显存预警。", first_line=True, align="justify")
    add_paragraph(doc, "2. watch_npu_mem 功能定位：首先判断是 warmup 阶段还是推理阶段。若为 warmup 阶段，走同步分支，仅打印显存占用，并写入 warmup_free_bytes 基线。若为推理阶段，走异步分支，此时剩余显存相比 warmup 后每减少 5%，打印一次目前的显存占用，接着再打印显存预警。异步是为了避免 torch.npu.mem_get_info() 在热路径上引入 Host-Device 同步，符合当前仓库 AGENTS.md 对 NPU 同步开销的约束。", first_line=True, align="justify")
    add_paragraph(doc, "3. 当前分支双 Runner：默认 Model Runner V2（vllm_ascend/worker/v2/model_runner.py），兼容 V1（vllm_ascend/worker/model_runner_v1.py）。watch 点必须在两个 Runner 对等插入，避免只覆盖一条路径。", first_line=True, align="justify")
    add_paragraph(doc, "4. 当前分支 SparseFlashMLA：A5 + cache_dtype=bfloat16 时，get_dsa_attn_kv_plan() 选择 cann_ops_transformer.sparse_flash_mla。该算子 workspace 必须在 warmup dummy run 中真正执行到，否则推理首包才申请 workspace，会表现为相对 warmup 基线的阶梯掉量并触发本 Story 告警。", first_line=True, align="justify")

    # ===== 3 Features =====
    add_heading_cn(doc, "3  功能点分解（必要）", 1)
    add_table(
        doc,
        ["序号", "功能点名称", "功能点描述", "当前分支落点"],
        [
            [
                "1",
                "显存预警，显存打印",
                "推理 prepare_inputs、forward、sampling、postprocess 后打印显存占用，且剩余显存相比 warmup 后剩余显存每减少 5%，打印告警。warmup 阶段同步打印占用并记录基线。",
                "新增 watch_npu_mem；NPUWorker.compile_or_warm_up_model 结束记录基线；V1/V2 execute_model 四阶段后调用。",
            ],
            [
                "2",
                "推理请求 shape 校验",
                "execute_model 收到 SchedulerOutput 后检查请求规格：本 batch 最大 seq_len 以及 batch size。若超过 max_model_len 或 max_num_seqs，打印异常。",
                "NPUModelRunner.execute_model 入口（dummy_run / is_profile 为 False 时）；校验 InputBatch.num_reqs 与 seq_lens。",
            ],
        ],
        [Cm(1.4), Cm(3.4), Cm(6.6), Cm(5.6)],
    )

    add_heading_cn(doc, "3.1  告警阈值与打印策略", 2)
    add_paragraph(doc, "记 warmup 结束后设备剩余显存为 F_warmup（bytes），推理某观察点剩余显存为 F_now。相对下降比例：", first_line=True, align="justify")
    add_code_block(doc, "drop_ratio = (F_warmup - F_now) / max(F_warmup, 1)")
    add_paragraph(doc, "令 last_alarm_level 为已打印过的最高阶梯（初始 0）。当 drop_ratio >= (last_alarm_level + 1) * 0.05 时，打印当前占用，再打印 WARNING，并将 last_alarm_level 更新为 floor(drop_ratio / 0.05)。即每跨过一个 5% 台阶只告警一次，避免每个 token 刷屏。剩余显存回升不回退阶梯，防止抖动重复告警；进程重启或重新 warmup 后清零。", first_line=True, align="justify")
    add_paragraph(doc, "同步分支（warmup）使用 logger.info；异步分支（推理）占用使用 logger.info，预警使用 logger.warning。日志字段至少包含：rank、tag、allocated、reserved、free、total、drop_ratio、warmup_free。", first_line=True, align="justify")

    # ===== 4 Design =====
    add_heading_cn(doc, "4  实现设计（必要）", 1)
    add_heading_cn(doc, "4.1  功能实现思路（必要）", 2)
    add_paragraph(
        doc,
        "当前 vLLM Ascend 没有面向推理阶段的显存异常告警，导致无法提前审视 NPU 内存使用情况。内存使用异常时容易出现 OOM，而目前仅在 warmup 结束有一次 reserved/allocated 打印，execute_model 入口仅 DEBUG 可见。现网 OOM 后无法准确分析是 prepare、forward、sample 还是 postprocess 导致增长，定位复杂。PrefixCache、Chunked Prefill、MTP、并行解码以及当前分支 SparseFlashMLA 都会在 warmup 之外引入 workspace / KV / draft 缓冲；若插件未在 warmup 真正跑到，推理期会突然多占显存。因此本版本新增显存增长预警和推理请求 shape 校验：在 warmup、prepare_inputs、forward、sample、postprocess 增加统计；推理阶段剩余显存相对 warmup 每减少 5% 打印告警，用于辅助优化配置、定位泄漏，保障推理稳定运行。",
        first_line=True,
        align="justify",
    )

    add_heading_cn(doc, "4.1.1  流程图", 3)
    add_paragraph(doc, "下图为当前分支 execute_model 主路径上的分流与观察点（对应原模板 generate_token 流程）。", space_after=6)

    add_flow_box(doc, "入口：NPUWorker.execute_model(scheduler_output)", STEP_FILL)
    add_flow_box(doc, "是否处于 warmup？（compile_or_warm_up_model / dummy_run / profile_run）", CHECK_FILL)
    add_branch_row(
        doc,
        "是：同步 watch_npu_mem\n仅打印占用，更新 warmup_free 基线",
        "否：推理路径\n先做 shape 校验，再分阶段异步 watch",
        SYNC_FILL,
        ASYNC_FILL,
    )
    add_flow_box(doc, "shape 校验：num_reqs ≤ max_num_seqs，seq_len ≤ max_model_len\n异常则 logger.error，不中断主路径", WARN_FILL)
    add_flow_box(doc, "prepare_inputs / _prepare_inputs  →  watch_npu_mem(tag=preprocess)", STEP_FILL)
    add_flow_box(doc, "model forward（含 SparseFlashMLA / MTP）  →  watch_npu_mem(tag=forward)", STEP_FILL)
    add_flow_box(doc, "sample / sample_tokens  →  watch_npu_mem(tag=sample)", STEP_FILL)
    add_flow_box(doc, "postprocess_sampled  →  watch_npu_mem(tag=postprocess)", STEP_FILL)
    add_flow_box(doc, "异步比较：剩余显存较 warmup_free 每下降 5% → 先打占用，再打 WARNING", WARN_FILL, arrow=False)

    # remove last extra arrow: add a closing note instead of another arrow after last box
    # (the helper always adds an arrow; add a finish label)
    add_paragraph(doc, "结束本 step；阶梯计数 last_alarm_level 进程内保持。", size=10, align="center", color=NAVY, space_after=10)

    add_heading_cn(doc, "4.1.2  流程说明", 3)
    add_paragraph(
        doc,
        "在 NPUModelRunner.execute_model 中，首先通过 dummy_run / is_profile / 尚未完成 compile_or_warm_up_model 判断是 warmup 还是推理。若为推理，则在入口处进行 shape 校验：读取本 batch 的 num_reqs 与各请求 seq_len（V2 使用 input_buffers.seq_lens_np，V1 使用 input_batch 中已更新的 CPU 侧长度）。若 max_model_len 或 max_num_seqs 被突破，打印异常。此后在 prepare_inputs、forward、sample、postprocess 之后均对显存进行检查；剩余显存相比 warmup 后每减少 5%，打印一次目前的显存占用，接着再打印显存告警。",
        first_line=True,
        align="justify",
    )
    add_paragraph(
        doc,
        "warmup 判定不能依赖 MindIE 的 warmup 布尔参数。当前分支约定：NPUWorker 在 compile_or_warm_up_model 成功返回前，watch_npu_mem 一律视为 warmup 同步路径；该函数返回后置位 _npu_mem_warmup_done，并冻结 warmup_free_bytes。profile_run 与 ACLGraph capture 过程中的 dummy forward 计入 warmup。Serving 阶段即使内部仍有 _dummy_run（例如 DP idle rank 对齐），也视为推理路径，避免把运行期 dummy 占用写进基线。",
        first_line=True,
        align="justify",
    )

    add_heading_cn(doc, "4.1.3  与当前分支特性叠加的关系", 3)
    add_table(
        doc,
        ["特性", "OOM 风险点", "Warmup 必须覆盖的行为", "本 Story 如何帮助定位"],
        [
            [
                "PrefixCache",
                "公共前缀 block 与 cascade attention workspace",
                "dummy run 走到 prefix / cascade 元数据构建",
                "preprocess 后占用跳变可指向 metadata / block table",
            ],
            [
                "Chunked Prefill",
                "分块 prefill 与 decode 混合 batch 的 padding、attn workspace",
                "compile_sizes / capture_sizes 覆盖混合 shape",
                "forward 后告警多见于 attn workspace 未预留",
            ],
            [
                "并行解码 / 投机并行",
                "draft token 扩展后的 logits 行数、lmhead TP pad",
                "V2 _dummy_run 已对 lmhead TP 补 logits 集体通信",
                "sample 后告警指向 sampler / rejection workspace",
            ],
            [
                "MTP",
                "draft model、rejection sampler、seq_lens D2H",
                "kernel_warmup 中 rejection_sampler_triton_warmup",
                "postprocess 后告警可能来自 MTP CPU 同步缓冲",
            ],
            [
                "SparseFlashMLA（本分支）",
                "cann_ops_transformer 算子 workspace、PA_BBND scatter",
                "A5 BF16 计划必须在 dummy forward 中真实调用 sparse_flash_mla",
                "forward 后相对 warmup 掉 5%+ 可判定算子未 warmup",
            ],
        ],
        [Cm(3.2), Cm(4.6), Cm(4.6), Cm(4.6)],
    )

    add_heading_cn(doc, "4.2  接口描述（必要）", 2)
    add_paragraph(doc, "新增函数（相对原 MindIE 接口，保留同名以便对照，参数按当前分支裁剪）：", space_after=6)
    add_code_block(
        doc,
        """def watch_npu_mem(
    rank_id: int,
    tag: str,
    warmup: bool,
    warmup_mem_free: int | None = None,
    is_multimodal: bool = False,
    max_input_len: int | None = None,
) -> int:
    \"\"\"Detect NPU device memory and optionally emit a 5% free-memory warning.

    rank_id:        当前进程 local/global rank，写入日志。
    tag:            观察点标签，取值 warmup / preprocess / forward / sample / postprocess。
    warmup:         True 走同步打印并返回当前 free bytes 作为基线；False 走异步比较。
    warmup_mem_free: warmup 结束后剩余显存（bytes）。推理路径必填。
    is_multimodal:  多模态时附加 encoder cache 提示，便于区分视觉占用。
    max_input_len:  可选，打印时带上本步 max token，便于对照异常 shape。

    Returns:
        当前剩余显存 bytes（同步路径）；异步路径立即返回上次缓存值或 -1。
    \"\"\"""",
    )

    add_paragraph(doc, "原模板参数 max_input / is_multimodal 予以保留，但当前分支主路径以 rank、tag、warmup、warmup_mem_free 为准。内存读取使用 torch.npu.mem_get_info() 得到 (free, total)，并同时记录 torch.npu.memory_allocated() 与 torch.npu.memory_reserved()，与 compile_or_warm_up_model 现有日志口径一致。", first_line=True, align="justify")

    add_heading_cn(doc, "4.2.1  请求 shape 校验接口", 3)
    add_code_block(
        doc,
        """def check_request_shape(
    num_reqs: int,
    max_seq_len: int,
    max_num_seqs: int,
    max_model_len: int,
    tag: str = "execute_model",
) -> None:
    \"\"\"Log an error when the scheduled batch exceeds engine limits.

    不抛异常、不改变调度结果。max_seq_len 为本 batch 最大序列长度，
    max_model_len / max_num_seqs 来自 vllm_config。
    \"\"\"""",
    )

    add_heading_cn(doc, "4.2.2  环境变量", 3)
    add_paragraph(
        doc,
        "按 AGENTS.md 要求，新环境变量必须登记在 vllm_ascend/envs.py 的 env_variables 中，命名 VLLM_ASCEND_*。建议：",
        first_line=True,
        align="justify",
    )
    add_table(
        doc,
        ["变量", "默认", "范围", "是否敏感", "用途"],
        [
            [
                "VLLM_ASCEND_NPU_MEM_WATCH",
                "1",
                "0/1",
                "否",
                "总开关。关闭后不采样、不告警，避免对极致吞吐场景的干扰。",
            ],
            [
                "VLLM_ASCEND_NPU_MEM_WATCH_STEP",
                "0.05",
                "(0, 1)",
                "否",
                "相对 warmup 剩余显存的告警台阶，默认 5%。",
            ],
        ],
        [Cm(5.4), Cm(1.8), Cm(2.0), Cm(2.0), Cm(5.8)],
    )

    add_heading_cn(doc, "4.3  模块与文件变更", 2)
    add_table(
        doc,
        ["文件", "变更类型", "内容"],
        [
            [
                "vllm_ascend/envs.py",
                "修改",
                "登记 VLLM_ASCEND_NPU_MEM_WATCH、VLLM_ASCEND_NPU_MEM_WATCH_STEP。",
            ],
            [
                "vllm_ascend/utils/npu_memory_watch.py（新建）",
                "新增",
                "watch_npu_mem、check_request_shape、异步工作线程与阶梯状态。",
            ],
            [
                "vllm_ascend/worker/worker.py",
                "修改",
                "compile_or_warm_up_model 结束调用同步 watch 并保存基线；execute_model 可保留现有 DEBUG log_memory_stats，不重复打 INFO。",
            ],
            [
                "vllm_ascend/worker/v2/model_runner.py",
                "修改",
                "execute_model 入口 shape 校验；prepare_inputs 后、sample 后、postprocess_sampled 后插入异步 watch。forward 观察点放在 super().execute_model 返回后（V2 将 prepare/forward 封装在父类）。",
            ],
            [
                "vllm_ascend/worker/model_runner_v1.py",
                "修改",
                "_prepare_inputs 后、模型 forward 后、sample_tokens / _sample 后插入对等 watch 与 shape 校验。",
            ],
            [
                "tests/ut/utils/test_npu_memory_watch.py（新建）",
                "新增",
                "覆盖台阶计算、warmup/推理分流、非法 seq_len/batch 告警。",
            ],
        ],
        [Cm(5.4), Cm(2.2), Cm(9.4)],
    )

    add_heading_cn(doc, "4.4  异步实现约束", 2)
    add_paragraph(doc, "推理路径禁止在热循环内同步调用 mem_get_info 后立刻 .item() 式等待。推荐单线程队列：观察点只投递 (rank, tag, timestamp)，后台线程执行 mem_get_info 与日志。阶梯状态用线程锁保护。进程退出时 daemon 线程可丢弃未处理项。ACLGraph replay 路径同样只投递、不插入额外 NPU 同步点。", first_line=True, align="justify")
    add_paragraph(doc, "warmup 路径必须同步：需要得到真实 free bytes 作为基线。compile_or_warm_up_model 本身已在初始化阶段，同步可接受。", first_line=True, align="justify")

    add_heading_cn(doc, "4.5  日志示例", 2)
    add_code_block(
        doc,
        "[rank0][warmup][npu-mem] allocated=12.31 GiB reserved=18.02 GiB "
        "free=42.10 GiB total=64.00 GiB tag=warmup",
    )
    add_code_block(
        doc,
        "[rank0][forward][npu-mem] allocated=14.02 GiB reserved=20.11 GiB "
        "free=39.90 GiB total=64.00 GiB drop=5.2% warmup_free=42.10 GiB",
    )
    add_code_block(
        doc,
        "WARNING [rank0][forward] NPU free memory dropped 5% vs warmup baseline "
        "(drop=5.2%, free=39.90 GiB, warmup_free=42.10 GiB). "
        "Check PrefixCache/ChunkedPrefill/MTP/SparseFlashMLA overlay and request shape.",
    )
    add_code_block(
        doc,
        "ERROR [rank0][execute_model] request shape anomaly: max_seq_len=131072 > "
        "max_model_len=32768 or num_reqs=33 > max_num_seqs=32",
    )

    # ===== 5 Test =====
    add_heading_cn(doc, "5  测试设计（必要）", 1)
    add_heading_cn(doc, "5.1  单元测试（UT）", 2)
    add_paragraph(doc, "原 MindIE 模板写“不涉及新增函数”。当前分支按 vLLM Ascend 规范，新功能必须有 UT，位于 tests/ut/。", first_line=True, align="justify")
    add_table(
        doc,
        ["用例", "输入", "期望"],
        [
            [
                "test_watch_warmup_sync_records_baseline",
                "warmup=True，mock mem_get_info 返回固定 free",
                "同步打印占用日志；返回值等于 free；不产生 WARNING。",
            ],
            [
                "test_watch_infer_warns_every_5_percent",
                "warmup_mem_free=1000，依次 free=960/900/849",
                "960 不告警（4%）；900 告警一次（10% 跨两级则按阶梯规则更新）；不每调用都刷屏。",
            ],
            [
                "test_watch_does_not_rearm_on_recovery",
                "先降 10% 再回升到 2%",
                "回升不再次告警。",
            ],
            [
                "test_watch_disabled_by_env",
                "VLLM_ASCEND_NPU_MEM_WATCH=0",
                "不调用 mem_get_info，无日志。",
            ],
            [
                "test_check_request_shape_seq_len",
                "max_seq_len > max_model_len",
                "logger.error 被调用，函数不抛异常。",
            ],
            [
                "test_check_request_shape_batch_size",
                "num_reqs > max_num_seqs",
                "logger.error 被调用，函数不抛异常。",
            ],
            [
                "test_check_request_shape_normal",
                "规格均未超限",
                "无 error 日志。",
            ],
        ],
        [Cm(5.4), Cm(5.6), Cm(6.0)],
    )

    add_heading_cn(doc, "5.2  接口测试", 2)
    add_paragraph(doc, "不涉及对外 HTTP / OpenAI API 变更。无新的用户命令行参数（仅环境变量）。Serving 行为不变，只增加日志。", first_line=True, align="justify")

    add_heading_cn(doc, "5.3  业务场景测试", 2)
    add_paragraph(doc, "见第 3 节功能点分解，建议在 NPU 上覆盖：", space_after=6)
    add_table(
        doc,
        ["场景", "建议命令要点", "观察"],
        [
            [
                "默认 warmup + 单请求",
                "vllm serve <model> ；一条 completions",
                "warmup 有占用日志；推理四阶段有占用；无 WARNING（显存平稳）。",
            ],
            [
                "PrefixCache + Chunked Prefill",
                "enable_prefix_caching、长 prompt 触发分块",
                "preprocess/forward 占用升高但仍相对基线 <5% 或按实际告警。",
            ],
            [
                "MTP",
                "speculative_config.method=mtp",
                "sample/postprocess 日志出现；warmup 已含 rejection sampler。",
            ],
            [
                "本分支 SparseFlashMLA",
                "A5、cache_dtype=bfloat16、DeepSeek-V4",
                "forward 标签日志；若未正确 warmup，应出现 5% 告警。",
            ],
        ],
        [Cm(3.8), Cm(6.2), Cm(7.0)],
    )

    add_heading_cn(doc, "5.4  异常场景测试", 2)
    add_paragraph(doc, "1. 特殊用例：在推理过程中人为占用一块空闲显存（UT 中 mock 递减的 mem_get_info，或 NPU 上 torch.empty 泄漏一块 buffer），验证推理过程中是否出现显存打印和显存预警，且按 5% 台阶而非每个 step 告警。", first_line=True, align="justify")
    add_paragraph(doc, "2. 设计 UT：在 execute_model 入口构造不合理的 seq_len 或 batch_size（超过 max_model_len / max_num_seqs），验证触发 ERROR 告警且主路径不抛异常。", first_line=True, align="justify")
    add_paragraph(doc, "3. 开关关闭：VLLM_ASCEND_NPU_MEM_WATCH=0 时，热路径零开销（无线程投递）。", first_line=True, align="justify")
    add_paragraph(doc, "4. 双 Runner：V1 与 V2 各跑一条最小 UT/补丁测试，防止只改 V2 导致 V1 无预警。", first_line=True, align="justify")

    add_heading_cn(doc, "6  风险与约束", 1)
    add_table(
        doc,
        ["风险", "影响", "对策"],
        [
            [
                "mem_get_info 同步导致吞吐下降",
                "AsyncScheduler 卡住，与 AGENTS.md 中 item() 问题同类",
                "推理路径强制异步；提供环境变量关闭。",
            ],
            [
                "异步日志与真实 OOM 时刻存在延迟",
                "OOM 倒序日志可能差 1~N 个 step",
                "日志带 timestamp 与 tag；接受可观测性延迟，不以本功能做硬熔断。",
            ],
            [
                "KV cache 占用被算进 drop_ratio",
                "正常 KV 增长也会告警，产生误报",
                "基线取 warmup 完成后（KV 已分配）的剩余显存；只捕捉 warmup 之后的增量，不把 KV 池本身当泄漏。",
            ],
            [
                "多卡 rank 日志爆炸",
                "TP/DP 每卡都打",
                "每 rank 独立阶梯；必要时后续可加仅 rank0 打印，本 Story 默认每卡都打以便定位不均衡。",
            ],
        ],
        [Cm(4.4), Cm(5.6), Cm(7.0)],
    )

    add_heading_cn(doc, "7  评审检查单（对照 AGENTS.md）", 1)
    add_table(
        doc,
        ["项", "要求", "本 Story"],
        [
            ["环境变量", "集中定义于 vllm_ascend/envs.py，VLLM_ASCEND_* 前缀", "是"],
            ["禁止魔法数", "5% 使用命名常量 / 环境变量", "NPU_MEM_WATCH_STEP_DEFAULT = 0.05"],
            ["禁止热路径 item()", "设备 tensor 不同步取值", "使用 mem_get_info 且推理异步"],
            ["补丁策略", "优先组合/工具模块，不新增模型文件", "新建 utils 模块，不 patch 上游模型"],
            ["双 Runner", "V1 与 V2 行为对等", "两处插入观察点"],
            ["测试", "新功能必须有 UT", "tests/ut/utils/test_npu_memory_watch.py"],
            ["提交信息", "Conventional Commits + sign-off", "feat(npu): add device memory growth warning"],
        ],
        [Cm(3.2), Cm(7.4), Cm(6.4)],
    )

    add_heading_cn(doc, "8  参考", 1)
    add_paragraph(doc, "1. 本说明书模板：MindIE《支持显存异常预警 Story（AR）实现设计说明书》。", first_line=True)
    add_paragraph(doc, "2. 当前分支基线：dsv4-sparse-flash-mla-bf16（3f0d4b4b1 feat(dsv4): add A5 BF16 SparseFlashMla KV path）。", first_line=True)
    add_paragraph(doc, "3. vllm_ascend/worker/worker.py、worker/v2/model_runner.py、worker/model_runner_v1.py。", first_line=True)
    add_paragraph(doc, "4. vllm_ascend/attention/dsa_attn_kv_plan.py、attention/sparse_flash_mla.py。", first_line=True)
    add_paragraph(doc, "5. 仓库 AGENTS.md：环境变量、NPU 同步、Model Runner 变更评审要求。", first_line=True)

    add_paragraph(doc, "", space_after=16)
    add_paragraph(doc, "— 文档结束 —", size=11, bold=True, color=NAVY, align="center")
    return doc


def main():
    out = Path("/workspace/docs/design/Story设计说明书-显存异常预警.docx")
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = build_document()
    doc.save(out)
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
