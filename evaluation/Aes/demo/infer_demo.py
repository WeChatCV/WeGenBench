# -*- coding: utf-8 -*-
"""WeGenBench 美学打分最小 demo（Levelwise Anchor Battle）。

方法概述：
    1. 把「待评图(Input) + 同一等级的 3 张锚点图(Anchor A/B/C)」拼成 2x2 网格，
       让 VLM 判断 Input 相对该等级"水位线"是 below / meets / above。
    2. 从 L3 起步，按投票结果向上(L4->L5)或向下(L2->L1->L0)探测。
    3. 聚合各档结果，得到最终美学等级 L0-L5。

依赖：
    - 一个 OpenAI 兼容的 VLM 服务（如用 vLLM 起的多模态模型），通过 --vllm_host/--port 指定。
    - 锚点集：仓库自带 ../anchors/level_reason.json，但其中 image_path 留空、锚点图不随仓库提供。
      使用前请自备每档锚点图，并把对应的 image_path 填好（相对 --anchor_root 的路径），
      或直接用 --anchor_json/--anchor_root 指向自己的锚点集。
    - Python 包：openai、pillow。

运行示例（已填好锚点图路径后）：
    python infer_demo.py --image path/to/generated.jpg --category human \
        --anchor_root /path/to/your/anchors \
        --vllm_host localhost --port 8000
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import re
from collections import Counter
from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# Prompt 模板（内嵌，便于单文件分发）
# --------------------------------------------------------------------------- #

BASE_PROMPT = """你是一位严格的图像美学评估专家。

你会看到一张 2x2 拼接图：
- 标注为 Input 的格子是输入图，也是本次唯一的评价对象
- 标注为 Anchor A/B/C 的三格是同一等级的锚点图，仅作水位线参考
- 评价前请先看清 Input 在哪一格（下方会告知其网格位置），不要把锚点格的内容误当成 Input

你的任务不是判断输入图分别赢过哪一张锚点图，而是判断输入图相对于这三张锚点图共同代表的 L_level 水位线：
- below：输入图明显低于该等级水位线
- meets：输入图基本达到该等级水位线，允许有轻微优劣差异
- above：输入图明显高于该等级水位线

只判断输入图相对于本次 L_level 水位线的位置，不要预测或声称输入图属于其他等级，不要使用“达到 L4/L5”“甚至更高等级”“商业级”等跨等级推断表述。

整体等级评价原则如下，用于理解 L0-L5 的水位含义：
- L5: "惊艳"：达到商业级标准，美学顶尖、质感完美、无明显技术瑕疵
- L4: "不错"：美学优秀、氛围/冲击力/艺术感强，允许非核心主体有极微小瑕疵
- L3: "合格"：美学中规中矩，或美学不错但核心主体有轻微 AI 感/轻微逻辑瑕疵
- L2: "略差"：美学较差，或核心主体 AI 痕迹明显/细节瑕疵较多
- L1: "差图"：美学较差且存在明显畸形、结构瑕疵、疑似文字乱码或局部逻辑崩坏
- L0: "废图"：明显文字错误、极度畸形、破坏性穿模、结构崩坏、严重逻辑错误、色彩坍塌、美学极差或引发生理不适

【评分原则】：
- 网格位置无关：Input 出现在 2x2 网格的任意位置都不影响质量判断。你的判断必须只基于图片内容本身，不得因为 Input 在左上、右下或其他位置而倾向给高分或低分。
- 大处着眼：不要像素级找错。只有正常手机/电脑屏幕尺寸下 5 秒内能明显感知的问题才算硬伤。
- 风格豁免：超现实、插画、意境化表达可以接受非常规形态，但如果导致主体结构崩坏、脏感、生理不适，仍然算硬伤。
- 文字严查：如果画面有核心文字，必须逐字判断是否乱码、错字、偏旁混乱或不可识别。
- 结构严于材质：结构正确是美学的底座。纹理精美但文字乱码或人物/动物/物体畸形，严禁因"氛围感强"给高评价。
- 风格公平：输入图和锚点图可能属于不同风格（写实摄影、插画、3D 渲染、水彩等）。评判质量时，必须在输入图自身的风格标准内评价，不能因为输入图不够真实而扣分。一张顶级插画和一张顶级照片可以处于同一美学等级。只比较：结构正确性、质感、光影水平、构图水平、色彩协调度、风格完成度、视觉冲击力。不比较：哪张更像真实照片。

请独立观察输入图自身的全部优缺点，尤其注意结构、文字、手部/肢体、物理逻辑、材质质感、光影、构图和完成度。

请同时判断输入图在 6 个维度上相对于该等级水位线的位置：
- structure：人体/动物/物体结构、手部/肢体、文字、透视、物理逻辑等硬伤
- texture：材质质感、细节纹理、AI 涂抹感、塑料感、锐化/糊化问题
- lighting：光影方向、明暗层次、曝光、高光和阴影逻辑
- color：色彩平衡、肤色/物体颜色自然度、色彩和谐和高级感
- composition：主体突出、构图稳定、视觉重心、空间关系
- completeness：整体完成度、风格统一性、主体与背景协调、细节收尾

若输入图第一眼观感不错但细节瑕疵明显，请如实判断为 meets 或 below，不要只根据整体氛围给 above。

【低等级硬伤速查】
判断任意等级时，都必须同时检查以下低等级硬伤；如果命中，不得因为输入图整体观感、色彩、光影、构图或局部质感更好而判 above：
- L0 废图级硬伤：核心文字明显错误/系统性乱码/不可读；核心主体不可辨认；极度畸形；破坏性穿模；结构崩坏；严重物理/空间逻辑错误；画面严重坍塌或引发生理不适。
- L1 差图级硬伤：明显畸形、缺肢/多肢、手脚/爪/翅膀/尾巴错接或缺失、身体连接崩坏、局部穿模、疑似文字乱码、主体局部融化/塌陷、局部逻辑崩坏。
- L2 略差级硬伤：核心主体 AI 痕迹明显、细节瑕疵较多、关键物体结构碎裂、明显塑料/蜡感/涂抹感、构图/光线/色彩显著拖累可用性。

如果发现低等级硬伤，必须遵守输出约束中的【硬伤一致性规则】，确保 hard_defects、dimension_decisions、decision 和 blocking_defects 互相一致。"""

ANCHOR_USAGE = """【锚点图使用说明】
- 三张锚点图共同代表本次等级的审美水位线，不是输入图的评分清单；判断输入图整体是低于、达到还是高于这组锚点共同代表的水位线，不要逐张判输赢。
- 你必须独立审视 Input 自身，在 6 个维度上判断它相对该水位线的位置，不要逐张比较 Input 是否赢过 Anchor A/B/C，也不要围绕锚点理由逐项找相同问题。
- Input 可能存在锚点图没有的新缺陷，也可能避开锚点图的弱项，都要独立计入判断。
- 瑕疵等价原则：若 Input 避开了锚点的某个缺点，但引入破坏力相当的新缺点，不视为 above。
- reasoning、key_strengths、blocking_defects 只能描述 Input 自身，不得写“优于锚点图”“缺乏锚点图那种”等相对锚点表述。"""

OUTPUT_SCHEMA = """【输出约束】
禁止废话，仅输出一个标准 JSON 块，字段说明如下：
- input_subject 先用一句话（10 字以内）复述 Input 格里画的主体内容，例如"街头遛狗的人和狗"。只描述 Input 格，不要把锚点格里的物体/动物写进来；后续所有判定都必须围绕这个主体。
- dimension_decisions 必须包含 6 个维度，每个维度的值必须是 "below"、"meets"、"above" 之一，表示输入图在该维度上相对于本次等级水位线的位置。
- decision 必须是 "below"、"meets"、"above" 之一。
- reasoning 用一句话说明本次 decision 的决定性依据，20 字以内，不要提及是否达到其他等级。
- key_strengths 列出输入图主要优势，最多 3 条，每条 12 字以内。
- blocking_defects 列出限制输入图达到/超过本次等级水位线的关键问题，最多 3 条，每条 12 字以内；没有则为空数组。
- hard_defects 标记是否存在低等级硬伤：
  - has_hard_defect 为 true/false。
  - types 只能从 "text"、"structure"、"physical_logic"、"completeness" 中选择；没有硬伤则为空数组。
  - severity 只能是 "none"、"minor"、"major"、"fatal"。
- 用词简洁，不写长句，不展开解释。

【硬伤一致性规则】
- 如果 blocking_defects 写到核心文字乱码/不可读、主体结构崩坏、肢体缺失/错接、严重穿模、严重物理或空间逻辑错误，则 hard_defects.has_hard_defect 必须为 true。
- structure 维度始终相对【本次等级水位线】判定，不是绝对的"有没有硬伤"：
  - 在 L3 及以上探测中，本档要求结构干净：若 types 含 text、structure 或 physical_logic，dimension_decisions.structure 必须为 below。
  - 在 L2 及以下探测中，结构硬伤本就是这些等级的常态（L1/L2 的典型画面就带畸形、瑕疵），应按"硬伤是否比本档锚点更差"判定 structure：更差才 below，与锚点相当即 meets，不要因为存在硬伤就一律判 below。
- 如果 hard_defects.severity 是 major 或 fatal，decision 不能是 above；在 L3/L2 探测中通常应为 below。
- 在 L3 及以上探测中，不要一边在 blocking_defects 写严重硬伤、一边把 structure 判为 meets/above。

{
  "input_subject": "10字以内，只复述Input格主体",
  "dimension_decisions": {
    "structure": "below|meets|above",
    "texture": "below|meets|above",
    "lighting": "below|meets|above",
    "color": "below|meets|above",
    "composition": "below|meets|above",
    "completeness": "below|meets|above"
  },
  "decision": "below|meets|above",
  "hard_defects": {
    "has_hard_defect": false,
    "types": ["text"],
    "severity": "none|minor|major|fatal"
  },
  "reasoning": "20字以内，只解释本次decision",
  "key_strengths": ["12字以内"],
  "blocking_defects": ["12字以内"]
}"""

LEVEL_FOCUS = {
    0: """【L0 水位判断重点】

below L0：
- 输入图比 L0 水位线还更严重崩坏、几乎不可辨认或完全不可用时判 below。

meets L0：
- 输入图存在废图级硬伤：主体结构崩坏、肢体数量严重错误、毁灭性穿模、核心文字系统性乱码、物理逻辑根本错误、色彩/画面严重坍塌。
- 只要核心主体不可用或第一眼就有破坏性错误，即使局部氛围尚可，也可以判为 meets L0。

above L0：
- 输入图虽然质量差，但主体仍基本可辨认，结构没有完全崩坏。
- 有明显 AI 痕迹、局部畸形或审美很弱，但还没到废图级不可用。
- 若命中 base 中的 L0 废图级硬伤，应 meets/below L0，而不是 above。

典型 meets L0：核心主体崩坏或系统性乱码、第一眼即废、基本不可用的图。

边界处理：
- above 与 meets 拿不准、且核心主体已不可用或第一眼有破坏性错误 → 默认 meets L0。
- below L0 仅在比废图水位更彻底崩坏时使用。
- 若 structure 维度为 below（主体结构已崩坏），不应判 above L0。

特别注意：
- 人、动物、鸟类、神兽、拟人角色必须检查头、身体、腿、脚/爪、翅膀、尾巴等关键结构是否缺失、重复、错接或融合。""",
    1: """【L1 水位判断重点】

below L1：
- 核心主体结构严重不可用，接近废图：明显缺肢、多肢、身体连接崩坏、穿模严重、文字大面积乱码、主体局部融化或塌陷。
- 动物/鸟类/神兽如果缺腿、缺脚/爪、翅膀错接、肢体数量明显错误，通常应低于 L1 或接近 L0。

meets L1：
- 输入图有明显但非彻底毁灭的结构或质感问题：局部畸形、手指/爪子粘连、肢体连接生硬、物体结构错位、局部文字错误。
- 整体仍能看出主体和基本意图，但质量明显差，不适合作为合格图。

above L1：
- 明显好过差图水位，主体基本完整，致命结构问题不明显。
- 仍可能有较重 AI 感、涂抹感、构图差或局部逻辑问题，但不应接近废图。
- 若命中 base 中的 L0/L1 硬伤，应 meets/below L1，而不是 above。

典型 meets L1：主体和意图还能看出，但有明显畸形/粘连/局部文字错误、质量明显差的图。

边界处理：
- above 与 meets 拿不准 → 默认 meets，不因色彩或氛围拔高。
- meets 与 below 拿不准、且核心主体有结构/文字/物理硬伤 → 默认 below。
- 判 above 时 structure 维度不应为 below；若 structure 为 below，本档最高只能 meets。

特别注意：
- 不要因为色彩或氛围不错就忽略核心主体缺肢、错肢、穿模、文字错误等硬伤。""",
    2: """【L2 水位判断重点】

below L2：
- 已经不只是“略差”，而是存在明显结构硬伤或局部崩坏：缺腿/缺爪/缺手指、肢体数量错误、关键物体结构碎裂、明显穿模、核心文字不可读。
- 审美很弱且伴随结构/物理逻辑问题，应低于 L2，继续下探 L1。

meets L2：
- 输入图没有废图级致命错误，但 AI 痕迹重或审美明显不足：塑料感、油腻感、涂抹感、过度锐化、纹理虚假平滑。
- 构图失衡、主体被截断、背景杂乱、光线破碎、色调发灰，整体低于合格素材。
- 可以有较多细节瑕疵，但核心主体仍基本可辨认、结构大体成立。

above L2：
- 输入图明显脱离略差水位，主体完整，结构和物理逻辑基本可靠。
- 仍可能有轻微 AI 感或普通审美，但不应有第一眼可见的核心结构硬伤。
- 若命中 base 中的 L0-L2 硬伤，应 meets/below L2，而不是 above。

典型 meets L2：无致命硬伤、但 AI 痕迹重、塑料/涂抹感明显、构图或色调拖后腿、低于合格素材的图。

边界处理：
- above 与 meets 拿不准 → 默认 meets，不因质感或色彩尚可而拔高。
- meets 与 below 拿不准、且核心主体有结构/文字/物理硬伤 → 默认 below。
- 判 above 时 structure 维度不应为 below；若 structure 为 below，本档最高只能 meets。

特别注意：
- 核心主体缺腿、缺爪、翅膀/尾巴错接、身体连接不合理，不应因为质感或色彩尚可而判 above L2。""",
    3: """【L3 水位判断重点】

below L3：
- 核心主体存在明显结构硬伤：缺腿、缺脚/爪、缺手指、肢体数量错误、翅膀/尾巴错接、身体连接不合理、明显穿模或物理逻辑崩坏。
- 核心文字乱码、错字、不可识别，或主体局部融化/塌陷。
- 这些问题即使画面色彩、光影、氛围不错，也必须判 below L3。

meets L3：
- 整体观感合格，没有第一眼破坏性的结构、文字或物理逻辑硬伤。
- 允许轻微 AI 感、轻微涂抹、局部细节不精致、背景略弱、构图中规中矩。
- 核心主体结构基本完整，肢体/附肢数量和连接关系合理。

above L3：
- 不仅没有明显硬伤，而且在质感、光影、构图、色彩或完成度上明显高于合格线。
- 画面具备较强观感或专业完成度，不能只是“无明显错误”。
- 若命中低等级硬伤或本节 below L3 硬伤，应 below L3，而不是 above。

典型 meets L3：观感合格、无第一眼硬伤、但细节平庸、没有突出亮点的普通成片。

边界处理：
- above 与 meets 拿不准 → 默认 meets，不因整体氛围拔高。
- meets 与 below 拿不准、且核心主体有结构/文字/物理硬伤 → 默认 below。
- 判 above 时 structure 维度不应为 below；若 structure 为 below，本档最高只能 meets。

特别注意：
- 动物、鸟类、神兽、拟人角色不要只看羽毛/皮毛/氛围，必须先检查骨架、腿、脚/爪、翅膀、尾巴等关键结构。""",
    4: """【L4 水位判断重点】

below L4：
- 输入图虽可能合格或好看，但达不到优秀成片水位：光影普通、构图常规、材质不够细腻、完成度不足、细节 AI 感明显。
- 核心主体有结构、文字、物理逻辑硬伤时，必须 below L4，通常还应触发结构下探。

meets L4：
- 达到优秀成片水平：主体突出，构图专业，光影有氛围，材质细腻，色彩协调，完成度高。
- 允许非核心区域有极轻微 AI 简化，但核心主体必须结构闭环、文字清晰正确。

above L4：
- 明显超过“不错”水位，具备强视觉冲击力、独特审美、精致材质和高级完成度，接近顶级商业/艺术水准。
- 不能仅因为无硬伤就判 above，必须有明显超出 L4 的亮点。
- 若命中 base 中的 L0-L4 硬伤，应 below L4，而不是 meets/above。

典型 meets L4：构图光影专业、核心主体结构闭环、可直接使用的优秀成片，但还不到惊艳。

边界处理：
- above 与 meets 拿不准 → 默认 meets；above 需要明显超出本档的亮点，不能只是无硬伤。
- meets 与 below 拿不准、且核心主体有结构/文字/物理硬伤 → 默认 below。
- 判 above 时 structure 维度不应为 below；若 structure 为 below，本档最高只能 meets。

特别注意：
- L4 对核心主体结构要求很高；人物/动物/鸟类缺肢、缺爪、肢体错接、文字乱码都不应 meets L4。""",
    5: """【L5 水位判断重点】

below L5：
- 只要存在明显 AI 感、核心主体轻微结构问题、文字不完美、材质不够极致、构图不够惊艳或视觉冲击力不足，都应 below L5。
- L5 不接受核心主体结构、文字、物理逻辑上的任何明显瑕疵。

meets L5：
- 达到顶尖商业级/艺术级水准：第一眼震撼，光影叙事强，材质极致，色彩高级，构图大师级，完成度非常高。
- 核心主体无可感知 AI 硬伤，文字如有必须完全正确。

above L5：
- 通常很少使用。只有输入图明显超越三张 L5 锚点共同水位，具备极强艺术/商业价值且几乎零瑕疵时才判 above。
- 若命中 base 中的任意低等级硬伤或存在任何可感知瑕疵，应 below L5。

典型 meets L5：第一眼震撼、美学/结构/质感/光影/构图/完成度同时顶级、核心主体零可感知瑕疵的商业级成片。

边界处理：
- meets 与 below 拿不准 → 默认 below；只要有任何可感知瑕疵或轻微 AI 感即 below L5。
- above L5 极少使用，仅在明显超越三张 L5 锚点共同水位时才判。
- 判 above/meets 时 structure 维度不应为 below；若 structure 为 below，应判 below L5。

特别注意：
- 不要因为单一维度优秀就判 L5；L5 要求美学、结构、质感、光影、构图和完成度同时达到顶级。""",
}

# --------------------------------------------------------------------------- #
# 锚点加载与分组
# --------------------------------------------------------------------------- #

_FILENAME_CATEGORY_MAP = {
    "human": "human", "animal": "animal", "food": "food", "object": "object",
    "building": "building", "indoor": "indoor", "natural": "natural",
    "plant": "plant", "flower": "plant", "logo": "logo", "vehicle": "vehicle",
}

CATEGORY_COMPAT = {
    "natural": {"natural", "plant", "object"},
    "plant": {"plant", "natural", "object"},
    "logo": {"logo", "object"},
    "vehicle": {"vehicle", "object"},
    "indoor": {"indoor", "building"},
    "building": {"building", "indoor"},
    "human": {"human"},
    "animal": {"animal"},
    "food": {"food"},
    "object": {"object"},
}


@dataclass
class Anchor:
    level: int
    image_path: str
    why_this_level: str
    category: str = "object"


def _infer_category(filename: str) -> str:
    stem = os.path.splitext(filename)[0]
    for part in re.split(r"[_\-]", stem):
        if part.lower() in _FILENAME_CATEGORY_MAP:
            return _FILENAME_CATEGORY_MAP[part.lower()]
    return "object"


def load_anchors(json_path: str, anchor_root: str | None = None) -> list[Anchor]:
    """从 level_reason.json 读取锚点。

    JSON 形如 {"level-5": [{"image_path": "...", "why_this_level": "..."}, ...], ...}。
    """
    if anchor_root is None:
        anchor_root = os.path.dirname(os.path.abspath(json_path))
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    anchors: list[Anchor] = []
    for level_key, items in data.items():
        match = re.search(r"(\d+)", level_key)
        if match is None:
            continue
        level = int(match.group(1))
        for item in items:
            raw_path = item.get("image_path", "")
            abs_path = raw_path if os.path.isabs(raw_path) else os.path.join(anchor_root, raw_path)
            anchors.append(Anchor(
                level=level,
                image_path=abs_path,
                why_this_level=item.get("why_this_level", ""),
                category=_infer_category(os.path.basename(raw_path)),
            ))
    anchors.sort(key=lambda a: (a.level, a.category))
    return anchors


def _stable_hash(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)


def _rotate(items: list, seed_key: str) -> list:
    if not items:
        return []
    start = _stable_hash(seed_key) % len(items)
    return items[start:] + items[:start]


def select_anchor_group(anchors, category, level, group_size=3, seed_key=""):
    """挑选同一等级的紧凑锚点组：优先同类别，其次兼容类别，最后任意类别。"""
    compat_set = CATEGORY_COMPAT.get(category, {category, "object"})
    same = [a for a in anchors if a.level == level and a.category == category]
    compat = [a for a in anchors if a.level == level and a.category in compat_set and a.category != category]
    fallback = [a for a in anchors if a.level == level and a.category not in {category, *compat_set}]

    picked: list[Anchor] = []
    for pool_name, pool in (("same", same), ("compat", compat), ("fallback", fallback)):
        for anchor in _rotate(pool, f"{seed_key}:{level}:{pool_name}"):
            if anchor not in picked:
                picked.append(anchor)
            if len(picked) >= group_size:
                return picked
    return picked


def build_prompt(probe_level: int, anchor_reasons: list[str]) -> str:
    reasons = [f"锚点{label}：{reason}" for label, reason in zip("ABC", anchor_reasons) if reason]
    reason_block = "\n".join(reasons) if reasons else "无"
    return "\n\n".join([
        BASE_PROMPT.replace("L_level", f"L{probe_level}"),
        f"本次探测等级：L{probe_level}",
        LEVEL_FOCUS[probe_level],
        ANCHOR_USAGE,
        f"本次三张锚点图被定级到该等级的原因（仅帮助理解水位线，不作为输入图评分清单）：\n{reason_block}",
        OUTPUT_SCHEMA,
    ])


# --------------------------------------------------------------------------- #
# 2x2 拼图 + 位置变体
# --------------------------------------------------------------------------- #

_GAP_PX = 16
_LABEL_H = 34
_OUTER_MARGIN = _LABEL_H
_GRID_CELL_NAMES = ("左上", "右上", "左下", "右下")


def _get_font(size: int = 22):
    from PIL import ImageFont

    for path in (
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    ):
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _resize_to_area(image, target_area: float, max_long_edge: int):
    from PIL import Image

    image = image.convert("RGB")
    scale = (target_area / (image.width * image.height)) ** 0.5
    scale = min(scale, max_long_edge / max(image.width, image.height))
    new_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(new_size, Image.LANCZOS)


def stitch_grid_2x2(image_paths: list[str], labels: list[str], max_long_edge: int = 1536):
    """把 4 张图拼成带标签的 2x2 网格，长边受限。"""
    from PIL import Image, ImageDraw

    if len(image_paths) != 4 or len(labels) != 4:
        raise ValueError("stitch_grid_2x2 需要正好 4 张图与 4 个标签")

    image_long_edge = max(128, (max_long_edge - _GAP_PX - 2 * _LABEL_H - _OUTER_MARGIN) // 2)
    originals = [Image.open(p).convert("RGB") for p in image_paths]

    def max_area_for(image):
        scale = image_long_edge / max(image.width, image.height)
        return image.width * image.height * scale * scale

    target_area = min(max_area_for(image) for image in originals)
    images = [_resize_to_area(image, target_area, image_long_edge) for image in originals]
    col_widths = [max(images[0].width, images[2].width), max(images[1].width, images[3].width)]
    row_heights = [max(images[0].height, images[1].height), max(images[2].height, images[3].height)]
    row_block_heights = [_LABEL_H + h for h in row_heights]
    canvas_w = sum(col_widths) + _GAP_PX + 2 * _OUTER_MARGIN
    canvas_h = sum(row_block_heights) + _GAP_PX + _OUTER_MARGIN
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = _get_font()

    row_tops = [_OUTER_MARGIN, _OUTER_MARGIN + row_block_heights[0] + _GAP_PX]
    for idx, (image, label) in enumerate(zip(images, labels)):
        row, col = divmod(idx, 2)
        x0 = _OUTER_MARGIN + sum(col_widths[:col]) + col * _GAP_PX
        ix = x0 + (col_widths[col] - image.width) // 2
        iy = row_tops[row] + _LABEL_H + (row_heights[row] - image.height) // 2
        canvas.paste(image, (ix, iy))
        label_y = iy - _LABEL_H
        draw.rectangle([(ix, label_y), (ix + image.width, label_y + _LABEL_H)], fill=(255, 255, 255))
        bbox = draw.textbbox((0, 0), label, font=font)
        tx = ix + (image.width - (bbox[2] - bbox[0])) // 2
        ty = label_y + (_LABEL_H - (bbox[3] - bbox[1])) // 2
        draw.text((tx, ty), label, fill=(64, 64, 64), font=font)

    if max(canvas.size) > max_long_edge:
        scale = max_long_edge / max(canvas.size)
        canvas = canvas.resize((round(canvas.width * scale), round(canvas.height * scale)), Image.LANCZOS)
    return canvas


def _position_variant(image_path, anchor_paths, variant):
    a = ["Anchor A", "Anchor B", "Anchor C"]
    layouts = [
        ([image_path, anchor_paths[0], anchor_paths[1], anchor_paths[2]], ["Input", *a]),
        ([anchor_paths[1], anchor_paths[2], anchor_paths[0], image_path], [a[1], a[2], a[0], "Input"]),
        ([anchor_paths[0], image_path, anchor_paths[2], anchor_paths[1]], [a[0], "Input", a[2], a[1]]),
        ([anchor_paths[2], anchor_paths[0], image_path, anchor_paths[1]], [a[2], a[0], "Input", a[1]]),
    ]
    return layouts[variant % len(layouts)]


def _position_instruction(labels: list[str]) -> str:
    cell = _GRID_CELL_NAMES[labels.index("Input")] if "Input" in labels else _GRID_CELL_NAMES[0]
    return (
        f"【定位·最重要】本次待评对象只有 Input 图，它位于 2×2 网格的【{cell}】格"
        f"（标签为“Input”）。Anchor A/B/C 是本档水位线参考，请把 Input 与这条水位线比较来判断高低；"
        f"但锚点里出现的具体主体/物体并不属于 Input——描述 input_subject / reasoning / "
        f"key_strengths / blocking_defects 时只能写 Input 格里真实存在的内容，不要混入锚点格的东西。"
    )


# --------------------------------------------------------------------------- #
# VLM 调用 + 解析
# --------------------------------------------------------------------------- #

def _image_to_b64(image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def call_vlm(client, model, grid_image, prompt_text, temperature=0.0, top_p=0.8):
    b64 = _image_to_b64(grid_image)
    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": prompt_text},
            ],
        }],
        temperature=temperature,
        top_p=top_p,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return response.choices[0].message.content


VALID_DECISIONS = {"below", "meets", "above"}
_DIMENSIONS = ("structure", "texture", "lighting", "color", "composition", "completeness")


def _extract_json(text: str) -> str:
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start != -1 and end > start else text


def parse_response(raw: str) -> dict | None:
    """解析单次探测响应，返回 {decision, dimension_decisions, ...}；非法则返回 None。"""
    try:
        data = json.loads(_extract_json(raw))
    except (json.JSONDecodeError, ValueError):
        return None
    decision = data.get("decision")
    dims = data.get("dimension_decisions")
    if decision not in VALID_DECISIONS or not isinstance(dims, dict):
        return None
    if not all(dims.get(d) in VALID_DECISIONS for d in _DIMENSIONS):
        return None
    hard = data.get("hard_defects") if isinstance(data.get("hard_defects"), dict) else {}
    return {
        "decision": decision,
        "dimension_decisions": {d: dims[d] for d in _DIMENSIONS},
        "input_subject": str(data.get("input_subject", "")),
        "reasoning": str(data.get("reasoning", "")),
        "hard_defects": {
            "has_hard_defect": bool(hard.get("has_hard_defect", False)),
            "types": [str(t) for t in hard.get("types", []) if t],
            "severity": hard.get("severity", "none"),
        },
    }


def _majority(decisions: list[str]) -> str:
    if not decisions:
        return "meets"
    votes = Counter(decisions)
    top = max(votes.values())
    cands = [d for d, c in votes.items() if c == top]
    if len(cands) == 1:
        return cands[0]
    score = {"below": -1, "meets": 0, "above": 1}
    avg = sum(score[d] for d in decisions) / len(decisions)
    return "above" if avg > 0 else "below" if avg < 0 else "meets"


# --------------------------------------------------------------------------- #
# 单档探测 + 路由 + 最终聚合
# --------------------------------------------------------------------------- #

def run_probe(client, model, image_path, anchors, category, level, seed_key, args):
    """对某一等级跑多个位置变体，聚合出该档 decision / 维度 decision。"""
    group = select_anchor_group(anchors, category, level, group_size=3, seed_key=seed_key)
    if len(group) < 3:
        raise RuntimeError(f"L{level} category={category} 锚点不足 3 张：{len(group)}")
    prompt = build_prompt(level, [a.why_this_level for a in group])
    anchor_paths = [a.image_path for a in group]

    parsed_list = []
    for variant in range(args.variants):
        paths, labels = _position_variant(image_path, anchor_paths, variant)
        grid = stitch_grid_2x2(paths, labels, max_long_edge=args.max_long_edge)
        prompt_text = f"{_position_instruction(labels)}\n\n{prompt}"
        try:
            raw = call_vlm(client, model, grid, prompt_text, temperature=args.temperature)
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] L{level} variant {variant} 调用失败: {e}")
            continue
        parsed = parse_response(raw)
        if parsed is not None:
            parsed_list.append(parsed)
        else:
            print(f"  [warn] L{level} variant {variant} 输出解析失败")

    if not parsed_list:
        return {"probe_level": level, "final_decision": "meets",
                "votes": {}, "structure": "meets"}

    decisions = [p["decision"] for p in parsed_list]
    structure_decisions = [p["dimension_decisions"]["structure"] for p in parsed_list]
    return {
        "probe_level": level,
        "final_decision": _majority(decisions),
        "votes": dict(Counter(decisions)),
        "structure": _majority(structure_decisions),
    }


def _ceiling_band(level: int) -> str:
    if level <= 0:
        return "L0"
    if level >= 5:
        return "L5"
    return f"L{level}/L{level + 1}"


def _has_vote_disagreement(probe: dict) -> bool:
    return sum(1 for count in probe["votes"].values() if count > 0) > 1


def _path_clip_bounds(probes: list[dict]) -> tuple[int, int]:
    levels = [max(0, min(5, p["probe_level"])) for p in probes]
    if not levels:
        return 0, 5
    return min(levels), max(levels)


def _candidate_levels(probe_level: int, decision: str, clip_min: int, clip_max: int) -> range:
    if decision == "below":
        start, stop = 0, max(0, probe_level)
    elif decision == "above":
        start, stop = min(5, probe_level + 1), 6
    else:
        level = max(0, min(5, probe_level))
        start, stop = level, level + 1
    start = max(start, clip_min)
    stop = min(stop, clip_max + 1)
    if start >= stop:
        boundary = max(clip_min, min(clip_max, probe_level))
        return range(boundary, boundary + 1)
    return range(start, stop)


def _constraint_majority_level(probes: list[dict]) -> tuple[int, str] | None:
    """投票分歧时的边界精修：把每档每个 decision 的票数摊到其候选等级区间上累计，
    取得票最高的等级（多个并列时给出区间）。无任何分歧时返回 None，回退决策树。
    """
    if not any(_has_vote_disagreement(probe) for probe in probes):
        return None

    votes = Counter()
    clip_min, clip_max = _path_clip_bounds(probes)
    for probe in probes:
        for decision, count in probe["votes"].items():
            if count <= 0:
                continue
            for level in _candidate_levels(probe["probe_level"], decision, clip_min, clip_max):
                votes[level] += count

    if not votes:
        return None

    top_count = max(votes.values())
    top_levels = sorted(level for level, count in votes.items() if count == top_count)
    final_level = top_levels[0]
    if len(top_levels) > 1:
        return final_level, f"L{top_levels[0]}/L{top_levels[-1]}"
    return final_level, f"L{final_level}"


def aggregate_levelwise(probes: list[dict]) -> dict:
    """把各档探测聚合成最终等级 / 区间，并应用结构天花板。"""
    by_level = {p["probe_level"]: p for p in probes}

    structural_ceiling = 5
    for p in probes:
        if p["probe_level"] >= 3 and p["structure"] == "below":
            structural_ceiling = min(structural_ceiling, max(0, p["probe_level"] - 1))

    if 3 not in by_level:
        return {
            "final_level": min(3, structural_ceiling),
            "level_band": "L3" if structural_ceiling >= 3 else _ceiling_band(structural_ceiling),
            "structural_ceiling": structural_ceiling,
            "route": [p["probe_level"] for p in probes],
            "probes": probes,
        }

    constraint_level = _constraint_majority_level(probes)
    if constraint_level is not None:
        final_level, band = constraint_level
        if final_level > structural_ceiling:
            final_level = structural_ceiling
            band = _ceiling_band(structural_ceiling)
        return {
            "final_level": final_level,
            "level_band": band,
            "structural_ceiling": structural_ceiling,
            "route": [p["probe_level"] for p in probes],
            "probes": probes,
        }

    def has_two_above(level):
        p = by_level.get(level)
        return bool(p and p["votes"].get("above", 0) >= 2)

    l3 = by_level[3]["final_decision"]
    if l3 == "below":
        if 2 not in by_level or by_level[2]["final_decision"] == "meets":
            final_level, band = 2, "L2"
        elif by_level[2]["final_decision"] == "above":
            final_level, band = 2, "L2/L3"
        else:  # below
            if 1 not in by_level or by_level[1]["final_decision"] == "above":
                final_level, band = 1, "L1/L2"
            elif by_level[1]["final_decision"] == "meets":
                final_level, band = 1, "L1"
            else:  # below
                if 0 not in by_level or by_level[0]["final_decision"] == "above":
                    final_level, band = 0, "L0/L1"
                else:
                    final_level, band = 0, "L0"
    elif l3 == "meets":
        if 4 not in by_level:
            final_level, band = 3, "L3"
        elif has_two_above(4):
            if 5 in by_level and by_level[5]["final_decision"] in {"meets", "above"}:
                final_level, band = 5, "L5"
            else:
                final_level, band = 4, "L4/L5"
        elif by_level[4]["final_decision"] in {"meets", "above"}:
            final_level, band = 4, "L4"
        else:
            final_level, band = 3, "L3/L4"
    else:  # above
        if 4 not in by_level:
            final_level, band = 4, "L3/L4"
        elif has_two_above(4):
            if 5 in by_level and by_level[5]["final_decision"] in {"meets", "above"}:
                final_level, band = 5, "L5"
            else:
                final_level, band = 4, "L4/L5"
        elif by_level[4]["final_decision"] in {"meets", "above"}:
            final_level, band = 4, "L4"
        else:
            final_level, band = 3, "L3/L4"

    if final_level > structural_ceiling:
        final_level = structural_ceiling
        band = _ceiling_band(structural_ceiling)

    return {
        "final_level": final_level,
        "level_band": band,
        "structural_ceiling": structural_ceiling,
        "route": [p["probe_level"] for p in probes],
        "probes": probes,
    }


def score_image(client, model, image_path, anchors, category, seed_key, args) -> dict:
    """对单张图执行 L3 起步的 levelwise 探测路由。"""
    probes = []
    l3 = run_probe(client, model, image_path, anchors, category, 3, seed_key, args)
    probes.append(l3)

    should_down = l3["votes"].get("below", 0) >= 1
    should_up = l3["votes"].get("above", 0) >= 2

    if should_down:
        for level in (2, 1, 0):
            probe = run_probe(client, model, image_path, anchors, category, level,
                              f"{seed_key}:{level}", args)
            probes.append(probe)
            if probe["votes"].get("below", 0) < 1:
                break
    if should_up:
        l4 = run_probe(client, model, image_path, anchors, category, 4, f"{seed_key}:4", args)
        probes.append(l4)
        if l4["votes"].get("above", 0) >= 2:
            l5 = run_probe(client, model, image_path, anchors, category, 5, f"{seed_key}:5", args)
            probes.append(l5)

    return aggregate_levelwise(probes)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    default_anchor_root = os.path.normpath(os.path.join(here, "..", "anchors"))
    default_anchor_json = os.path.join(default_anchor_root, "level_reason.json")
    parser = argparse.ArgumentParser(description="WeGenBench 美学打分最小 demo（levelwise anchor battle）")
    parser.add_argument("--image", required=True, help="待评测的生成图路径")
    parser.add_argument("--category", default="object",
                        help="主体类别：human/animal/food/object/building/indoor/natural/plant/logo/vehicle")
    parser.add_argument("--anchor_json", default=os.environ.get("ANCHOR_JSON", default_anchor_json),
                        help="锚点 level_reason.json 路径（默认用仓库自带 ../anchors/level_reason.json）")
    parser.add_argument("--anchor_root", default=os.environ.get("ANCHOR_ROOT"),
                        help="锚点图根目录，用于解析相对 image_path（默认取 anchor_json 同级目录）")
    parser.add_argument("--vllm_host", default="localhost", help="VLM 服务 host")
    parser.add_argument("--port", default="8000", help="VLM 服务端口")
    parser.add_argument("--model", default=None, help="VLM 模型名（默认取服务返回的第一个）")
    parser.add_argument("--variants", type=int, default=3, help="每档跑几个位置变体并投票")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_long_edge", type=int, default=1536, help="拼图长边上限")
    parser.add_argument("--id", default=None, help="样本 id（仅作为锚点选择的随机种子，可选）")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.anchor_json:
        raise SystemExit("请通过 --anchor_json 或环境变量 ANCHOR_JSON 指定锚点 level_reason.json")
    if not os.path.exists(args.image):
        raise SystemExit(f"图片不存在: {args.image}")

    from openai import OpenAI

    anchors = load_anchors(args.anchor_json, args.anchor_root)
    usable = [a for a in anchors if a.image_path and os.path.isfile(a.image_path)]
    if not usable:
        raise SystemExit(
            "未找到可用锚点图：仓库自带的 level_reason.json 中 image_path 为空、锚点图不随仓库提供。\n"
            "请先自备每档锚点图并把 image_path 填好（相对 --anchor_root），"
            "或用 --anchor_json/--anchor_root 指向自己的锚点集。")
    print(f"已加载 {len(anchors)} 条锚点记录，其中 {len(usable)} 张锚点图可用")

    base_url = f"http://{args.vllm_host}:{args.port}/v1"
    client = OpenAI(api_key="EMPTY", base_url=base_url)
    model = args.model or client.models.list().data[0].id
    print(f"VLM: {base_url} model={model}")

    seed_key = args.id or args.image
    result = score_image(client, model, args.image, anchors, args.category, seed_key, args)

    print("\n========== 美学打分结果 ==========")
    print(f"最终等级: L{result['final_level']}  (区间 {result['level_band']})")
    print(f"结构天花板: L{result['structural_ceiling']}")
    print(f"探测路径: {' -> '.join('L%d' % lv for lv in result['route'])}")
    print("各档投票:")
    for p in result["probes"]:
        print(f"  L{p['probe_level']}: decision={p['final_decision']:5s} "
              f"structure={p['structure']:5s} votes={p['votes']}")
    print("\n完整结果(JSON):")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
