#!/usr/bin/env python3
# ============================================================================
# infer_demo.py
#
# WeGenBench 一致性 CoT 打分器的极简单图推理 demo。
#
# 这是从完整评测链路 (run_msswift_consistency_score.sh ->
# pipe_msswift_consistency.py) 中剥离出来的最小可运行版本：去掉了多卡分片、
# 断点续跑、批量合并等工程逻辑，只保留与全量链路 **完全一致** 的 PROMPT、
# 推理调用与解析正则，因此单图打分结果与全量链路一致。
#
# 依赖（不随仓库打包，请自行安装）：
#   pip install ms-swift
#   transformers>=4.57  qwen_vl_utils>=0.0.14  decord
#
# 用法：
#   python infer_demo.py --image path/to/generated.jpg --prompt "一只在草地上奔跑的柯基"
#   python infer_demo.py --image path/to/generated.jpg --prompt "..." --ckpt /custom/checkpoint
# ============================================================================

import argparse
import os
import re

# 环境变量必须在 import torch / swift 之前设置（与 pipe_msswift_consistency.py 一致）
os.environ.setdefault('MAX_PIXELS', '1003520')
os.environ.setdefault('VIDEO_MAX_PIXELS', '50176')
os.environ.setdefault('FPS_MAX_FRAMES', '12')


# ── 与全量链路一致的常量 ────────────────────────────────────────────────
PROMPT = ("你的任务是做文生图一致性评测。请基于给定的图像和prompt描述，"
          "判断生成图像与prompt的一致程度，给出 1-10 分的评分，"
          "先给出综合评价，再列出扣分细节和错误原因。 prompt描述如下：")

# 模型默认相对路径：demo 在 CoT/demo/，权重在 CoT/model/checkpoint/
DEFAULT_CKPT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 '..', 'model', 'checkpoint'))


# ── 解析正则（严格沿用全量链路）────────────────────────────────────────
_SCORE_RE = re.compile(r'得分[：:]\s*(\d+)\s*分')
_TOTAL_DED_RE = re.compile(r'总扣分[：:]\s*(\d+)\s*分')
_SUMMARY_RE = re.compile(r'综合评价[：:]\s*(.*?)(?=\n\s*扣分细节[：:]|\Z)', re.DOTALL)
_DED_RE = re.compile(r'\d+\.\s*(\w+)[：:](.*?)扣(\d+)分')


def parse_response(text):
    """把模型 raw_response 解析成结构化结果。"""
    out = {
        'score': None,
        'total_deduction': None,
        'summary': '',
        'deductions': [],
        'raw_response': text or '',
    }
    if not text:
        out['error'] = 'empty response'
        return out

    m = _SCORE_RE.search(text)
    if m:
        out['score'] = int(m.group(1))

    m = _TOTAL_DED_RE.search(text)
    if m:
        out['total_deduction'] = int(m.group(1))

    m = _SUMMARY_RE.search(text)
    if m:
        out['summary'] = m.group(1).strip()

    for m in _DED_RE.finditer(text):
        out['deductions'].append({
            'category': m.group(1).strip().lower(),
            'points': int(m.group(3)),
            'reason': m.group(2).strip(),
        })

    if out['score'] is None:
        out['error'] = 'score not parsed'

    return out


def main():
    ap = argparse.ArgumentParser(
        description='WeGenBench 一致性 CoT 打分 — 单图推理 demo')
    ap.add_argument('--image', required=True, help='待评测的生成图像路径')
    ap.add_argument('--prompt', required=True, help='生成该图所用的文本 prompt')
    ap.add_argument('--ckpt', default=DEFAULT_CKPT,
                    help=f'打分模型 checkpoint 目录（默认: {DEFAULT_CKPT}）')
    ap.add_argument('--max_tokens', type=int, default=1024)
    ap.add_argument('--temperature', type=float, default=0.0)
    args = ap.parse_args()

    if not os.path.isfile(args.image):
        raise SystemExit(f'[ERROR] 图像不存在: {args.image}')
    if not os.path.isdir(args.ckpt):
        raise SystemExit(f'[ERROR] checkpoint 目录不存在: {args.ckpt}')

    # 延迟导入，确保上面的环境变量先生效
    from swift.infer_engine import TransformersEngine, RequestConfig, InferRequest

    # full 权重直接加载（与 pipe 的 full 模式一致）
    engine = TransformersEngine(args.ckpt, max_batch_size=1)
    request_config = RequestConfig(max_tokens=args.max_tokens,
                                   temperature=args.temperature)
    request = InferRequest(
        messages=[{'role': 'user', 'content': PROMPT + args.prompt}],
        images=[args.image],
    )

    resp = engine.infer([request], request_config)[0]
    text = resp.choices[0].message.content
    parsed = parse_response(text)

    print('=' * 60)
    print(f'图像   : {args.image}')
    print(f'Prompt : {args.prompt}')
    print('-' * 60)
    print(parsed['raw_response'])
    print('-' * 60)
    print(f'score           : {parsed["score"]}')
    print(f'total_deduction : {parsed["total_deduction"]}')
    print(f'summary         : {parsed["summary"]}')
    print(f'deductions      : {len(parsed["deductions"])} 项')
    for d in parsed['deductions']:
        print(f'  - [{d["category"]}] 扣{d["points"]}分: {d["reason"]}')
    if 'error' in parsed:
        print(f'[WARN] 解析提示: {parsed["error"]}')
    print('=' * 60)


if __name__ == '__main__':
    main()
