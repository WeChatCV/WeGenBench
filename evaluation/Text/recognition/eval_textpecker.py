#!/usr/bin/env python3
"""
TextPecker 文字识别（recognition）

对一份 jsonl（每行含 生成图路径 + GT 文本）逐图调用 TextPecker，得到带
结构化错误标记 (<#>/<###>) 的识别结果，并算出 pecker_qua / pecker_gned /
pecker_recs，逐行写出 eval_results.jsonl。识别结果之后会被 scoring/ 里的
打分脚本消费。

依赖（均为外部 ref，不随本仓库打包）：
  1. TextPecker 仓库（提供 parse_utils_pecker.get_score_v2 等）：
     https://github.com/...  （你自己的 TextPecker checkout）
     通过 --textpecker_root 或环境变量 TEXTPECKER_ROOT 指向仓库根目录。
  2. 一个用 vLLM 起好的 TextPecker 服务（OpenAI 兼容接口），
     通过 --vllm_host / --port 指向。

输入 jsonl 每行：
  {
    "id": 185,
    "prompt": "生成提示词",
    "text": ["目标文本1", "目标文本2"],
    "generated_imagepath": "/path/to/image.png",
    ...其他字段（原样保留到结果中）
  }

用法：
  export TEXTPECKER_ROOT=/path/to/TextPecker
  python eval_textpecker.py \
      --input_jsonl data.jsonl \
      --output_dir  ./results/flux1 \
      --vllm_host localhost --port 1925 \
      --resume
"""

import os
import sys
import json
import argparse
import asyncio
import base64
import time
import threading
import warnings
from io import BytesIO
from typing import Union
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from concurrent import futures

warnings.filterwarnings("ignore")


# ── 把 TextPecker 仓库加入 sys.path（ref，不随本仓库打包）─────────────────
def _bootstrap_textpecker_path(textpecker_root):
    """把 TextPecker 仓库相关目录加入 sys.path，使 parse_utils_pecker 可被导入。"""
    if not textpecker_root:
        sys.exit(
            "[ERROR] 未指定 TextPecker 仓库路径。请用 --textpecker_root 传入，"
            "或设置环境变量 TEXTPECKER_ROOT 指向你的 TextPecker checkout 根目录。")
    root = os.path.abspath(os.path.expanduser(textpecker_root))
    if not os.path.isdir(root):
        sys.exit(f"[ERROR] textpecker_root 不存在: {root}")
    # parse_utils_pecker.py 位于 <root>/eval/TextPecker_eval/
    candidates = [
        os.path.join(root, 'eval', 'TextPecker_eval'),
        os.path.join(root, 'eval'),
        root,
    ]
    for p in candidates:
        if os.path.isdir(p) and p not in sys.path:
            sys.path.append(p)


# 全局 vLLM 客户端缓存
from openai import AsyncOpenAI

_thread_local = threading.local()
_process_clients = {}
_vllm_client_lock = threading.Lock()


def parse_args():
    parser = argparse.ArgumentParser(description='TextPecker 文字识别')
    parser.add_argument('--input_jsonl', type=str, required=True,
                        help='输入的 JSONL 文件路径')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='识别结果输出目录')
    parser.add_argument('--textpecker_root', type=str,
                        default=os.environ.get('TEXTPECKER_ROOT', ''),
                        help='TextPecker 仓库根目录（默认读环境变量 TEXTPECKER_ROOT）')
    parser.add_argument('--port', type=int, default=1925,
                        help='TextPecker vLLM 服务端口')
    parser.add_argument('--vllm_host', type=str, default='localhost',
                        help='TextPecker vLLM 服务地址')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='每批处理的图像数量')
    parser.add_argument('--image_field', type=str, default='generated_imagepath',
                        help='JSONL 中图像路径字段名（默认: generated_imagepath）')
    parser.add_argument('--text_field', type=str, default='text',
                        help='JSONL 中目标文本字段名（默认: text）')
    parser.add_argument('--prompt_field', type=str, default='prompt',
                        help='JSONL 中 prompt 字段名（默认: prompt）')
    parser.add_argument('--resume', action='store_true',
                        help='是否断点续传（跳过已识别的样本）')
    return parser.parse_args()


def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"警告: 第 {line_num} 行 JSON 解析失败: {e}")
                continue
    return data


def load_evaluated_ids(output_dir):
    """加载已识别的样本 ID（用于断点续传）"""
    result_file = os.path.join(output_dir, 'eval_results.jsonl')
    evaluated_ids = set()
    if os.path.exists(result_file):
        with open(result_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if 'id' in obj:
                        evaluated_ids.add(obj['id'])
                except Exception:
                    continue
    return evaluated_ids


def prepare_data(raw_data, image_field, text_field, prompt_field, evaluated_ids=None):
    """把用户 jsonl 转成识别所需格式。text 列表用空格连接为 target。"""
    data = []
    skipped = 0
    missing_image = 0

    for item in raw_data:
        image_path = item.get(image_field) or item.get("generated_image_path", "")
        if not image_path:
            print(f"警告: 样本 id={item.get('id', '?')} 缺少图像路径字段 '{image_field}'，跳过")
            skipped += 1
            continue

        if not os.path.exists(image_path):
            missing_image += 1
            continue

        text_data = item.get(text_field, [])
        if isinstance(text_data, list):
            target = ' '.join(text_data)
        elif isinstance(text_data, str):
            target = text_data
        else:
            target = str(text_data)

        prompt = item.get(prompt_field, '')
        sample_id = item.get('id', len(data))

        if evaluated_ids and sample_id in evaluated_ids:
            continue

        data.append({
            'image': image_path,
            'prompt': prompt,
            'target': target,
            'ori_target': target,
            'id': sample_id,
            '_original': item,
        })

    if missing_image > 0:
        print(f"警告: {missing_image} 个样本的图像文件不存在，已跳过")
    if skipped > 0:
        print(f"警告: {skipped} 个样本缺少必要字段，已跳过")

    return data


def build_scorer(vllm_host, vllm_port, get_score_v2):
    """构建 TextPecker 识别函数（依赖 vLLM 服务 + parse_utils_pecker.get_score_v2）。"""
    if ":" in vllm_host:
        vllm_base_url = f"http://[{vllm_host}]:{vllm_port}/v1"
    else:
        vllm_base_url = f"http://{vllm_host}:{vllm_port}/v1"

    vllm_api_key = "EMPTY"
    model_name = "TextPecker"

    print(f"连接 TextPecker 服务: {vllm_base_url}")

    client_key = f"{vllm_base_url}_{vllm_api_key}"
    if client_key not in _process_clients:
        with _vllm_client_lock:
            if client_key not in _process_clients:
                _process_clients[client_key] = AsyncOpenAI(
                    base_url=vllm_base_url, api_key=vllm_api_key)
    client = _process_clients[client_key]

    def pil_image_to_base64(image: Union[Image.Image, str]) -> str:
        if isinstance(image, str):
            image = Image.open(image)
        buffered = BytesIO()
        image.convert("RGB").save(buffered, format="JPEG")
        encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"

    def get_rec_template():
        return """
        This is a text-generated image. Please recognize all visible text in the entire image.
        Marking rules: 
        1. Use <#> for structurally flawed (e.g., extra/missing strokes, distortion) unrecognizable Chinese characters or single English letters;
        2. Use <###> exclusively for structurally flawed unrecognizable single English words (not multi-word phrases, lines, or sentences).
        Output in the following JSON format:
        {
        "recognized_text": "All text in the image (including structural error markers)"
        }
        """

    def run_async_task(coro):
        if not hasattr(_thread_local, 'loop'):
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                _thread_local.loop = loop
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                _thread_local.loop = loop
        else:
            loop = _thread_local.loop
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                _thread_local.loop = loop

        try:
            if loop.is_running():
                future = asyncio.run_coroutine_threadsafe(coro, loop)
                return future.result()
            return loop.run_until_complete(coro)
        except Exception as e:
            print(f"异步任务执行出错: {e}，正在重试...")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            _thread_local.loop = loop
            return loop.run_until_complete(coro)

    async def evaluate_image(prompt, base64_image, max_retries=3):
        retry_count = 0
        last_exception = None
        while retry_count < max_retries:
            try:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": base64_image}},
                            {"type": "text", "text": prompt},
                        ],
                    }],
                    temperature=0.0,
                    max_tokens=2048,
                    extra_body={"repetition_penalty": 1.2},
                )
                return response.choices[0].message.content
            except Exception as e:
                retry_count += 1
                last_exception = e
                wait_time = min(5 * (2 ** (retry_count - 1)), 30)
                print(f"请求失败 (第 {retry_count}/{max_retries} 次): {e}，{wait_time}s 后重试...")
                await asyncio.sleep(wait_time)
        raise RuntimeError(f"请求失败，已重试 {max_retries} 次") from last_exception

    async def evaluate_batch_image(prompts, images):
        images_base64 = [pil_image_to_base64(img) for img in images]
        tasks = [evaluate_image(p, b64) for p, b64 in zip(prompts, images_base64)]
        return await asyncio.gather(*tasks)

    def scorer(images, prompts, targets):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)
            images = [Image.fromarray(img) for img in images]

        vllm_prompts = [get_rec_template() for _ in targets]
        text_outputs = run_async_task(evaluate_batch_image(vllm_prompts, images))

        qua_rewards, gned_scores, recs = [], [], []
        qua_amplify_factor = 1.0

        for j, response in enumerate(text_outputs):
            try:
                quality_score, gned_score, cls_results = get_score_v2(
                    response, targets[j], qua_amplify_factor, True)
            except Exception as e:
                print(f'评分出错: {e}\n响应内容: {response}')
                quality_score, gned_score, cls_results = 0, 0, {}
            qua_rewards.append(quality_score)
            gned_scores.append(gned_score)
            recs.append(cls_results)

        return {
            'pecker_quas': qua_rewards,
            'pecker_gned': gned_scores,
            'pecker_recs': recs,
        }

    return scorer


def run_evaluation(data, scorer, output_dir, batch_size=8):
    os.makedirs(output_dir, exist_ok=True)
    result_file = os.path.join(output_dir, 'eval_results.jsonl')

    executor = futures.ThreadPoolExecutor(max_workers=8)
    new_data = []
    start_time = time.time()

    total_batches = (len(data) + batch_size - 1) // batch_size
    print(f"总计 {len(data)} 个样本，分 {total_batches} 批处理（batch_size={batch_size}）")

    with open(result_file, 'a', encoding='utf-8') as f_out:
        with tqdm(total=total_batches, desc="识别进度", unit="batch") as pbar:
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                current_batch = i // batch_size + 1
                try:
                    pil_images, prompts, targets, valid_items = [], [], [], []
                    for item in batch:
                        try:
                            img = Image.open(item['image']).convert('RGB')
                            pil_images.append(img)
                            prompts.append(item['prompt'])
                            targets.append(item['target'])
                            valid_items.append(item)
                        except Exception as e:
                            print(f"加载图像失败 {item['image']}: {e}")
                            continue

                    if not pil_images:
                        pbar.update(1)
                        continue

                    future = executor.submit(scorer, pil_images, prompts, targets)
                    score_details = future.result()

                    pecker_quas = score_details['pecker_quas']
                    pecker_gned = score_details['pecker_gned']
                    pecker_recs = score_details['pecker_recs']

                    for j, item in enumerate(valid_items):
                        result_item = item.get('_original', {}).copy()
                        result_item['pecker_qua'] = pecker_quas[j]
                        result_item['pecker_gned'] = pecker_gned[j]
                        result_item['pecker_recs'] = pecker_recs[j]
                        new_data.append(result_item)
                        f_out.write(json.dumps(result_item, ensure_ascii=False) + '\n')
                        f_out.flush()

                    pbar.set_postfix(
                        processed=len(new_data),
                        avg_qua=f"{np.mean(pecker_quas):.3f}",
                        avg_gned=f"{np.mean([g for g in pecker_gned if g != 'None']):.3f}"
                        if any(g != 'None' for g in pecker_gned) else "N/A",
                    )
                    pbar.update(1)
                except Exception as e:
                    print(f"批次 {current_batch}/{total_batches} 处理出错: {e}")
                    pbar.update(1)
                    continue

    total_time = time.time() - start_time
    print(f"\n识别完成！共处理 {len(new_data)} 个样本，耗时 {total_time:.2f}s")
    print(f"JSONL 结果已保存到: {result_file}")

    save_as_json(result_file, os.path.join(output_dir, 'eval_results.json'))
    return new_data


def save_as_json(jsonl_file, json_file):
    all_results = []
    if os.path.exists(jsonl_file):
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    all_results.append(json.loads(line))
                except Exception:
                    continue
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"JSON 结果已保存到: {json_file}")


def print_summary(result_file):
    if not os.path.exists(result_file):
        return

    qua_scores, gned_scores = [], []
    type_stats = defaultdict(lambda: {'qua': [], 'gned': []})

    with open(result_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                qua = item.get('pecker_qua')
                gned = item.get('pecker_gned')
                if qua is not None and qua != 'ERROR':
                    qua_scores.append(float(qua))
                if gned is not None and gned not in ('None', 'ERROR'):
                    gned_scores.append(float(gned))
                item_type = item.get('type') or item.get('一级细化场景') or 'unknown'
                if qua is not None and qua != 'ERROR':
                    type_stats[item_type]['qua'].append(float(qua))
                if gned is not None and gned not in ('None', 'ERROR'):
                    type_stats[item_type]['gned'].append(float(gned))
            except Exception:
                continue

    print("\n" + "=" * 60)
    print("TextPecker 识别结果汇总")
    print("=" * 60)
    print(f"总样本数: {len(qua_scores)}")
    if qua_scores:
        print("\n【整体指标】")
        print(f"  Quality Score (Qua) 平均值: {np.mean(qua_scores):.4f}")
        if gned_scores:
            print(f"  Semantic Score (Sem) 平均值: {np.mean(gned_scores):.4f}")
    if type_stats:
        print("\n【按类型统计】")
        print(f"{'类型':<20} {'样本数':>6} {'Qua均值':>10} {'Sem均值':>10}")
        print("-" * 50)
        for t in sorted(type_stats.keys()):
            stats = type_stats[t]
            n = len(stats['qua'])
            avg_qua = f"{np.mean(stats['qua']):.4f}" if stats['qua'] else "N/A"
            avg_gned = f"{np.mean(stats['gned']):.4f}" if stats['gned'] else "N/A"
            print(f"{t:<20} {n:>6} {avg_qua:>10} {avg_gned:>10}")
    print("=" * 60)


def main():
    args = parse_args()

    _bootstrap_textpecker_path(args.textpecker_root)
    from parse_utils_pecker import get_score_v2

    print(f"输入文件: {args.input_jsonl}")
    print(f"输出目录: {args.output_dir}")
    print(f"TextPecker 服务: {args.vllm_host}:{args.port}")
    print(f"批次大小: {args.batch_size}")

    print("\n[1/4] 加载 JSONL 数据...")
    raw_data = load_jsonl(args.input_jsonl)
    print(f"  共加载 {len(raw_data)} 条数据")

    evaluated_ids = None
    if args.resume:
        evaluated_ids = load_evaluated_ids(args.output_dir)
        if evaluated_ids:
            print(f"  断点续传模式：已跳过 {len(evaluated_ids)} 个已识别样本")

    print("\n[2/4] 准备识别数据...")
    data = prepare_data(
        raw_data,
        image_field=args.image_field,
        text_field=args.text_field,
        prompt_field=args.prompt_field,
        evaluated_ids=evaluated_ids,
    )
    print(f"  待识别样本数: {len(data)}")
    if not data:
        print("没有需要识别的样本，退出。")
        return

    print("\n[3/4] 连接 TextPecker 服务...")
    scorer = build_scorer(args.vllm_host, args.port, get_score_v2)

    print("\n[4/4] 开始识别...")
    run_evaluation(data, scorer, args.output_dir, args.batch_size)

    print_summary(os.path.join(args.output_dir, 'eval_results.jsonl'))


if __name__ == '__main__':
    main()
