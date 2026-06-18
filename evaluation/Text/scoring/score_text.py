#!/usr/bin/env python3
"""文字打分（单文件自包含版）

把 TextPecker 识别结果与 OCR 识别结果对照 GT 文本，算出一组文字渲染指标，
输出一份 metrics json。指标实现（段级匈牙利匹配 + 字符级编辑距离对齐）与合并
逻辑都内联在本文件，除 numpy / scipy 外无其他依赖。

两种用法：

  # ① 已经合并好（每行同时含 pecker_recs 与 ocr_result）：
  python score_text.py --input merged.jsonl --output metrics.json

  # ② 分别传两套识别结果，脚本按 id 合并后再打分：
  python score_text.py \
      --pecker examples/sample_pecker.jsonl \
      --ocr    examples/sample_ocr.jsonl \
      --output examples/metrics.json

  # 只有一路时省略另一个参数，缺失那路指标自动跳过；--limit N 做 smoke test。

输入字段（GT 必需，两套识别字段任一命中即算对应指标）：
  GT 文本           : text (str 或 list[str])
  TextPecker 识别   : pecker_recs.recognized_text  或  recognized_text
  OCR 识别          : ocr_results / ocr_texts / ocr_result.res.items[*].text

输出指标：
  Pecker : pecker_qua / pecker_gned / pecker_char_F1 / _P / _R
  OCR    : edit_sim / sen_acc / ocr_char_F1 / _P / _R
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from itertools import combinations, permutations
from math import comb

import numpy as np
from scipy.optimize import linear_sum_assignment


# ════════════════════════════════════════════════════════════════════════
# 指标实现（节选自原 recalc_metrics.py，行为完全一致）
# ════════════════════════════════════════════════════════════════════════

def _edit_dist(s1, s2):
    n, m = len(s1), len(s2)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[m]


def _ned(s1, s2):
    """Normalized edit distance: 0 = identical, 1 = completely different."""
    mx = max(len(s1), len(s2))
    return _edit_dist(s1, s2) / mx if mx > 0 else 0.0


def _tokenize(text):
    """English words stay as tokens; Chinese characters split individually."""
    tokens = []
    for word in text.split():
        if not any('\u4e00' <= c <= '\u9fff' for c in word):
            tokens.append(word)
        else:
            buf = ''
            for c in word:
                if '\u4e00' <= c <= '\u9fff':
                    if buf:
                        tokens.append(buf)
                        buf = ''
                    tokens.append(c)
                else:
                    buf += c
            if buf:
                tokens.append(buf)
    return [t for t in tokens if t]


def _edit_alignment(s1, s2):
    """Levenshtein DP + backtrace → (n_match, n_sub, n_del, n_ins)."""
    n, m = len(s1), len(s2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])

    n_match = n_sub = n_del = n_ins = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and s1[i - 1] == s2[j - 1] \
                and dp[i][j] == dp[i - 1][j - 1]:
            n_match += 1; i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            n_sub += 1; i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            n_del += 1; i -= 1
        else:
            n_ins += 1; j -= 1
    return n_match, n_sub, n_del, n_ins


def calc_char_f1(gt_segs, pred_segs):
    """段级匈牙利匹配 + 字符级 TP/FP/FN → char-level (f1, precision, recall)."""
    gt_segs = [s.strip() for s in (gt_segs or []) if s and s.strip()]
    pred_segs = [s.strip() for s in (pred_segs or []) if s and s.strip()]
    n_gt, n_pred = len(gt_segs), len(pred_segs)

    if n_gt == 0 and n_pred == 0:
        return 1.0, 1.0, 1.0
    if n_gt == 0:
        return 0.0, 0.0, 0.0
    if n_pred == 0:
        return 0.0, 0.0, 0.0

    cost = np.zeros((n_gt, n_pred))
    for i in range(n_gt):
        for j in range(n_pred):
            cost[i, j] = _ned(gt_segs[i].lower(), pred_segs[j].lower())
    row_ind, col_ind = linear_sum_assignment(cost)
    matched = dict(zip(row_ind.tolist(), col_ind.tolist()))
    matched_pred = set(col_ind.tolist())

    tp = fp = fn = 0
    for i in range(n_gt):
        if i in matched:
            j = matched[i]
            g = gt_segs[i].lower()
            p = pred_segs[j].lower()
            m, s, d, ins = _edit_alignment(g, p)
            tp += m
            fp += s + ins
            fn += s + d
        else:
            fn += len(gt_segs[i])

    for j in range(n_pred):
        if j not in matched_pred:
            fp += len(pred_segs[j])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) \
        if (precision + recall) > 0 else 0.0
    return f1, precision, recall


def calc_pecker_metrics(gt_segs, recognized_text):
    """Token-level Hungarian matching → (pecker_qua, pecker_gned)."""
    clean = (recognized_text or '').replace('<#>', '#')
    gt_segs = [s for s in (gt_segs or []) if s and s.strip()]

    chars_no_space = re.sub(r'\s+', '', clean)
    total_chars = len(chars_no_space)
    if total_chars == 0:
        qua = 0.0
    else:
        qua = max(0.0, 1.0 - chars_no_space.count('#') / total_chars)

    gt_flat = ' '.join(gt_segs).strip()
    pred_flat = clean.strip()

    if not gt_flat and not pred_flat:
        return qua, 1.0
    if not gt_flat or not pred_flat:
        return qua, 0.0

    if '#' in gt_flat:
        for ch in '$&￥§':
            if ch not in gt_flat:
                pred_flat = pred_flat.replace('#', ch)
                break

    pred_tok = [t.lower() for t in _tokenize(pred_flat)]
    gt_tok = [t.lower() for t in _tokenize(gt_flat)]
    n_p, n_g = len(pred_tok), len(gt_tok)

    if n_g == 0 and n_p == 0:
        return qua, 1.0
    if n_g == 0 or n_p == 0:
        return qua, 0.0

    cost = np.zeros((n_p, n_g))
    for i in range(n_p):
        for j in range(n_g):
            cost[i, j] = _ned(pred_tok[i], gt_tok[j])

    ri, ci = linear_sum_assignment(cost)
    matched_cost = cost[ri, ci].sum()
    gned = 1.0 - (matched_cost + abs(n_p - n_g)) / max(n_p, n_g)
    return qua, gned


def calc_pecker_char_f1(gt_segs, recognized_text):
    """TextPecker char_F1（段级匈牙利匹配 + 字符级 F1）→ (f1, precision, recall)."""
    gt_segs = [s for s in (gt_segs or []) if s and s.strip()]
    clean = (recognized_text or '').replace('<#>', '\ufffd')

    if not gt_segs and not clean.strip():
        return 1.0, 1.0, 1.0
    if not gt_segs or not clean.strip():
        return 0.0, 0.0, 0.0

    gt_stripped = [re.sub(r'\s+', '', s) for s in gt_segs]
    n_gt = len(gt_stripped)

    fragments = clean.split()
    n_frag = len(fragments)

    if n_frag == 0:
        return 0.0, 0.0, 0.0

    if n_gt == 1 or n_frag == 1:
        pred_joined = ''.join(fragments)
        gt_joined = ''.join(gt_stripped)
        return calc_char_f1([gt_joined], [pred_joined])

    n_combos = comb(n_frag - 1, n_gt - 1) if n_frag > n_gt else 1

    if n_combos > 5000:
        pred_joined = ''.join(fragments)
        gt_joined = ''.join(gt_stripped)
        return calc_char_f1([gt_joined], [pred_joined])

    if n_frag <= n_gt:
        return calc_char_f1(gt_stripped, fragments)

    best_f1, best_p, best_r = -1.0, 0.0, 0.0
    for split_pos in combinations(range(1, n_frag), n_gt - 1):
        segs = []
        prev = 0
        for sp in split_pos:
            segs.append(''.join(fragments[prev:sp]))
            prev = sp
        segs.append(''.join(fragments[prev:]))

        f1, p, r = calc_char_f1(gt_stripped, segs)
        if f1 > best_f1:
            best_f1, best_p, best_r = f1, p, r

    return best_f1, best_p, best_r


def calc_ocr_metrics(gt_segs, pred_segs):
    """edit_sim: 全拼接 + 最优排序;  sen_acc: 段级匈牙利匹配判完全匹配."""
    gt_segs = [s for s in (gt_segs or []) if s and s.strip()]
    pred_segs = [s for s in (pred_segs or []) if s and s.strip()]
    n_gt = len(gt_segs)

    gt_joined = ''.join(re.sub(r'\s+', '', s) for s in gt_segs) if gt_segs else ''
    pred_stripped = [re.sub(r'\s+', '', s) for s in pred_segs]
    pred_stripped = [s for s in pred_stripped if s]
    n_pred = len(pred_stripped)

    if not gt_joined and not pred_stripped:
        return 1.0, 1.0
    if not gt_joined or not pred_stripped:
        return 0.0, 0.0

    def _calc_edit_sim():
        if n_pred == 1:
            return 1.0 - _ned(gt_joined, pred_stripped[0])

        def _greedy_align_order():
            gl = len(gt_joined)

            def _best_pos(seg):
                best_p, best_s = 0, -1
                for p in range(gl):
                    s = sum(1 for a, b in zip(gt_joined[p:], seg) if a == b)
                    if s > best_s:
                        best_s = s
                        best_p = p
                return best_p
            return sorted(range(n_pred), key=lambda i: _best_pos(pred_stripped[i]))

        if n_pred <= 5:
            best_ned = float('inf')
            for perm in permutations(range(n_pred)):
                joined = ''.join(pred_stripped[i] for i in perm)
                ned = _ned(gt_joined, joined)
                if ned < best_ned:
                    best_ned = ned
                    if ned == 0.0:
                        break
            return 1.0 - best_ned

        candidates = [list(range(n_pred)), _greedy_align_order()]
        best_ned = float('inf')
        for order in candidates:
            joined = ''.join(pred_stripped[i] for i in order)
            ned = _ned(gt_joined, joined)
            if ned < best_ned:
                best_ned = ned
        return 1.0 - best_ned

    def _calc_sen_acc():
        gt_low = [re.sub(r'\s+', '', s).lower() for s in gt_segs]
        pred_low = [re.sub(r'\s+', '', s).lower() for s in pred_segs if s and s.strip()]
        if not pred_low:
            return 0.0
        cost = np.zeros((n_gt, len(pred_low)))
        for i in range(n_gt):
            for j in range(len(pred_low)):
                cost[i, j] = _ned(gt_low[i], pred_low[j])
        row_ind, col_ind = linear_sum_assignment(cost)
        exact = sum(1 for r, c in zip(row_ind, col_ind) if gt_low[r] == pred_low[c])
        return exact / n_gt

    return _calc_edit_sim(), _calc_sen_acc()


# ════════════════════════════════════════════════════════════════════════
# 合并：TextPecker + OCR → 待打分行
# ════════════════════════════════════════════════════════════════════════

def _load_jsonl_map(path):
    out = {}
    if not (path and os.path.exists(path)):
        return out
    with open(path, 'r', encoding='utf-8') as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                d = json.loads(ln)
            except json.JSONDecodeError:
                continue
            key = (d.get('generated_imagepath')
                   or d.get('image_path')
                   or d.get('id'))
            if key is not None:
                out[key] = d
    return out


def merge_records(pecker_path, ocr_path):
    """按 id union-join 合并两套识别结果，返回 list[dict]。"""
    pmap = _load_jsonl_map(pecker_path)
    amap = _load_jsonl_map(ocr_path)
    keys = set(pmap) | set(amap)

    both = pecker_only = ocr_only = 0
    merged = []
    for k in sorted(keys, key=lambda x: str(x)):
        p = pmap.get(k)
        a = amap.get(k)
        base = dict(p) if p else dict(a)
        if a:
            for kk in ('ocr_ret', 'ocr_result'):
                if kk in a:
                    base[kk] = a[kk]
        if p and a:
            both += 1
        elif p:
            pecker_only += 1
        else:
            ocr_only += 1
        merged.append(base)

    print(f"[merge] pecker={len(pmap)} rows  ocr={len(amap)} rows")
    print(f"[merge] joined: both={both} pecker_only={pecker_only} "
          f"ocr_only={ocr_only} total={len(keys)}")
    return merged


# ════════════════════════════════════════════════════════════════════════
# 字段抽取 + 打分
# ════════════════════════════════════════════════════════════════════════

def _parse_gt(text_field):
    if text_field is None:
        return []
    if isinstance(text_field, list):
        return [str(t) for t in text_field if t]
    if isinstance(text_field, str):
        return [text_field] if text_field.strip() else []
    return [str(text_field)]


def _extract_pecker(item):
    pecker_recs = item.get('pecker_recs')
    if isinstance(pecker_recs, dict):
        rec = pecker_recs.get('recognized_text')
        if rec is not None:
            return rec, pecker_recs
    rec = item.get('recognized_text')
    if rec is not None:
        return rec, {'recognized_text': rec}
    return None, None


def _extract_ocr(item):
    if isinstance(item.get('ocr_results'), list):
        return [t for t in item['ocr_results'] if isinstance(t, str) and t]
    if isinstance(item.get('ocr_texts'), list):
        return [t for t in item['ocr_texts'] if isinstance(t, str) and t]
    ocr_result = item.get('ocr_result')
    if isinstance(ocr_result, dict):
        res = ocr_result.get('res') or {}
        items = res.get('items') or []
        return [it.get('text') for it in items
                if isinstance(it, dict) and it.get('text')]
    return None


def _ocr_char_f1(gt_segs, ocr_segs):
    """multiset (无序) 版本 char-level F1."""
    gt_segs = [s for s in (gt_segs or []) if s and s.strip()]
    ocr_segs = [s for s in (ocr_segs or []) if s and s.strip()]

    if not gt_segs and not ocr_segs:
        return 1.0, 1.0, 1.0
    if not gt_segs or not ocr_segs:
        return 0.0, 0.0, 0.0

    gt_joined = re.sub(r'\s+', '', ''.join(gt_segs))
    ocr_joined = re.sub(r'\s+', '', ''.join(ocr_segs))
    if not gt_joined and not ocr_joined:
        return 1.0, 1.0, 1.0
    if not gt_joined or not ocr_joined:
        return 0.0, 0.0, 0.0

    gt_counter = Counter(gt_joined)
    ocr_counter = Counter(ocr_joined)
    tp = sum((gt_counter & ocr_counter).values())
    fp = sum(ocr_counter.values()) - tp
    fn = sum(gt_counter.values()) - tp

    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return f1, p, r


def score_one(item):
    out = dict(item)
    gt_segs = _parse_gt(item.get('text'))
    out['text'] = gt_segs

    has_pecker = has_ocr = False

    rec_text, pecker_recs = _extract_pecker(item)
    if rec_text is not None:
        has_pecker = True
        qua, gned = calc_pecker_metrics(gt_segs, rec_text)
        f1, p, r = calc_pecker_char_f1(gt_segs, rec_text)
        out['pecker_recs'] = pecker_recs
        out['pecker_qua'] = qua
        out['pecker_gned'] = gned
        out['pecker_char_F1'] = f1
        out['pecker_char_P'] = p
        out['pecker_char_R'] = r

    ocr_segs = _extract_ocr(item)
    if ocr_segs is not None:
        has_ocr = True
        edit_sim, sen_acc = calc_ocr_metrics(gt_segs, ocr_segs)
        f1, p, r = _ocr_char_f1(gt_segs, ocr_segs)
        out['ocr_results'] = ocr_segs
        out['edit_sim'] = edit_sim
        out['sen_acc'] = sen_acc
        out['ocr_char_F1'] = f1
        out['ocr_char_P'] = p
        out['ocr_char_R'] = r

    return out, has_pecker, has_ocr


def _avg(rows, key):
    vals = [r[key] for r in rows if key in r]
    return (sum(vals) / len(vals)) if vals else float('nan')


def _iter_input_records(args):
    """根据参数产出待打分行：优先 --pecker/--ocr 合并，否则读 --input。"""
    if args.pecker or args.ocr:
        for rec in merge_records(args.pecker, args.ocr):
            yield rec
        return
    if not args.input:
        sys.exit('[ERROR] 需要 --input，或 --pecker / --ocr 之一')
    if not os.path.isfile(args.input):
        sys.exit(f'[ERROR] input not found: {args.input}')
    with open(args.input, 'r', encoding='utf-8') as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f'  [WARN] line {ln} json parse error: {e}', file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(
        description='文字打分（Pecker + OCR），单文件自包含。',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument('--input', help='已合并好的 jsonl（每行含 pecker + ocr 字段）')
    ap.add_argument('--pecker', default='', help='TextPecker 识别结果 jsonl（与 --ocr 合用，脚本内部合并）')
    ap.add_argument('--ocr', default='', help='OCR 识别结果 jsonl')
    ap.add_argument('--output', required=True, help='输出 json 路径')
    ap.add_argument('--limit', type=int, default=0,
                    help='只处理前 N 行 (0 表示全部, 用于 smoke test)')
    args = ap.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.output)) or '.'
    os.makedirs(out_dir, exist_ok=True)

    results = []
    n_pecker = n_ocr = n_with_text = 0

    for item in _iter_input_records(args):
        if not isinstance(item, dict):
            continue
        if item.get('text'):
            n_with_text += 1
        scored, hp, ho = score_one(item)
        if hp:
            n_pecker += 1
        if ho:
            n_ocr += 1
        results.append(scored)
        if args.limit and len(results) >= args.limit:
            break

    if not results:
        print('[ERROR] no valid rows', file=sys.stderr)
        sys.exit(2)

    if 'id' in results[0]:
        try:
            results.sort(key=lambda x: (x.get('id') is None, x.get('id')))
        except TypeError:
            pass

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    n = len(results)
    print('')
    print(f'[done] -> {args.output}')
    print(f'  rows         : {n}')
    print(f'  with_text    : {n_with_text}')
    print(f'  with_pecker  : {n_pecker}')
    print(f'  with_ocr     : {n_ocr}')

    if n_pecker > 0:
        print('  -- Pecker --')
        print(f'    avg pecker_qua    = {_avg(results, "pecker_qua"):.4f}')
        print(f'    avg pecker_gned   = {_avg(results, "pecker_gned"):.4f}')
        print(f'    avg pecker_char_F1= {_avg(results, "pecker_char_F1"):.4f}')
        print(f'    avg pecker_char_P = {_avg(results, "pecker_char_P"):.4f}')
        print(f'    avg pecker_char_R = {_avg(results, "pecker_char_R"):.4f}')

    if n_ocr > 0:
        print('  -- OCR --')
        print(f'    avg edit_sim      = {_avg(results, "edit_sim"):.4f}')
        print(f'    avg sen_acc       = {_avg(results, "sen_acc"):.4f}')
        print(f'    avg ocr_char_F1   = {_avg(results, "ocr_char_F1"):.4f}')
        print(f'    avg ocr_char_P    = {_avg(results, "ocr_char_P"):.4f}')
        print(f'    avg ocr_char_R    = {_avg(results, "ocr_char_R"):.4f}')

    if n_pecker == 0 and n_ocr == 0:
        print('  [WARN] 没有任何行命中 pecker / ocr 字段')


if __name__ == '__main__':
    main()
