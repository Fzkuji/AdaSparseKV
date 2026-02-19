# Phase 1 Baseline Results — Qwen3-8B

> Date: 2026-02-19
> Server: C (8×H20)
> Model: /mnt/data/zichuanfu/models/Qwen3-8B
> Results dir: /mnt/data/zichuanfu/SparseKV/results/phase1_qwen3

## RULER 4096 (avg string_match across 13 tasks)

| Method | 0.30 | 0.50 | 0.70 |
|--------|------|------|------|
| no_press | 95.35 | — | — |
| SnapKV | 78.74 | 55.74 | 36.73 |
| CriticalSnapKV | **91.15** | **85.02** | 67.52 |
| StreamingLLM | 74.56 | 61.08 | 47.06 |
| KVZip | — | — | **95.15** |

## RULER 16384 (avg string_match across 13 tasks)

| Method | 0.30 | 0.50 | 0.70 |
|--------|------|------|------|
| no_press | 93.02 | — | — |
| SnapKV | 78.21 | 62.81 | 46.29 |
| CriticalSnapKV | **88.21** | **83.01** | **72.59** |
| StreamingLLM | 70.43 | 55.39 | 39.60 |
| KVZip | — (failed) | — (failed) | — |

## LongBench-v2 (average score)

| Method | 0.30 | 0.50 | 0.70 |
|--------|------|------|------|
| no_press | 0.1849 | — | — |
| SnapKV | 0.1869 | 0.1869 | 0.1630 |
| CriticalSnapKV | 0.1869 | 0.1889 | 0.1730 |
| StreamingLLM | **0.2962** | **0.3062** | **0.2962** |

⚠️ StreamingLLM > no_press 反直觉，可能评测噪声或模型在该benchmark上full attention不占优

## InfiniteBench longbook_qa_eng (score)

| Method | 0.30 | 0.50 | 0.70 |
|--------|------|------|------|
| no_press | 0.0301 | — | — |
| SnapKV | 0.0326 | 0.0322 | 0.0356 |
| CriticalSnapKV | 0.0289 | 0.0318 | 0.0317 |
| StreamingLLM | 0.0351 | 0.0394 | 0.0413 |

⚠️ 所有方法 ~3-4%，Qwen3-8B 本身无法胜任此任务，无参考价值

## Key Takeaways

1. **CriticalSnapKV 是最强 eviction baseline**（RULER 上远超 SnapKV）
2. CriticalSnapKV 0.30 只掉 4-5 点（91.15 vs 95.35 on 4096; 88.21 vs 93.02 on 16384）
3. KVZip 0.70 在 4096 上几乎无损（95.15 vs 95.35），但其他 ratio 和 16384 跑失败了
4. SnapKV 很弱，不应作为 beat target
5. InfiniteBench 对 Qwen3-8B 无意义

## Missing Data

- KVZip: ruler_4096 只有 0.70 成功，0.30/0.50 FAIL；ruler_16384 全部 FAIL；longbench-v2/infinitebench 未完成
- 之前 v6a/v6b/v7 的具体数值未记录，只知道"都没 beat base SnapKV"，需要从 Server B 获取

## SparseKV Trained Models (Server B)

- v6a adapter: /root/autodl-tmp/output/ (CE+KL)
- v6b adapter: /root/autodl-tmp/output/ (CE only, no eviction)
- v7 adapter: /root/autodl-tmp/output/ (pure distillation)
- 排名: v6a > v7 > v6b
- 均未 beat base SnapKV（但 SnapKV 本身很弱，需重新对比 CriticalSnapKV）
