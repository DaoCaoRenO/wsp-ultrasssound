#!/usr/bin/env bash
# 组装 data_nothy/ 目录: 保留非甲状腺 11 子集 symlink, 给被排除的 DDTI/private_Thyroid
# 补空 split.txt + config.yaml (兼容 blsamus omni_trainer.py 硬编码 seg_val_set 遍历)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
SRC="$ROOT/data/segmentation"
DST="$ROOT/data_nothy/segmentation"
mkdir -p "$DST"
# 排除: DDTI, private_Thyroid, TN3K (甲状腺); 其他 symlink
for d in "$SRC"/*/; do
  name="$(basename "$d")"
  case "$name" in
    DDTI|private_Thyroid|TN3K) continue ;;
  esac
  ln -sfn "$(realpath "$d")" "$DST/$name"
done
# 给被排除的两个甲状腺补空目录: 让 val loader 读到 0 样本、循环 skip
for name in DDTI private_Thyroid; do
  mkdir -p "$DST/$name"
  : > "$DST/$name/train.txt"
  : > "$DST/$name/val.txt"
  : > "$DST/$name/test.txt"
  cp "$SRC/$name/config.yaml" "$DST/$name/config.yaml"
done
# classification + Val 目录 symlink 保持(非甲状腺数据)
ln -sfn "$(realpath "$ROOT/data/classification")" "$ROOT/data_nothy/classification"
ln -sfn "$(realpath "$ROOT/data/Val")" "$ROOT/data_nothy/Val"
echo "OK: $DST 11 非甲状腺 symlink + 2 甲状腺空壳 (config.yaml only)"
