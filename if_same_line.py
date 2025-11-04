#!/usr/bin/env python3
import sys
from collections import defaultdict

if len(sys.argv) != 2:
    print(f"用法: {sys.argv[0]} 文件名")
    sys.exit(1)

filename = sys.argv[1]

lines_seen = defaultdict(list)

with open(filename, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f, start=1):
        line = line.rstrip("\n")  # 去掉换行符
        lines_seen[line].append(idx)

found_duplicate = False
for line, indexes in lines_seen.items():
    if len(indexes) > 1:
        found_duplicate = True
        print(f"重复行: '{line}' 出现在行 {indexes}")

if not found_duplicate:
    print("没有重复的行。")
