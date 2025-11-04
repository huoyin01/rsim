#!/usr/bin/env python3
import os
import argparse

def count_lines_in_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return sum(1 for _ in f)
    except Exception as e:
        print(f"⚠️ Could not read {filepath}: {e}")
        return 0

def count_lines_in_dir(directory):
    total_lines = 0
    for root, _, files in os.walk(directory):
        for file in files:
            path = os.path.join(root, file)
            total_lines += count_lines_in_file(path)
    return total_lines

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count total lines in all files under a directory")
    parser.add_argument('directory', type=str, help="Target directory to count lines")
    args = parser.parse_args()

    directory = os.path.abspath(args.directory)
    total = count_lines_in_dir(directory)
    print(f"📄 Total lines in all files under {directory}: {total}")
