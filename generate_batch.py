#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import torch
from modeling_rism import RsimDecoder, RsimForCausalLM
from riscvTokenizer import riscvTokenizer
from utils import INSTRUCTION_MAX_LENGTH, check_ins_legal

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, required=True, help='Path to the model')
parser.add_argument('--input_text', type=str, default=None, help='Input text to generate from')
parser.add_argument('--max_length', type=int, default=INSTRUCTION_MAX_LENGTH, help='Maximum length of the generated text')
parser.add_argument('--file', type=str, required=True, help='File to save the generated text')
parser.add_argument('--batch_size', type=int, default=2048, help='Batch size for generation')
parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for sampling')
parser.add_argument('--top_k', type=int, default=50, help='Top k for sampling')
parser.add_argument('--top_p', type=float, default=1.0, help='Top p for sampling')


def tokenize_input(cover_modules_input, device='cpu'):
    """将输入的 cover_modules 字符串转换成 tensor"""
    inputs = []
    pairs = cover_modules_input.split(',')
    for pair in pairs:
        if ':' in pair:
            _, val = pair.split(':')
            inputs.append(int(val))
        else:
            inputs.append(int(pair))
    return torch.tensor(inputs, device=device, dtype=torch.long)


def generate_instructions_batch_with_check(
    model,
    tokenizer,
    input_texts,
    output_file,
    max_length,
    batch_size=1024,
    temperature=1.0,
    top_k=50,
    top_p=1.0,
):
    """批量生成 + 合法性检测"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # 批量复制输入
    cover_batch = input_texts.unsqueeze(0).repeat(batch_size, 1)

    with torch.no_grad():
        generated_tokens, _ = model.generate(
            inputs=torch.full((batch_size, 1), -1, device=device, dtype=torch.long),
            cover_modules=cover_batch,
            max_length=max_length + 1,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    # 检测合法性
    legal_instructions = []
    illegal_count = 0
    log_file = open("log.txt", "a")
    for g in generated_tokens[:, 1:]:
        instr = tokenizer.decode(g).replace(',<pad>', '')
        if check_ins_legal(g, log_file):
            legal_instructions.append(instr)
        else:
            illegal_count += 1
            log_file.write(f"Illegal: '{instr}' from {g.tolist()}\n")
    log_file.close()

    # 保存合法指令
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for instr in legal_instructions:
            f.write(instr + '\n')

    print(f"[Done] Generated {len(generated_tokens)} instructions in batch")
    print(f"Legal instructions: {len(legal_instructions)}")
    print(f"Illegal instructions: {illegal_count}")
    print(f"Saved legal instructions to {output_file}")


def main():
    args = parser.parse_args()

    tokenizer = riscvTokenizer()
    model = RsimForCausalLM.from_pretrained(args.model_name_or_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # 准备输入
    if args.input_text:
        prompt = tokenize_input(args.input_text, device=device)
    else:
        default_text = "2,6,2,5,2,2,2,1,24,4,12,1,4,1,1,2,2,1,1,1,1,2"
        prompt = torch.tensor([int(cov) for cov in default_text.split(',')], device=device, dtype=torch.long)

    print(f"Using input cover_modules: {prompt.tolist()}")
    print(f"Params -> temperature: {args.temperature}, top_k: {args.top_k}, top_p: {args.top_p}, batch_size: {args.batch_size}")

    generate_instructions_batch_with_check(
        model,
        tokenizer,
        prompt,
        output_file=args.file,
        max_length=args.max_length,
        batch_size=args.batch_size,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )


if __name__ == "__main__":
    main()
