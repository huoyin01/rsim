import argparse
import torch
from tqdm import tqdm
import os
import time

from modeling_rism import RsimDecoder, RsimForCausalLM
from riscvTokenizer import riscvTokenizer
from utils import COVER_MODULES_RANGE, INSTRUCTION_MAX_LENGTH, check_ins_legal


parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, required=True)
parser.add_argument('--input_text', type=str, default=None)
parser.add_argument('--max_length', type=int, default=INSTRUCTION_MAX_LENGTH)
parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument('--top_k', type=int, default=50)
parser.add_argument('--top_p', type=float, default=1.0)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--output_dir', type=str, default="generation_parallel")


def tokenize_input(cover_modules_input, batch_size=16, device='cpu'):
    """生成 batch_size 个相同输入，用于并行"""
    pairs = cover_modules_input.split(',')
    inputs = [0] * len(COVER_MODULES_RANGE)
    for pair in pairs:
        index, value = pair.split(':')
        inputs[int(index)-1] = int(value)
    tensor = torch.tensor(inputs, device=device, dtype=torch.long)
    tensor = tensor.unsqueeze(0).repeat(batch_size, 1)
    return tensor


def batch_generate(model, tokenizer, cover_inputs, max_length, temperature, top_k, top_p):
    device = cover_inputs.device
    model.eval()
    generated_texts = [[] for _ in range(cover_inputs.size(0))]

    steps = 1000
    with tqdm(total=steps, desc="Generating batch instructions") as pbar:
        for _ in range(steps):
            with torch.no_grad():
                generated_tokens, next_cov = model.generate(
                    inputs=torch.full((cover_inputs.size(0), 1), -1, device=device, dtype=torch.long),
                    cover_modules=cover_inputs,
                    max_length=max_length + 1,
                    do_sample=True,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )

            for i, tokens in enumerate(generated_tokens[:, 1:]):
                generated_texts[i].append(tokens)

            cover_inputs = next_cov
            pbar.update(1)

    return generated_texts


def filter_illegal_instructions(instruction_lists):
    """顺序过滤非法指令"""
    filtered_results = []
    for lst in tqdm(instruction_lists, desc="Filtering illegal instructions"):
        legal = []
        for ins in lst:
            if check_ins_legal(ins):
                legal.append(ins)
        filtered_results.append(legal)
    return filtered_results


def save_results(filtered_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = riscvTokenizer()

    for i, lst in enumerate(filtered_results, start=1):
        path = os.path.join(output_dir, f"{i}.txt")
        with open(path, "w", encoding="utf-8") as f:
            for ins in lst:
                decoded = tokenizer.decode(ins)
                f.write(decoded.replace(',<pad>', '') + "\n")

    print(f"✅ Saved {len(filtered_results)} files in {output_dir}/")


def main():
    args = parser.parse_args()
    tokenizer = riscvTokenizer()
    model = RsimForCausalLM.from_pretrained(args.model_name_or_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    input_text = args.input_text or \
        "1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0,12:0,13:0,14:0,15:0,16:0,17:0,18:0,19:0,20:0,21:0,22:0"
    cover_inputs = tokenize_input(input_text, args.batch_size, device=device)

    print(f"🚀 Using batch size = {args.batch_size}")

    start_time = time.time()
    generated = batch_generate(
        model,
        tokenizer,
        cover_inputs,
        args.max_length,
        args.temperature,
        args.top_k,
        args.top_p
    )
    print(f"⏱️ 生成时间: {time.time() - start_time:.2f} 秒")

    print("🔍 Filtering illegal instructions...")
    start_time = time.time()
    filtered = filter_illegal_instructions(generated)
    save_results(filtered, args.output_dir)
    print(f"⏱️ 处理时间: {time.time() - start_time:.2f} 秒")


if __name__ == "__main__":
    main()
