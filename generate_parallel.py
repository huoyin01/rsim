import argparse
import torch
from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from modeling_rism import RsimDecoder, RsimForCausalLM
from riscvTokenizer import riscvTokenizer
from utils import COVER_MODULES_RANGE, INSTRUCTION_MAX_LENGTH, check_ins_legal
import time


parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, required=True)
parser.add_argument('--input_text', type=str, default=None)
# 假设 INSTRUCTION_MAX_LENGTH 默认为 7
parser.add_argument('--max_length', type=int, default=INSTRUCTION_MAX_LENGTH if 'INSTRUCTION_MAX_LENGTH' in globals() else 7) 
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


def batch_generate(
    model,
    tokenizer,
    cover_inputs,
    max_length,
    temperature,
    top_k,
    top_p,
    batch_size  # <--- 增加 batch_size 参数
):
    device = cover_inputs.device
    model.eval()
    
    # 每组生成1000条
    steps = 1000

    # <--- 更改：在GPU上预先分配好 (steps, batch_size, max_length) 的 Tensor
    # 注意: model.generate 返回 (batch_size, max_length + 1)
    # 我们取 [:, 1:]，所以长度是 max_length (例如 7)
    all_generated_tokens = torch.empty(
        (steps, batch_size, max_length), 
        dtype=torch.long, 
        device=device
    )

    with tqdm(total=steps, desc="Generating batch instructions") as pbar:
        # <--- 更改：使用 step_idx 来索引
        for step_idx in range(steps):
            # 批量生成
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

            # <--- 更改：直接填充到预分配的Tensor中
            # generated_tokens[:, 1:] 的形状是 (batch_size, max_length)
            all_generated_tokens[step_idx] = generated_tokens[:, 1:]

            # 下一轮输入使用 next_cov
            cover_inputs = next_cov
            pbar.update(1)

    # <--- 更改：循环结束后，进行变形和CPU转移
    
    # 变形: [steps, batch_size, max_length] -> [batch_size, steps, max_length]
    # (1000, 16, 7) -> (16, 1000, 7)
    all_generated_tokens = all_generated_tokens.permute(1, 0, 2)
    
    # 一次性将大Tensor转移到CPU
    all_generated_tokens_cpu = all_generated_tokens.cpu()
    
    print(f"\n✅ All tokens generated and moved to CPU. Tensor shape: {all_generated_tokens_cpu.shape}")

    return all_generated_tokens_cpu


# <--- 关键修改：将 check_group 移到顶层（全局作用域） ---
def check_group(idx, ins_list_tensor): 
    """
    一个顶层函数，用于被子进程 pickle 和调用。
    检查一个 [N, 7] 的 Tensor，返回合法的指令。
    """
    legal = []
    # 这里的 check_ins_legal 必须也是顶层可导入的（你已经做到了）
    for ins in ins_list_tensor: 
        if check_ins_legal(ins): 
            legal.append(ins)
    return idx, legal


def filter_illegal_instructions(instructions_tensor):
    """对 16 组结果 (来自一个 [16, 1000, 7] 的Tensor) 进行并行非法检测"""
    
    # <--- 本地定义的 check_group 已被移除 ---

    num_batches = instructions_tensor.size(0) 
    filtered_results = [None] * num_batches
    print(f"Using ProcessPoolExecutor with max_workers= {num_batches}")
    
    with ProcessPoolExecutor(max_workers=256) as executor:
        futures = {
            # <--- 关键修改：现在调用的是顶层的 check_group ---
            executor.submit(check_group, i, instructions_tensor[i]): i 
            for i in range(num_batches)
        }
        
        with tqdm(total=len(futures), desc="Filtering batches (Parallel)") as pbar:
            for future in as_completed(futures):
                idx, legal = future.result()
                filtered_results[idx] = legal
                pbar.update(1)
                
    return filtered_results


# <--- 移除一个重复的 ThreadPoolExecutor 导入
# from concurrent.futures import ThreadPoolExecutor, as_completed

def save_results(filtered_results, output_dir):
    """此函数无需修改"""
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = riscvTokenizer()  # decode 时需要

    print(f"Saving results to {output_dir}...")
    for i, lst in enumerate(filtered_results, start=1):
        with open(os.path.join(output_dir, f"{i}.txt"), "w", encoding="utf-8") as f:
            for ins in lst:
                decoded = tokenizer.decode(ins)
                f.write(decoded.replace(',<pad>', '') + "\n")
    print(f"✅ Saved {len(filtered_results)} files in {output_dir}/")


def main():
    args = parser.parse_args()
    tokenizer = riscvTokenizer()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RsimForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32).to(device)
    print(f"Using device: {device}")

    input_text = args.input_text or "1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0,12:0,13:0,14:0,15:0,16:0,17:0,18:0,19:0,20:0,21:0,22:0"
    cover_inputs = tokenize_input(input_text, args.batch_size, device=device)

    print(f"🚀 Using batch size = {args.batch_size}")
    start_time = time.time()
    
    # <--- 更改：传入 args.batch_size 并接收 tensor
    generated_cpu_tensor = batch_generate(
        model,
        tokenizer,
        cover_inputs,
        args.max_length,
        args.temperature,
        args.top_k,
        args.top_p,
        args.batch_size # <--- 传入 batch_size
    )
    end_time = time.time()
    run_time = end_time - start_time
    print(f"⏱️ 生成时间: {run_time:.4f} 秒")
    print("🔍 Filtering illegal instructions...")
    start_time = time.time()
    # <--- 更改：将 tensor 传入
    filtered = filter_illegal_instructions(generated_cpu_tensor) 

    save_results(filtered, args.output_dir)
    end_time = time.time()
    run_time = end_time - start_time
    print(f"⏱️ 处理时间: {run_time:.4f} 秒")


if __name__ == "__main__":
    main()
