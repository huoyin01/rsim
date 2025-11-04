import argparse

import torch
from tqdm import tqdm

from modeling_rism import RsimDecoder, RsimForCausalLM
from riscvTokenizer import riscvTokenizer
from utils import COVER_MODULES_RANGE, INSTRUCTION_MAX_LENGTH, check_ins_legal

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, required=True, help='Path to the model')
parser.add_argument('--input_text', type=str, default=None, help='Input text to generate from')
parser.add_argument('--max_length', type=int, default=INSTRUCTION_MAX_LENGTH, help='Maximum length of the generated text')
parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for sampling')
parser.add_argument('--top_k', type=int, default=50, help='Top k for sampling')
parser.add_argument('--top_p', type=float, default=1.0, help='Top p for sampling')
parser.add_argument('--file', type=str, required=True, help='File to save the generated text')


def tokenize_input(cover_modules_input, device='cpu'):
    pairs = cover_modules_input.split(',')
    inputs = [0] * len(COVER_MODULES_RANGE)
    for pair in pairs:
        index, value = pair.split(':')
        inputs[int(index)-1] = int(value)
    return {
        "input_ids": torch.tensor([[-1]], device=device, dtype=torch.long),
        "cover_modules": torch.tensor(inputs, device=device, dtype=torch.long).unsqueeze(0)
    }

def generate_instructions(
    model, 
    tokenizer, 
    input_texts, 
    file, 
    max_length,
    do_sample=False,
    penalty_alpha=None,
    temperature=1.0, 
    top_k=50,
    top_p=1.0,
    repetition_penalty=1.0,
    no_repeat_ngram_size=0,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    generated_instructions = []
    coverage_history = []  # 保存每个合法指令生成前的覆盖率状态
    
    # cover_modules_input为[1, 22]的tensor
    pbar = tqdm(range(1000), desc="Generating instructions")
    i = 0
    total_illegal_count = 0
    # log_file = open("log.txt", "a")
    while i < 1000:
        instruction_is_legal = False
        illegal_count = 0
        
        # 保存当前状态，以便回退
        last_valid_coverage = input_texts.clone()
        
        while not instruction_is_legal:
            generated_tokens, next_cov = model.generate(
                inputs=torch.tensor([[-1]], device=device, dtype=torch.long),
                cover_modules=input_texts,
                max_length=max_length+1,
                do_sample=do_sample,
                penalty_alpha=penalty_alpha,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            instruction_is_legal = check_ins_legal(generated_tokens[:, 1:][0])
            if not instruction_is_legal:
                illegal_count += 1
                total_illegal_count += 1
                #print(f"\rIllegal instructions generated: {illegal_count}", end="")
                if illegal_count > 30:
                    print("\nToo many illegal instructions, backtracking...")
                    if coverage_history:
                        # 回退到上一个状态
                        input_texts = coverage_history.pop()
                        generated_instructions.pop()
                        i -= 1 # tqdm 进度回退
                        pbar.update(-1)
                    else:
                        # 如果没有历史记录，无法回退，只能重置计数器
                        print("No history to backtrack, resetting count.")
                    illegal_count = 0
                    continue # 重新生成上一步
            
        # print() # 换行
        
        coverage_history.append(last_valid_coverage)
        for g in generated_tokens[:, 1:]:
            decoded_instruction = tokenizer.decode(g)
            # log_file.write(f"Decoded '{decoded_instruction}' from {g.tolist()}\n")
            generated_instructions.append(decoded_instruction)
        input_texts = next_cov
        # print("Next Cov Is")
        # print(next_cov)
        i += 1
        pbar.update(1)

        
    pbar.close()
    print(f"Generated {len(generated_instructions)} instructions")
    print(f"Total illegal instructions generated: {total_illegal_count}")
    
    # log_file.close()

    # Ensure the output directory exists
    import os
    output_dir = os.path.dirname(file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    with open(file, 'w', encoding='utf-8') as f:
        for instruction in generated_instructions:
            instruction = instruction.replace(',<pad>', '')
            f.write(instruction + '\n')

def main():
    args = parser.parse_args()
    
    tokenizer = riscvTokenizer()
    model = RsimForCausalLM.from_pretrained(args.model_name_or_path)
    # model.set_decoder(RsimDecoder(model.config))
    
    import os
    from safetensors.torch import load_file as safe_load_file
        
    # # 确定文件路径
    # filepath = args.model_name_or_path
    # # model.load_state_dict(safe_load_file(os.path.join(filepath, "model.safetensors"), device="cuda"))

    # # 加载safetensors文件
    # if os.path.exists(filepath):
    #     # 使用utils.safe_load加载权重
    #     weights = safe_load_file(os.path.join(filepath, "model.safetensors"), device="cuda")
    #     model.load_state_dict(weights)
    #     # 打印权重信息
    #     # for name, weight in weights.items():
    #     #     print(f"{name}: {weight.shape}")
    # else:
    #     print(f"File {filepath} does not exist.")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # if args.input_text is None:
    #     print("Please enter the input cover_modules to generate from:")
    #     cover_modules_input = input()
    # else:
    #     cover_modules_input = args.input_text
    
    file = args.file
    if args.input_text:
        # use the input_text from command line
        inputs = tokenize_input(args.input_text, device=device)
        prompt = inputs['cover_modules'].squeeze(0)
    else:
        # use the hardcoded input_text
        input_texts = "2,6,2,5,2,2,2,1,24,4,12,1,4,1,1,2,2,1,1,1,1,2"
        prompt = torch.tensor([int(cov) for cov in input_texts.split(',')], device=device, dtype=torch.long)
    prompt = prompt.view(1, -1)
    print(f"Using input cover_modules: {prompt.tolist()}")
    generate_instructions(
        model, 
        tokenizer, 
        prompt, 
        file, 
        max_length=args.max_length,
        do_sample=True,
        # penalty_alpha=0.6,
        temperature=args.temperature, 
        top_k=args.top_k,
        top_p=args.top_p,
        # repetition_penalty=1.2,
        # no_repeat_ngram_size=3,
    )
    
    # inputs = tokenize_input(cover_modules_input, device=device)
    # generated_tokens = model.generate(
    #     inputs=inputs["input_ids"],
    #     cover_modules=inputs["cover_modules"], 
    #     temperature=args.temperature, 
    #     max_length=args.max_length+1,
    #     bos_token_id=tokenizer.bos_token_id,
    #     eos_token_id=tokenizer.eos_token_id,
    #     pad_token_id=tokenizer.pad_token_id,
    # )
    # generated_texts = [tokenizer.decode(g) for g in generated_tokens[:, 1:]]
    
    # for text in generated_texts:
    #     print(text)
    

if __name__ == "__main__":
    main()