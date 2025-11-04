import sys
from enum import Enum
from typing import Optional

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import Trainer
from transformers.trainer_pt_utils import LengthGroupedSampler
from transformers.trainer_utils import has_length, seed_worker
from transformers.utils import is_datasets_available

if is_datasets_available():
    import datasets


COVER_MODULES_RANGE = [
    ("TLB", 0, 32767, 0),
    ("DCache", 32768, 65535, 1),
    ("TLB_1", 65536, 98303, 2),
    ("BTB", 98304, 131071, 3),
    ("FPU", 131072, 163839, 4),
    ("PTW", 163840, 196607, 5),
    ("CSRFile", 196608, 229375, 6),
    ("MulDiv", 229376, 262143, 7),
    ("Rocket", 262144, 294911, 8),
    ("Frontend", 294912, 295935, 9),
    ("ICache", 327680, 328191, 10),
    ("FPToFP", 360448, 360511, 11),
    ("TLXbar_8", 393216, 393231, 12),
    ("FPToInt", 425984, 425999, 13),
    ("IntToFP", 458752, 458767, 14),
    ("DivSqrtRawFN_small", 491520, 491527, 15),
    ("DivSqrtRawFN_small_1", 524288, 524295, 16),
    ("IBuf", 557056, 557059, 17),
    ("ShiftQueue", 589824, 589825, 18),
    ("MulAddRecFNPipe", 622592, 622593, 19),
    ("MulAddRecFNPipe_1", 655360, 655361, 20),
    ("HellaCacheArbiter", 688128, 688129, 21)
]

DATASETS_NUM = 8
INSTRUCTION_MAX_LENGTH = 6
MAX_CONTEXT_SIZE = 4096
MODEL_HIDDEN_SIZE = 768
VOCAB_SIZE = 257 # 256 instruction tokens + 1 padding token

class InstType(Enum):
    R = 1
    I = 2
    I_shift = 3
    S = 4
    B = 5
    U = 6
    J = 7
    I_fence = 8
    I_system = 9
    R4 = 10

def judge_type(first, second):
    match first:
        case 51 | 59 | 83:
            return InstType.R
        case 3 | 103 | 7:
            return InstType.I
        case 35 | 39:
            return InstType.S
        case 99:
            return InstType.B
        case 23 | 55:
            return InstType.U
        case 111:
            return InstType.J
        case 19 | 27:
            if second == 1 or second == 5:
                return InstType.I_shift
            else:
                return InstType.I
        case 15:
            return InstType.I_fence
        case 115:
            return InstType.I_system
        case 67 | 71 | 75 | 79:
            return InstType.R4
        case _:
            return None

def myhex(n):
    return "".join(f"{n:08x}")


def cat_inst(inst_type, array):
    opcode = array[0]
    # log_file.write(f"array={array},len={len(array)}\n")

    match inst_type:
        case InstType.R:
            if (len(array) < 6) or any(x == 256 for x in array[:6]):
                #print("InstType.R, (len(array) < 6) or (array[5] == 256)")
                #print(array)
                return None
            funct7 = array[1] % 128
            funct3 = array[2] % 8
            rd = array[3] % 32
            rs1 = array[4] % 32
            rs2 = array[5] % 32
            if (opcode == 0b0110011):
                if (funct7 != 0b0000000 and funct7 != 0b0100000 and funct7 != 0b0000001):
                    return None
                if (funct7 == 0b0100000 and (funct3 != 0b000 and funct3 != 0b101)):
                    return None
            if (opcode == 0b0111011):
                if (funct7 != 0 and funct7 != 0b0100000 and funct7 != 1):
                    return None
                if (funct7 == 0b0000000 and (funct3 != 0b000 and funct3 != 0b001 and funct3 != 0b101)):
                    return None
                if (funct7 == 0b0100000 and (funct3 != 0b000 and funct3 != 0b101)):
                    return None
                if (funct7 == 0b0000001 and (funct3 == 0b001 or funct3 == 0b010 or funct3 ==0b011)):
                    return None
            if (opcode == 0b1010011):
                funct5 = funct7 % 4
                if(funct3 == 0b101 or funct3 == 0b110):
                    return None
                if(funct5 != 0b00000 and funct5 != 0b00001 and funct5 != 0b00010 and funct5 != 0b00011
                   and funct5 != 0b01011 and funct5 != 0b00100 and funct5 != 0b00101
                   and funct5 != 0b11000 and funct5 != 0b11100 and funct5 != 0b10100
                   and funct5 != 0b11010 and funct5 != 0b11110):
                    return None
                if (funct5 == 0b01011):
                    rs2 = 0
                if (funct5 == 0b00100 or funct5 ==0b10100):
                    funct3 = funct3 % 3
                if (funct5 == 0b00101 or funct5 == 0b11100):
                    funct3 = funct3 % 2
                if (funct5 == 0b11110):
                    funct3 = 0
                    
            inst = (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode
            return myhex(inst)
        case InstType.I:
            if (len(array) < 6) or any(x == 256 for x in array[:6]):
                #print("InstType.I, (len(array) < 6) or (array[5] == 256)")
                #print(array)
                return None
            funct3 = array[1] % 8
            rd = array[2] % 32
            rs1 = array[3] % 32
            immLo = array[4] % 256
            immMi = array[5] % 16
            #if (opcode == 0b0000011) and ((funct3 == 0b011) or (funct3 == 0b110) or (funct3 == 0b111)):
            if (opcode == 0b0000011) and (funct3 == 0b111): # 011 and 110 legal in RV64I
                #print("InstType.I, opcode == 0b0000011) and ((funct3 == 0b011) or (funct3 == 0b110) or (funct3 == 0b111)")
                #print(array)
                return None
            if (opcode == 0b0010011) and ((funct3 == 0b001) or (funct3 == 0b101)):
                #print("InstType.I, opcode == 0b0010011) and ((funct3 == 0b001) or (funct3 == 0b101)")
                #print(array)
                return None
            if (opcode == 0b0000111) and (funct3 != 0b010 and funct3 != 0b011): #RV32F/D
                return None
            if (opcode == 0b0011011):
                funct3 = 0b000
            if (opcode == 0b1100111):
                funct3 =0b000
            inst = (immMi << 28) | (immLo << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode
            return myhex(inst)
        case InstType.I_fence:
            # Do Not Use FENCE
            return None
            if (len(array) < 6) or any(x == 256 for x in array[:6]):
                #print("InstType.I_fence, (len(array) < 6) or (array[5] == 256)")
                #print(array)
                return None
            funct3 = array[1] % 8
            succ = array[5] % 16
            pred = array[5] >> 4
            if (funct3 == 0b000):
                inst = (pred << 24) | (succ << 20) | (funct3 << 12) | opcode
            elif (funct3 == 0b001):
                inst = (funct3 << 12) | opcode
            else:
                #print("InstType.I_fence, funct3 else branch")
                #print(array)
                return None
            return myhex(inst)
        case InstType.I_system:
            # Do Not Use FENCE
            return None
            if (len(array) < 6) or any(x == 256 for x in array[:6]):
                #print("InstType.I_system, (len(array) < 6) or (array[5] == 256)")
                #print(array)
                return None
            funct3 = array[1] %8
            zimm_or_rs1 = array[3] % 32
            csr = ((array[5] % 16) << 8) | (array[4] % 256)
            if(funct3 == 0b100):
                #print("InstType.I_system, funct3 == 0b100")
                #print(array)
                return None
            if (funct3 == 0b000):
                inst = ((csr % 2) << 20) | opcode
            else:
                inst = (csr << 20) | (zimm_or_rs1 << 15) | (funct3 << 12) | opcode
            return myhex(inst)  
        case InstType.S:
            if (len(array) < 6) or any(x == 256 for x in array[:6]):
                #print("InstType.S, (len(array) < 6) or (array[5] == 256)")
                #print(array)
                return None
            funct3 = array[1] % 8
            rs1 = array[2] % 32
            rs2 = array[3] % 32
            immLo = array[4] % 256
            immMi = array[5] % 16
            # if (funct3 > 0b010):
            if (opcode == 0b0100111) and (funct3 !=0b010 and funct3!=0b011):
                return None
            if (opcode == 0b0100011 and funct3 > 0b011): # 011 legal in RV64I
                return None
            imm = (immMi << 8) | immLo
            inst = ((imm & 0b111111100000) << 20) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | ((imm & 0b11111) << 7) | opcode
            return myhex(inst)
        case InstType.B:
            if (len(array) < 6) or any(x == 256 for x in array[:6]):
                #print("InstType.B, (len(array) < 6) or (array[5] == 256)")
                #print(array)
                return None
            funct3 = array[1] % 8
            rs1 = array[2] % 32
            rs2 = array[3] % 32
            immLo = array[4] % 256
            immMi = array[5] % 16
            imm = (immMi << 8) | immLo
            if (funct3 == 0b010) or (funct3 == 0b011):
                #print("InstType.B, (funct3 == 0b010) or (funct3 == 0b011)")
                #print(array)
                return None
            inst = ((imm & 0b100000000000) << 20) | ((imm & 0b1111110000) << 21) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | ((imm & 0b1111) << 8) | ((imm & 0b10000000000) >> 3) | opcode
            return myhex(inst)
        case InstType.U:
            if (len(array) < 5) or any(x == 256 for x in array[:5]):
                #print("InstType.U, (len(array) < 5) or (array[4] == 256)")
                #print(array)
                return None
            rd = array[1] % 32
            immLo = array[2] % 256
            immMi = array[3] % 256
            immHi = array[4] %16
            inst = (immHi << 28) | (immMi << 20) | (immLo << 12) | (rd << 7) | opcode
            return myhex(inst)
        case InstType.J:
            if (len(array) < 5) or any(x == 256 for x in array[:5]):
                #print("InstType.J, (len(array) < 5) or (array[4] == 256)")
                #print(array)
                return None
            rd = array[1] % 32
            immLo = array[2] % 256
            immMi = array[3] % 256
            immHi = array[4] % 16
            immShifted = ((immHi & 0b1000) << 16) | ((immMi & 0b11) << 17) | (immLo << 9) | ((immMi & 0b100) << 6) | ((immHi & 0b111) << 5) | ((immMi & 0b11111000) >> 3)
            inst = (immShifted << 12) | (rd << 7) | opcode
            return myhex(inst)
        case InstType.I_shift:
            if (len(array) < 5) or any(x == 256 for x in array[:5]):
                #print("InstType.I_shift, (len(array) < 5) or (array[4] == 256)")
                #print(array)
                return None
            funct3 = array[1] % 8
            rd = array[2] % 32
            rs1 = array[3] % 32
            shamt = array[4] % 64
            direction = array[4] % 2
            if (funct3 == 0b001):
                inst = ((shamt % 32) << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode
            elif (funct3 == 0b101):
                inst = (direction << 30) | ((shamt % 32) << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode
            else:
                #print("InstType.I_shift, funct3 else branch")
                #print(array)
                return None
            return myhex(inst)
        case InstType.R4:
            if (len(array) < 6) or any(x == 256 for x in array[:6]):
                return None
            funct2 = array[1] % 4
            rs3 = array[1] % 32
            funct3 = array[2] % 8
            rd = array[3] % 32
            rs1 = array[4] % 32
            rs2 = array[5] % 32
            if (opcode == 0b1000011 or opcode == 0b1000111 or opcode == 0b1001011 or opcode == 0b1001111):
                if (funct3 == 0b101 or funct3 == 0b110):
                    return None
            inst = (rs3 << 27) | (funct2 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode
            return myhex(inst)
        case _:
            #print("_NoType")
            #print(array)
            return None

def check_ins_legal(instrution):
    if torch.equal(instrution, torch.tensor([3, 2, 2, 1, 0, 0], device=instrution.device, dtype=instrution.dtype)):
        return False
    inst_type = judge_type(instrution[0], instrution[1])
    out_inst = cat_inst(inst_type, instrution)
    return out_inst is not None

class NoShuffleTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # {loss_name: {"sum": float, "count": int}}
        self._custom_loss_accumulator = {}

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        每个 step 调用一次，计算 loss 并记录子 loss。
        """
        outputs = model(**inputs)
        loss = outputs.loss

        if self.state.is_local_process_zero:
            for name in ["instruction_loss", "coverage_module_loss"]:
                value = getattr(outputs, name, None)
                if value is not None:
                    value = value.detach().float().item()
                    record = self._custom_loss_accumulator.get(name, {"sum": 0.0, "count": 0})
                    record["sum"] += value
                    record["count"] += 1
                    self._custom_loss_accumulator[name] = record

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict) -> None:
        """
        每到 logging_steps 时调用。
        将子 loss 的平均值合并进 logs。
        """
        if "loss" in logs and self.state.is_local_process_zero:
            for name, record in self._custom_loss_accumulator.items():
                if record["count"] > 0:
                    logs[name] = record["sum"] / record["count"]
            # 清空累计器
            self._custom_loss_accumulator = {}

        super().log(logs)

    
    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "shuffle": self.args.shuffle_on_dataset,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        dataloader = DataLoader(train_dataset, **dataloader_params)
        dataloader = self.accelerator.prepare(dataloader)
        return dataloader
        
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        # Build the sampler.
        if not self.args.shuffle_on_dataset:
            return SequentialSampler(self.train_dataset)
        elif self.args.group_by_length:
            if is_datasets_available() and isinstance(self.train_dataset, datasets.Dataset):
                lengths = (
                    self.train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=self.train_dataset,
                lengths=lengths,
                model_input_name=model_input_name,
            )
        else:
            return RandomSampler(self.train_dataset)