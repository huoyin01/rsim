#!/usr/bin/env bash
set -euo pipefail

# --- 可修改变量 ---
N=1000
TEMP="1.2"
TOP_K="30"
TOP_P="0.8"

: "${HARDWARE_FUZZER_DIR:=$HOME/Hardware-Fuzzer}"

MODEL_PATH="results/20251013/checkpoint-650230"
INPUT_TEXT="1:2,2:3,3:2,4:1,5:2,6:2,7:2,8:1,9:5,10:4,11:8,12:1,13:4,14:1,15:1,16:2,17:2,18:1,19:1,20:1,21:1,22:2"

GEN_DIR="generation_l1000"
TOKEN2INST_DIR="../token2inst"
SAVE_DIR="../gen_inst_save"

mkdir -p "${GEN_DIR}"
mkdir -p "${SAVE_DIR}"

START_PWD="$(pwd)"

echo "Run settings: N=${N}, temp=${TEMP}, top_k=${TOP_K}, top_p=${TOP_P}"
echo "Model: ${MODEL_PATH}"
echo

for (( i=1; i<=N; i++ )); do
    base="${i}"
    outfile="${GEN_DIR}/${base}.txt"

    echo "==== Iteration ${i}/${N}: generating -> ${outfile} ===="

    (
        # 1) 生成
        python generate.py \
            --model_name_or_path="${MODEL_PATH}" \
            --input_text="${INPUT_TEXT}" \
            --temperature="${TEMP}" \
            --top_k="${TOP_K}" \
            --top_p="${TOP_P}" \
            --file="${outfile}"

        # # 2) 拷贝到 token2inst/generation/
        # mkdir -p "${TOKEN2INST_DIR}/generation"
        # cp "${outfile}" "${TOKEN2INST_DIR}/generation/"

        # # 3) 在 token2inst 目录下执行 make
        # pushd "${TOKEN2INST_DIR}" >/dev/null
        # echo "Running: make -f Makefile.manual all TARGETS=\"${base}.txt\""
        # make -f Makefile.manual all TARGETS="${base}.txt"

        # # 检查输出文件是否存在
        # if [[ ! -f "cleaned_dumps/${base}.txt" ]]; then
        #     echo "ERROR: cleaned_dumps/${base}.txt not found after make" >&2
        #     popd >/dev/null
        #     exit 1
        # fi

        # # 4) 拷贝清理后的文件到 Hardware-Fuzzer/sim/inst_gen
        # cp "cleaned_dumps/${base}.txt" "${SAVE_DIR}/"
        # popd >/dev/null

        # 5) 运行硬件仿真
        # pushd "${HARDWARE_FUZZER_DIR}/sim" >/dev/null
        # echo "Running hardware fuzzer: make run_cov_test INPUT_INST=1 INPUT_GEN_INST=${base} SHOW_COV=1"
        # make run_cov_test INPUT_INST=1 INPUT_GEN_INST="${base}" SHOW_COV=1
        # popd >/dev/null

        # echo "Iteration ${i} completed."
    )

    echo
done

echo "All ${N} iterations finished. Returned to ${START_PWD}."
