temperature=(
    # 0.6
    # 0.7
    # 0.8
    # 0.9
    # 1

    #1.1
    #1.2
    #1.3
    #1.4
    1.5
    #1.6
    #1.7
    #1.8
)
top_k=(
    # 10
    # 20
    #30
    40
    #50
)
top_p=(
    #0.7
    0.8
    #0.9
)

: "${HARDWARE_FUZZER_DIR:=$HOME/Hardware-Fuzzer}"

for temp in "${temperature[@]}"; do
    for k in "${top_k[@]}"; do
        for p in "${top_p[@]}"; do
            (
                #ts -G 1 
                python generate.py \
                    --model_name_or_path="results/do_sample_temp1.2_repe1.2_norepe3/checkpoint-750000" \
                    --input_text="1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0,12:0,13:0,14:0,15:0,16:0,17:0,18:0,19:0,20:0,21:0,22:0" \
                    --temperature="${temp}" \
                    --top_k="${k}" \
                    --top_p="${p}" \
                    --file="generation_i/do_sample_temp${temp}_topk${k}_topp${p}.txt"
                cp "generation_i/do_sample_temp${temp}_topk${k}_topp${p}.txt" ../token2inst/generation/
                cd ../token2inst
                make -f Makefile.manual all TARGETS="do_sample_temp${temp}_topk${k}_topp${p}.txt"
                cp "cleaned_dumps/do_sample_temp${temp}_topk${k}_topp${p}.txt" "${HARDWARE_FUZZER_DIR}/sim/inst_gen"
                cd "${HARDWARE_FUZZER_DIR}/sim"
                make run_cov_test INPUT_INST=1 INPUT_GEN_INST=do_sample_temp${temp}_topk${k}_topp${p} SHOW_COV=1
            )
        done
    done
done

# -m debugpy --listen 9431 --wait-for-client 
# -m debugpy --listen 5679 --wait-for-client