temperature=(
    # 0.1
    # 0.6
    # 0.7
    # 0.8
    # 0.9
    # 1

    # 1.1
    # 1.2
    # 1.3
    # 1.4
    # 1.5
    # 1.6
    # 1.7
    # 1.8
    # 2
    3
    # 5
)
top_k=(
    # 10
    # 20
    # 30
    # 40
    # 50
    60
)
top_p=(
    # 0.6
    # 0.7
    # 0.8
    0.9
)
for temp in "${temperature[@]}"; do
    for k in "${top_k[@]}"; do
        for p in "${top_p[@]}"; do
            #ts -G 1 
            python generate_batch.py \
                --model_name_or_path="results/20251014/checkpoint-2600920" \
                --input_text="1:2,2:3,3:2,4:1,5:2,6:2,7:2,8:1,9:5,10:4,11:8,12:1,13:4,14:1,15:1,16:2,17:2,18:1,19:1,20:1,21:1,22:2" \
                --temperature="${temp}" \
                --top_k="${k}" \
                --top_p="${p}" \
                --file="generation_batch/do_sample_temp${temp}_topk${k}_topp${p}.txt" \
                --batch_size=4096
        done
    done
done

# -m debugpy --listen 9431 --wait-for-client 
# -m debugpy --listen 5679 --wait-for-client