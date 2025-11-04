# ts -G 1 \
python clm_train.py \
    --dataset_name="AsahiRokkaLOCK/testdata_new" \
    --config_name="facebook/opt-125m" \
    --do_train \
    --do_eval \
    --do_generate \
    --num_train_epochs=10 \
    --per_device_train_batch_size=32 \
    --per_device_eval_batch_size=2048 \
    --gradient_accumulation_steps=1 \
    --learning_rate=5e-5 \
    --output_dir="results/20251014" \
    --run_name="do_sample_temp1.2_repe1.2_norepe3" \
    --eval_strategy="steps" \
    --eval_steps=500000 \
    --save_strategy="steps" \
    --save_steps=500000 \
    --save_total_limit=10 \
    --logging_strategy="steps" \
    --logging_steps=1000 \
    --lr_scheduler_type="cosine" \
    --do_sample \
    --temperature=1.2 \
    --repetition_penalty=1.2 \
    --no_repeat_ngram_size=3 \
    --fp16 \
    --fp16_full_eval \
    --eval_accumulation_steps=1 \
    --dataloader_num_workers=12 \
    --dataloader_pin_memory=True \
    --torch_compile \
    --report_to=none \
    --optim="adamw_torch_fused" \
    --tf32=True 

# ts -G 1
# -m debugpy --listen 5679 --wait-for-client
# no/steps/epoch
# penalty_alpha 0.6
# temperature 1.2
# warmup_ratio=0.05 \
# --overwrite_output_dir \