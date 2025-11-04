python generate_parallel_old.py \
  --model_name_or_path="results/do_sample_temp1.2_repe1.2_norepe3/checkpoint-750000" \
  --input_text="1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0,12:0,13:0,14:0,15:0,16:0,17:0,18:0,19:0,20:0,21:0,22:0" \
  --temperature=1.5 \
  --top_k=40 \
  --top_p=0.8 \
  --batch_size=256 \
  --output_dir="generation_batch/"
