lang_pair=$1
python3 trainer.py \
        --do_train --do_eval \
        --logging_strategy "epoch" \
        --save_strategy "epoch" \
        --evaluation_strategy "epoch" \
        --overwrite_output_dir \
        --num_train_epochs 100 \
        --train_file ./data/indicbart/$lang_pair/train.csv \
        --validation_file ./data/indicbart/$lang_pair/valid.csv \
        --output_dir saved_models/indicbart/$lang_pair \
        --per_device_train_batch_size=3 \
        --per_device_eval_batch_size=3 \
        --gradient_accumulation_steps 4 \
        --my_model_type indicbartss \
        --model_name_or_path ai4bharat/IndicBARTSS

