
mkdir -p ./saved_models/indicbart/

export CUDA_VISIBLE_DEVICES=0,1,2,3,4

for lang_pair in as-as bn-bn gu-gu hi-hi kn-kn ml-ml mni-mni mr-mr or-or pa-pa ta-ta te-te en-en
do
python3 trainer.py \
        --do_train --do_eval \
        --logging_strategy "epoch" \
        --save_strategy "epoch" \
        --evaluation_strategy "epoch" \
        --overwrite_output_dir \
        --num_train_epochs 100 \
        --train_file ./data/indicbart/${lang_pair}/train.csv \
        --validation_file ./data/indicbart/${lang_pair}/valid.csv \
        --output_dir saved_models/indicbart/${lang_pair} \
        --per_device_train_batch_size=3 \
        --per_device_eval_batch_size=3 \
        --gradient_accumulation_steps 4 \
        --my_model_type indicbartss \
        --model_name_or_path ai4bharat/IndicBARTSS
done
