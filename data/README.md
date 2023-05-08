## Data Preparation

## PMIndiaSum dataset with split information

https://drive.google.com/file/d/1KkJ4UbDprtoeeCA6wxfMknWXykYgnLUY/view?usp=sharing


## Prepare data to run IndicBART

For a given language pair, `prepare_data.py` creates `train.csv`, `valid.csv` and `test.csv` files, with four columns: `text`, `summary`, `src_lang`, `trg_lang`

```
python3 prepare_data.py --model_type ${MODEL_TYPE} --data_file ${DATA_PATH} --lang_pair ${LANG_PAIR} --output_dir ${OUTPUT_DIR}
```
