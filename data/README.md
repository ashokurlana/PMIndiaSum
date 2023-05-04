### Data Preparation

## PMIndiaSum dataset with split information

https://drive.google.com/file/d/1KkJ4UbDprtoeeCA6wxfMknWXykYgnLUY/view?usp=sharing


## Prepare data to run IndicBART

The script creates the train.csv, valid.csv and test.csv files with columns text, summary, src_lang, trg_lang

```
python3 prepare_data.py --model_type $MODEL_NAME --data_file $DATA_PATH --lang_pair $LANG --output_dir ./
```

