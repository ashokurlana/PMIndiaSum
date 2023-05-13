## Data
### PMIndiaSum download
https://drive.google.com/file/d/1KkJ4UbDprtoeeCA6wxfMknWXykYgnLUY/view?usp=sharing

The data comes in `.jsonl` format, where each json object corresponds to document (text)-summary pairs for all available languages extracted from a single article. Each json object has fields: `${lang}_text`, `${lang}_summary`, and `${lang}_url` for each language ${lang} available. Texts (documents) and summaries can be paired arbitrarily to form cross-lingual data pairs.

The corpus is released under the CC-BY-4.0, in other words the corpus can be freely shared and adapted as long as appropriate
credit is give. https://creativecommons.org/licenses/by/4.0/

### Prepare data for MBART or IndicBART training

If you use our code, you can run our `prepare_data.py` script to prepare data compatible with IndicBART or mBART. For a given language pair, the script creates `train.csv`, `valid.csv` and `test.csv` files, with four columns: `text`, `summary`, `src_lang`, `trg_lang`

```
# MODEL_TYPE can be "mbart" or "indicbart".
# DATA_FILE is the path to the data file.
# LANG_PAIR is the language pair of interest, e.g. "en-en", "as-bn", or "all" to train a single multilingual model.
# OUTPUT_DIR is the directory where the files will be written to.

python3 prepare_data.py \
  --model_type ${MODEL_TYPE} \
  --data_file ${DATA_FILE} \
  --lang_pair ${LANG_PAIR} \
  --output_dir ${OUTPUT_DIR}
```
