## PMIndiaSum Corpus

### Download
https://drive.google.com/file/d/1KkJ4UbDprtoeeCA6wxfMknWXykYgnLUY/view?usp=sharing

The corpus is released under the CC-BY-4.0, in other words the corpus can be freely shared and adapted as long as appropriate credit is given. https://creativecommons.org/licenses/by/4.0/

### Overview
The data comes in `.jsonl` format, where each JSON object correspond to all the data extracted from a single article. Each JSON object has the following fields:
```
{article_number,
 ${lang1}_text,
 ${lang1}_summary,
 ${lang1}_url,
 ${lang2}_text,
 ${lang2}_summary,
 ${lang2}_url,
 ...,
 split}
```
where `${lang}_text`, `${lang}_summary`, and `${lang}_url` correspond to the text (document), summary (headline), and the source URL for each language `${lang}` available for that article. `split` indicates the train/valid/test split of that entire article. Texts and summaries can be paired to form cross-lingual data pairs.

Unless you have a specific need, we request you to respect the split decision to prevent test data leakage, especially for multilingual models. You could refer to our paper for a detailed explanation.

### Preparation
If you use our model code, you can easily run `prepare_data.py` to prepare data compatible with IndicBART or mBART. For any given language pair, the script creates `train.csv`, `valid.csv` and `test.csv` files, with four columns: `text`, `summary`, `src_lang`, `trg_lang`. You could also process the data in any way that suits you.

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
