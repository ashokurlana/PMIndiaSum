## Train

For a given language pair ${lang_pair} (e.g `hi-hi`, `en-te`, or `all`, see `../data/` for download and preparation details), `train_indicbart.sh` and `train_mbart.sh` are useful to perform monolingual, cross-lingual, or multilingual finetuning with IndicBART and mBART respectively. The finetuned model will be saved in the specified output directory.

#### IndicBART
```
sh train_indicbart.sh ${lang_pair}
```

#### mBART

```
sh train_mbart.sh ${lang_pair}
```

## Test

For a given language pair, `test.sh` file obtains the predictions of the model and corresponding ROUGE scores. We utilized the multilingual ROUGE implemented in [XL-Sum](https://github.com/csebuetnlp/xl-sum/tree/master/multilingual_rouge_scoring). 

```
# MODEL_PATH is the fine-tuned model type: either "mbart" or "indicbart".
# TEST_DATA is the path to the test data file. Refer to ../data/ for more details
# OUTPUT_DIR is the directory where the files will be written to.

python3 tester.py \
  --model_path ${MODEL_PATH} \
  --test_data ${TEST_DATA} \
  --output_dir ${OUTPUT_DIR}

```
Optionally you can inspect and directly run `test.sh`.
