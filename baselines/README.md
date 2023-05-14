

We use a modified fork of [huggingface transformers](https://github.com/huggingface/transformers) for our experiments.


## Train

For a given language pair, `train_indicbart.sh` and `train_mbart.sh` are useful to perform mono-, cross-, and multi-lingual finetuning with IndicBART and mBART respectively. The finetuned model will be saved in the specified output directory. You can get the `train.csv` and `valid.csv` using the [prepare_data.py](https://github.com/ashokurlana/PMIndiaSum/blob/main/data/prepare_data.py) script. To perform multilingual fine-tuning, you can combine all the available language pairs data and perform the finetuning. 

#### IndicBART
```
sh train_indicbart.sh
```

#### mBART

```
sh train_mbart.sh
```


## Test

For a given language pair, `test.sh` file obtains the predictions of the model and corresponding ROUGE scores. We utilized the multi-lingual score mentioned in the [XL-Sum git repo](https://github.com/csebuetnlp/xl-sum/tree/master/multilingual_rouge_scoring). 

```
# MODEL_PATH can be finetuned model path of "mbart" or "indicbart".
# TEST_DATA is the path to the test data file. Which can be obtained using the prepare_data.py
# OUTPUT_DIR is the directory where the files will be written to.

python3 tester.py \
  --model_path ${MODEL_PATH} \
  --test_data ${TEST_DATA} \
  --output_dir ${OUTPUT_DIR}


```
Optionally you can run `test.sh` by passing the necessary arguemnts to perform the testing.
