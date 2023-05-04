MODEL=mbart
#MODEL=indicbart


export CUDA_VISIBLE_DEVICES=0,1

mkdir -p test_outputs

for lang_pair in as-as bn-bn en-en gu-gu hi-hi kn-kn ml-ml mni-mni mr-mr or-or pa-pa ta-ta te-te
do

python3 tester.py \
    --model_path ./saved_models/${MODEL}/${lang_pair} \
    --test_data ./data/${MODEL}/${lang_pair}/test.csv \
    --output_dir ./test_outputs/${MODEL}/${lang_pair}
echo "-----done testing ${lang_pair}, see results above----"
done
