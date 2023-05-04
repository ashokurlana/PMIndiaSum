import torch
import argparse
from tqdm import tqdm
import pandas as pd
from rouge_score import rouge_scorer
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from transformers import MBartForConditionalGeneration, AutoConfig, PreTrainedTokenizerFast


# Required args
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True, help="path to the folder containing fine-tuned model.")
parser.add_argument("--test_data", type=str, required=True,
                    help="path to the test data in csv (four columns: text, summary, src_lang, trg_lang).")
parser.add_argument("--output_dir", type=str, required=True, help="path to save predictions")
# Default args
parser.add_argument("--tokenizer_path", type=str, default=None,
                    help="path to the (saved) tokenizer. Defaults to model_path/tokenizer.json if not set")
parser.add_argument("--max_document_len", type=int, default=1022, help="maximum input document length after tokenization") # leave 2 for eos and lang_id
parser.add_argument("--max_summary_len", type=int, default=64, help="maximum output summary length") # ideally set it the same as the one used during training

args = parser.parse_args()
if args.tokenizer_path is None:
    args.tokenizer_path = args.model_path + "/tokenizer.json"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Read test samples
assert args.test_data.split(".")[-1] == "csv"
test_samples = pd.read_csv(args.test_data, encoding='utf-8')

# Check whether the model loaded is IndicBART to determine the tokenization scheme.
is_IndicBART = AutoConfig.from_pretrained(args.model_path).tokenizer_class == "AlbertTokenizer"

# Load the tokenizer
tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer_path)
tokenizer.src_lang = None
tokenizer.trg_lang = None
eos_id = tokenizer.convert_tokens_to_ids("</s>")

# Load the model
model = MBartForConditionalGeneration.from_pretrained(args.model_path)
model.eval()
model.to(device)

# Inference
with open(args.output_dir + 'generated_predictions.txt', 'w', encoding='utf-8') as fp:
    # Not batched because we need src_lang and trg_lang, yet the test set could be multilingual.
    for i in tqdm(range(len(test_samples))):
        doc_ids = tokenizer.encode(test_samples['text'][i], add_special_tokens=False, padding=False, truncation=True,
                                   max_length=args.max_document_len)
        src_lang_id = tokenizer.convert_tokens_to_ids(test_samples['src_lang'][i])
        trg_lang_id = tokenizer.convert_tokens_to_ids(test_samples['trg_lang'][i])

        if is_IndicBART:
            doc_ids = doc_ids + [eos_id, src_lang_id]
            input = torch.tensor([doc_ids], device=device)
            model_output = model.generate(input, num_beams=4, max_length=args.max_summary_len, min_length=1, early_stopping=True,
                                          decoder_start_token_id=trg_lang_id)
        else:  # is_mBart50
            doc_ids = [src_lang_id] + doc_ids + [eos_id]
            input = torch.tensor([doc_ids], device=device)
            model_output = model.generate(input, num_beams=4, max_length=args.max_summary_len, min_length=1, early_stopping=True,
                                          forced_bos_token_id=trg_lang_id)

        # de-tokenization
        decoded_output = tokenizer.decode(model_output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        # special treatment for mni because the language id was made up as an ordinary token not a special token.
        if test_samples['trg_lang'][i] == "<2mni>" or test_samples['trg_lang'][i] == "mni_IN":
            decoded_output = decoded_output.replace(test_samples['trg_lang'][i], "").strip()
        # write generations to file
        fp.write(decoded_output + "\n")

# Scoring
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'])
with open(args.output_dir + 'generated_predictions.txt', 'r', encoding='utf-8') as fp:
    dec_out = fp.readlines()

dec_summaries = [dec_out[i].strip() for i in range(len(dec_out))]
ref_summaries = [test_samples['summary'][i] for i in range(len(dec_out))]

rouge_scores = []
for i in range(len(ref_summaries)):
    pred_sents = "\n".join(sent_tokenize(dec_summaries[i]))
    ref_sents = "\n".join(sent_tokenize(ref_summaries[i]))
    scores = scorer.score(ref_sents, pred_sents)
    rouge_scores.append(
        {'rouge-1_f': scores['rouge1'][2], 'rouge-2_f': scores['rouge2'][2], 'rouge-l_f': scores['rougeL'][2],
         'rouge-l-sum_f': scores['rougeLsum'][2]})

print("eval_rouge1: ", round(sum([item['rouge-1_f'] for item in rouge_scores]) / len(rouge_scores), 4))
print("eval_rouge2: ", round(sum([item['rouge-2_f'] for item in rouge_scores]) / len(rouge_scores), 4))
print("eval_rougeL: ", round(sum([item['rouge-l_f'] for item in rouge_scores]) / len(rouge_scores), 4))
print("eval_rougeLsum: ", round(sum([item['rouge-l-sum_f'] for item in rouge_scores]) / len(rouge_scores), 4))
