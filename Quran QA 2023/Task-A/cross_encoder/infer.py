import os
import random
import sys

import pandas as pd
from sentence_transformers import InputExample, CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryAccuracyEvaluator
from torch.utils.data import DataLoader
from transformers import HfArgumentParser

sys.path.append(os.getcwd())  # for relative imports
sys.path.append(os.getcwd() + 'kaggle/working/forked/quran-qa/Quran QA 2023/Task-A')
print(os.getcwd())

from configs.data_training_args import DataArguments
from configs.model_args import ModelArguments
from configs.training_args import CustomTrainingArguments
from configs.utils import handle_seed
from cross_encoder import make_training_triplets, make_dev_triplets, make_inference_data, save_model_to_drive, \
    prepare_my_output_dirs, config_logger, zip_inference_data
from cross_encoder.inference import infer_relevance
from data_scripts import read_docs_file, read_query_file, read_qrels_file, read_run_file
from metrics.Custom_TaskA_eval import evaluate_task_a

parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()
log_level = config_logger(training_args)
handle_seed(training_args)

print('Create Output Directory')
prepare_my_output_dirs(training_args)

doc_df = read_docs_file(data_args.doc_file)

# Create Model
model = CrossEncoder(model_args.model_name_or_path, num_labels=1, max_length=data_args.max_seq_length)

# Evaluate on Validation set
if training_args.do_eval:
    print('Preparing DEV data ...')
    # dev_qrel_df = read_qrels_file(data_args.validation_qrel_file, )
    dev_query_df = read_query_file(data_args.validation_query_file, )
    # dev_triplets = make_dev_triplets(dev_qrel_df, dev_query_df, doc_df)
    dev_infer_data = make_inference_data(dev_query_df, doc_df)

    # if data_args.max_eval_samples is not None:
    #     dev_triplets = random.choices(dev_triplets, k=data_args.max_eval_samples)

    # dev_pairs = [[q_text, doc_text] for q_text, doc_text, _ in dev_triplets]
    # dev_labels = [float(label) for _, _, label in dev_triplets]

    # Infer
    dev_infer_df = infer_relevance(model, dev_infer_data, tok_k_relevant=data_args.tok_k_relevant)

    # Save inference results
    dev_infer_df.to_csv(os.path.join(training_args.my_output_dir, "eval_inference.tsv"), sep="\t", index=False,
                        header=False)

# Evaluate on Test set
if training_args.do_predict:
    print('Preparing TEST data ...')
    test_query_df = read_query_file(data_args.test_query_file, )
    test_infer_data = make_inference_data(test_query_df, doc_df)
    # no qrel file for testing data

    # Infer
    test_infer_df = infer_relevance(model, test_infer_data, tok_k_relevant=data_args.tok_k_relevant)

    # Save inference results
    test_infer_df.to_csv(os.path.join(training_args.my_output_dir, "test_inference.tsv"), sep="\t", index=False,
                         header=False)

# Zip inference results
zip_inference_data(training_args, data_args)
