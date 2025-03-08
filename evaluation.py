#!/usr/bin/env python
# coding: utf-8

# # FinanceBench: Evaluation Playground
#
# ##### (1) API Requirements
# Add the following API keys into your `.env` file:
#
# ```ruby
# OPENAI_API_KEY = 'INSERT API KEY HERE'
# ANTHROPIC_API_KEY = 'INSERT API KEY HERE'
# REPLICATE_API_TOKEN = 'INSERT API KEY HERE'
# ```
#
# ##### (2) Required Folder Structure
#
# ```bash
# |-- /
# |    |-- data/
# |    |      | -- financebench_open_source.jsonl
#      |      | -- financebench_document_information.jsonl
# |    |-- pdfs/
# |           | -- <... provided filings as PDF documents ...>
# |    |-- results/
# |    |-- vectorstores/
# |    |-- evaluation_playground.ipynb
# ```
#
#
import warnings
from langchain._api import LangChainDeprecationWarning
from pymupdf import FitzDeprecation

warnings.simplefilter("ignore", category=LangChainDeprecationWarning)
warnings.simplefilter("ignore", category=FitzDeprecation)

import pandas as pd
import time

from typing import Dict, List, Tuple
from collections import deque
from tqdm import tqdm

from model_api import get_model, get_answer
from helper import get_pdf_text, build_vectorstore_retriever
from global_var import PATH_RESULTS, CONFIGS
from global_var import PATH_DATASET_JSONL, PATH_DOCUMENT_INFO_JSONL, DATASET_PORTION


##############################################################################
# LOAD DATASET
##############################################################################
def load_dataset() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    print("-------------------------------------------------")

    # Load Full Dataset
    df_questions = pd.read_json(PATH_DATASET_JSONL, lines=True)
    df_meta = pd.read_json(PATH_DOCUMENT_INFO_JSONL, lines=True)
    df_full = pd.merge(df_questions, df_meta, on="doc_name")

    # Get all docs
    df_questions = df_questions.sort_values("doc_name")
    global ALL_DOCS
    ALL_DOCS = df_questions["doc_name"].unique().tolist()
    print(f"Total number of distinct PDF: {len(ALL_DOCS)}")

    # Select relevant dataset portion
    if DATASET_PORTION != "ALL":
        df_questions = df_questions.loc[df_questions["dataset_subset_label"] == DATASET_PORTION]
    print(f"Number of questions: {len(df_questions)}")

    # Check relevant documents
    df_questions = df_questions.sort_values("doc_name")
    docs = df_questions["doc_name"].unique().tolist()
    print(f"Number of distinct PDF: {len(docs)}")
    print("-------------------------------------------------")
    return df_questions, df_meta, df_full


##############################################################################
# EVALUATION
##############################################################################
def evaluate(model_config: Dict, df_questions: pd.DataFrame):

    # Set evaluation questions
    df_eval = df_questions

    # Get the model
    model = get_model(
        provider=model_config["provider"],
        model_name=model_config["model_name"],
        temp=model_config["temp"],
        max_tokens=model_config["max_tokens"],
    )

    # print(f"--> Evaluating: {model_config['model_name']} / {model_config['eval_mode']}")

    last_docs = None
    last_timestamp = 0
    prompt_logs = deque()
    results = []

    # Run evaluation on the model  --> Sort along doc_name to reuse retriever configs in memory
    for idx, row in tqdm(
        df_eval.sort_values("doc_name").iterrows(), total=len(df_eval), desc=f"{model_config['model_name']} / {model_config['eval_mode']}"
    ):

        # (A) Setup Context or Retriever
        if model_config["eval_mode"] == "closedBook":
            retriever = None
            context = ""

        elif model_config["eval_mode"] in ["inContext", "inContext_reverse"]:
            retriever = None
            docs = row["doc_name"]
            if not (last_docs == docs):
                pages = get_pdf_text(row["doc_name"])
                context = "\n\n".join([page.page_content for page in pages])

        elif model_config["eval_mode"] in ["oracle", "oracle_reverse"]:
            context = "\n\n".join([evidence["evidence_text_full_page"] for evidence in row["evidence"]])
            retriever = None

        elif model_config["eval_mode"] in ["singleStore", "sharedStore"]:
            context = ""
            is_all = model_config["eval_mode"] == "sharedStore"

            if is_all:
                docs = ALL_DOCS
            else:
                docs = [row["doc_name"]]

            if last_docs != docs:
                retriever, _ = build_vectorstore_retriever(docs=docs, is_all=is_all)
                last_docs = docs

        else:
            raise ValueError("Unknown 'eval_mode'!")

        # (B) Model Call
        (answer, retrieved_documents, last_timestamp, chars_cutoff) = get_answer(
            provider=model_config["provider"],
            model=model,
            eval_mode=model_config["eval_mode"],
            question=row["question"],
            context=context,
            retriever=retriever,
            timestamp=last_timestamp,
            prompt_logs=prompt_logs,
        )

        # (C) Bookkeeping
        if model_config["eval_mode"] in ["inContext", "inContext_reverse"]:
            results.append(
                {
                    **model_config,
                    "financebench_id": row["financebench_id"],
                    "question": row["question"],
                    "gold_answer": row["answer"],
                    "model_answer": answer,
                    "retrieved_documents": retrieved_documents,
                    "chars_cutoff": chars_cutoff,
                }
            )
        else:
            results.append(
                {
                    **model_config,
                    "financebench_id": row["financebench_id"],
                    "question": row["question"],
                    "gold_answer": row["answer"],
                    "model_answer": answer,
                    "retrieved_documents": retrieved_documents,
                }
            )

    df_results = pd.DataFrame(results)
    df_results.to_csv(PATH_RESULTS + "/" + model_config["model_name"] + "_" + model_config["eval_mode"] + ".csv")


def main():

    df_questions, _, _ = load_dataset()

    openai_configs = [config for config in CONFIGS if config["provider"] == "openai"]
    anthropic_configs = [config for config in CONFIGS if config["provider"] == "anthropic"]

    tqdm.get_lock().locks = []

    # https://github.com/tqdm/tqdm/issues/461
    for config in openai_configs:
        evaluate(config, df_questions)
        # For clearing TPM
        print("Waiting 60 sec for the next config...")
        time.sleep(60)


if __name__ == "__main__":
    main()
