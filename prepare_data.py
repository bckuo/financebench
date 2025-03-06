import os
import pandas as pd
import numpy as np  # Add numpy import

from typing import List, Dict, Deque, Tuple

##############################################################################
# DATASET CONFIG
##############################################################################
PATH_CURRENT = os.path.abspath(os.getcwd())
PATH_DATASET_JSONL = PATH_CURRENT + "/data/financebench_open_source.jsonl"
PATH_DOCUMENT_INFO_JSONL = PATH_CURRENT + "/data/financebench_document_information.jsonl"

# Choose DATASET PORTION:
# - ALL: Full Dataset
# - OPEN_SOURCE: Open Source Part (n=150)
# - CLOSED_SOURCE: Closed Source Part --> Request access at contact@patronus.ai
DATASET_PORTION = "OPEN_SOURCE"

##############################################################################
# LOAD DATASET
##############################################################################
# 
# Set Number of Groups
N_GROUPS = 5

# Load Full Dataset
df_questions = pd.read_json(PATH_DATASET_JSONL, lines=True)
df_questions = df_questions.sort_values("doc_name")

# Split Dataset by Question Type
categorized_questions = {}
categorized_questions["metrics-generated"] = df_questions[df_questions["question_type"] == "metrics-generated"]
categorized_questions["domain-relevant"] = df_questions[df_questions["question_type"] == "domain-relevant"]
categorized_questions["novel-generated"] = df_questions[df_questions["question_type"] == "novel-generated"]

##############################################################################
# SPLIT DATAFRAME INTO N EQUAL GROUPS RANDOMLY
##############################################################################

def split_dataframe_randomly(df: pd.DataFrame, n: int) -> List[pd.DataFrame]:
    """
    Splits a DataFrame into n equal groups randomly using iloc.
    
    :param df: The DataFrame to split.
    :param n: The number of groups to split the DataFrame into.
    :return: A list of DataFrames.
    """
    df_shuffled = df.sample(frac=1, random_state=0).reset_index(drop=True)
    group_size = len(df) // n
    groups = [df_shuffled.iloc[i*group_size:(i+1)*group_size] for i in range(n)]
    return groups

# Split DataFrames
categorized_questions_groups = {}
for key, df in categorized_questions.items():
    categorized_questions_groups[key] = split_dataframe_randomly(df, N_GROUPS)

# Merge questions by index
merged_questions_groups: List[pd.DataFrame] = []
for i in range(N_GROUPS):
    merged_questions = pd.concat([categorized_questions_groups[key][i] for key in categorized_questions_groups.keys()])
    merged_questions.sample(frac=1, random_state=0).reset_index(drop=True)
    merged_questions_groups.append(merged_questions)

# Save DataFrames
for i, df in enumerate(merged_questions_groups):
    df.to_json(f"data/financebench_open_source_group_{i}.jsonl", orient="records", lines=True)


