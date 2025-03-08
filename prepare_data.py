import os
import pandas as pd

from typing import List, Dict

##############################################################################
# DATASET CONFIG
##############################################################################
PATH_CURRENT = os.path.abspath(os.getcwd())
PATH_DATASET_PREFIX = "data/financebench_open_source"
PATH_DATASET_JSONL = f"{PATH_CURRENT}/{PATH_DATASET_PREFIX}.jsonl"
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


# Split DataFrame into N Groups Randomly
def split_dataframe_randomly(df: pd.DataFrame, n: int) -> List[pd.DataFrame]:

    df_shuffled = df.sample(frac=1, random_state=0).reset_index(drop=True)
    group_size = len(df) // n
    groups = [df_shuffled.iloc[i * group_size : (i + 1) * group_size] for i in range(n)]

    return groups


# Merge questions by index
def merge_dataframes(categorized_questions_groups: Dict[str, List[pd.DataFrame]], n: int) -> List[pd.DataFrame]:

    merged_questions_groups: List[pd.DataFrame] = []

    for i in range(n):
        merged_questions = pd.concat([categorized_questions_groups[key][i] for key in categorized_questions_groups.keys()])
        merged_questions.sample(frac=1, random_state=0).reset_index(drop=True)
        merged_questions_groups.append(merged_questions)

    return merged_questions_groups


def split_and_save_dataframe():

    # Load Full Dataset
    df_questions = pd.read_json(PATH_DATASET_JSONL, lines=True)
    df_questions = df_questions.sort_values("doc_name")

    # Split Dataset by Question Type
    categorized_questions = {}

    categorized_questions["metrics-generated"] = df_questions[df_questions["question_type"] == "metrics-generated"]
    categorized_questions["domain-relevant"] = df_questions[df_questions["question_type"] == "domain-relevant"]
    categorized_questions["novel-generated"] = df_questions[df_questions["question_type"] == "novel-generated"]

    # Split DataFrames
    categorized_questions_groups: Dict[str, List[pd.DataFrame]] = {}

    for key, df in categorized_questions.items():
        categorized_questions_groups[key] = split_dataframe_randomly(df, N_GROUPS)

    # Merge DataFrames
    merged_questions_groups = merge_dataframes(categorized_questions_groups, N_GROUPS)

    # Save DataFrames
    for i, df in enumerate(merged_questions_groups):
        df.to_json(f"{PATH_DATASET_PREFIX}_group_{i}.jsonl", orient="records", lines=True)


def load_dataset() -> List[pd.DataFrame]:

    path_dataset_partial = [f"{PATH_CURRENT}/{PATH_DATASET_PREFIX}_group_{i}.jsonl" for i in range(N_GROUPS)]
    df_question_groups = [pd.read_json(path_dataset_partial[i], lines=True) for i in range(N_GROUPS)]

    return df_question_groups


def merge_selected_groups(selected_groups: List[int]):

    df_question_groups = load_dataset()

    postfix = "".join([str(i) for i in sorted(selected_groups)])

    df_question_groups = pd.concat([df_question_groups[i] for i in selected_groups])
    df_question_groups.to_json(f"{PATH_DATASET_PREFIX}_group_{postfix}.jsonl", orient="records", lines=True)


def main():
    
    selected_groups = [1, 2, 3, 4]
    merge_selected_groups(selected_groups)


if __name__ == "__main__":
    main()
