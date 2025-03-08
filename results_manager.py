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

N_GROUPS = 5

RESULTS_ALL_PATH = PATH_CURRENT + "/results/all/"
RESULTS_GROUP_PATH = [PATH_CURRENT + f"/results/group_{i}/" for i in range(N_GROUPS)]
RESULTS_PATH = PATH_CURRENT + "/results/"


def load_dataset() -> List[pd.DataFrame]:

    df_questions = pd.read_json(PATH_DATASET_JSONL, lines=True)
    path_dataset_partial = [f"{PATH_CURRENT}/{PATH_DATASET_PREFIX}_group_{i}.jsonl" for i in range(N_GROUPS)]
    df_question_groups = [pd.read_json(path_dataset_partial[i], lines=True) for i in range(N_GROUPS)]

    return df_questions, df_question_groups


def merge_selected_groups(selected_groups: List[int]):

    df_questions, df_question_groups = load_dataset()

    df_question_groups = pd.concat([df_question_groups[i] for i in selected_groups])
    postfix = "".join([str(i) for i in sorted(selected_groups)])

    df_question_groups.to_json(f"{PATH_DATASET_PREFIX}_group_{postfix}.jsonl", orient="records", lines=True)


def sort_and_save_dataframe(path: str):

    df = pd.read_csv(path)
    df = df[["financebench_id", "model_name", "eval_mode", "temp", "question", "gold_answer", "model_answer"]]
    df = df.sort_values("financebench_id")
    path = path.replace(".csv", ".jsonl")

    df.to_json(path, orient="records", lines=True)


def merge_and_save_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, path: str):

    df = pd.concat([df1, df2])
    df = df.sort_values("financebench_id")

    df.to_json(path, orient="records", lines=True)


def merge_0_and_1234(model: str, eval_mode: str):

    config = f"{model}_{eval_mode}"

    df_1234 = pd.read_json(RESULTS_PATH + f"{config}_group_1234.jsonl", lines=True)
    df_0 = pd.read_json(RESULTS_GROUP_PATH[0] + f"{config}_group_0.jsonl", lines=True)

    merge_and_save_dataframes(df_1234, df_0, RESULTS_ALL_PATH + f"{config}_all.jsonl")


def save_group_results_in_all(model: str, eval_mode: str, g_idx: int, df_question_groups: List[pd.DataFrame]):

    config = f"{model}_{eval_mode}"

    df_all = pd.read_json(RESULTS_ALL_PATH + f"{config}_all.jsonl", lines=True)
    df_group = df_all[df_all["financebench_id"].isin(df_question_groups[g_idx]["financebench_id"])]
    df_group.sort_values("financebench_id")

    df_group.to_json(RESULTS_GROUP_PATH[g_idx] + f"{config}_group_{g_idx}_.jsonl", orient="records", lines=True)


def main():
    df_questions, df_question_groups = load_dataset()
    
    # save_group_results_in_all("claude-3-5-haiku-20241022", "closedBook", 0, df_question_groups)
    save_group_results_in_all("gpt-4o-mini", "closedBook", 0, df_question_groups)


if __name__ == "__main__":
    main()
