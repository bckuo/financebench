import os
from dotenv import load_dotenv

load_dotenv()

# Model Providers
import openai
import anthropic


# import ANTHROPIC TOKENIZER
CLIENT = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
ANTHROPIC_TOKENIZER = CLIENT.get_tokenizer()
openai.api_key = os.environ["OPENAI_API_KEY"]

##############################################################################
# DATASET CONFIG
##############################################################################
PATH_CURRENT = os.path.abspath(os.getcwd())
PATH_DATASET_JSONL = PATH_CURRENT + "/data/financebench_open_source_group_0.jsonl"
PATH_DOCUMENT_INFO_JSONL = PATH_CURRENT + "/data/financebench_document_information.jsonl"
PATH_RESULTS = PATH_CURRENT + "/results/"
PATH_PDFS = PATH_CURRENT + "/pdfs/"

# Choose DATASET PORTION:
# - ALL: Full Dataset
# - OPEN_SOURCE: Open Source Part (n=150)
# - CLOSED_SOURCE: Closed Source Part --> Request access at contact@patronus.ai
DATASET_PORTION = "OPEN_SOURCE"

##############################################################################
# VECTOR STORE SETUP
##############################################################################
VS_CHUNK_SIZE = 1024
VS_CHUNK_OVERLAP = 30
VS_DIR_VS = PATH_CURRENT + "/vectorstores"

##############################################################################
# SET RPM/TPM LIMITS
##############################################################################
RPM = {"openai": 500, "anthropic": 50}
TPM = {"openai": 200000, "anthropic": 50000}

##############################################################################
# MODEL CONFIGS
##############################################################################
# fmt: off
CONFIGS = [
            #{"provider": "openai",      "model_name": "gpt-4o-mini",                "eval_mode": "closedBook",          "temp": 0.01,   "max_tokens": 2048},
            #{"provider": "openai",      "model_name": "gpt-4o-mini",                "eval_mode": "inContext",           "temp": 0.01,   "max_tokens": 2048},
            #{"provider": "openai",      "model_name": "gpt-4o-mini",                "eval_mode": "inContext_reverse",   "temp": 0.01,   "max_tokens": 2048},
            #{"provider": "openai",      "model_name": "gpt-4o-mini",                "eval_mode": "oracle",              "temp": 0.01,   "max_tokens": 2048},
            #{"provider": "openai",      "model_name": "gpt-4o-mini",                "eval_mode": "oracle_reverse",      "temp": 0.01,   "max_tokens": 2048},
            #{"provider": "openai",      "model_name": "gpt-4o-mini",                "eval_mode": "singleStore",         "temp": 0.01,   "max_tokens": 2048},
            #{"provider": "openai",      "model_name": "gpt-4o-mini",                "eval_mode": "sharedStore",         "temp": 0.01,   "max_tokens": 2048},
            
            #{"provider": "anthropic",   "model_name": "claude-3-5-haiku-20241022",  "eval_mode": "closedBook",          "temp": 0.01,   "max_tokens": 2048},
            {"provider": "anthropic",   "model_name": "claude-3-5-haiku-20241022",  "eval_mode": "inContext",           "temp": 0.01,   "max_tokens": 2048},
            #{"provider": "anthropic",   "model_name": "claude-3-5-haiku-20241022",  "eval_mode": "inContext_reverse",   "temp": 0.01,   "max_tokens": 2048},
            {"provider": "anthropic",   "model_name": "claude-3-5-haiku-20241022",  "eval_mode": "oracle",              "temp": 0.01,   "max_tokens": 2048},
            {"provider": "anthropic",   "model_name": "claude-3-5-haiku-20241022",  "eval_mode": "oracle_reverse",      "temp": 0.01,   "max_tokens": 2048},
            #{"provider": "anthropic",   "model_name": "claude-3-5-haiku-20241022",  "eval_mode": "singleStore",         "temp": 0.01,   "max_tokens": 2048},
            {"provider": "anthropic",   "model_name": "claude-3-5-haiku-20241022",  "eval_mode": "sharedStore",         "temp": 0.01,   "max_tokens": 2048},

]
# fmt: on
