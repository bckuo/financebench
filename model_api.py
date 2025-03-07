from global_var import ANTHROPIC_TOKENIZER, RPM, TPM

import os
import time
from typing import Deque, Tuple

# LangChain Stuff
from langchain.chains.retrieval_qa.base import RetrievalQA

# LangChain Model Wrappers
from langchain.chat_models.base import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# Model Providers
import tiktoken


##############################################################################
# MODEL + CALL HANDLERS
##############################################################################
def sleep_if_reach_RPM(rpm: int, timestamp: float) -> None:

    # For not exceeding the RPM
    elapsed_time = time.time() - timestamp
    min_elapsed_time = 60 / rpm
    required_sleep_duration = min_elapsed_time - elapsed_time
    if required_sleep_duration > 0:
        print(f"[RPM] Sleeping for {required_sleep_duration} seconds")
        time.sleep(required_sleep_duration)
    return time.time()


def sleep_if_reach_TPM(tpm: int, context_tokens: int, token_usage: Deque[Tuple[float, int]]) -> None:

    tokens = context_tokens
    exceeded_limit_usage = None
    for usage in token_usage:
        tokens += usage[1]
        if tokens > tpm:
            exceeded_limit_usage = usage
            break

    if exceeded_limit_usage:
        required_sleep_duration = 60 - (time.time() - exceeded_limit_usage[0])
        if required_sleep_duration > 0:
            print(f"[TPM] Sleeping for {required_sleep_duration} seconds")
            time.sleep(required_sleep_duration)
        # Remove all token usages until the last exceeded limit
        usage = token_usage.pop()
        while usage != exceeded_limit_usage:
            usage = token_usage.pop()


def get_max_context_length(prompt, anthropic_cutoff=47500, openai_cutoff=57000):

    # (0) Check Anthropic Tokenizer
    tokens_anthropic = ANTHROPIC_TOKENIZER.encode(prompt)
    nb_tokens_anthropic = len(tokens_anthropic)
    number_of_chars_anthropic = len(prompt)

    if nb_tokens_anthropic > anthropic_cutoff:
        tokens_anthropic_tokens = tokens_anthropic.tokens
        token_lengths_anthropic = [len(token) for token in tokens_anthropic_tokens]
        number_of_chars_anthropic = sum(token_lengths_anthropic[:anthropic_cutoff])

    # (1) Check OpenAI Tokenizer
    # tokenizer_openai = tiktoken.encoding_for_model("gpt-4-1106-preview")
    tokenizer_openai = tiktoken.encoding_for_model("gpt-4o-mini")
    tokens_openai = tokenizer_openai.encode(prompt)
    nb_tokens_openai = len(tokens_openai)
    number_of_chars_openai = len(prompt)

    if nb_tokens_openai > openai_cutoff:
        tokens_openai_tokens = [tokenizer_openai.decode_single_token_bytes(token) for token in tokens_openai]
        token_lengths_openai = [len(token) for token in tokens_openai_tokens]
        number_of_chars_openai = sum(token_lengths_openai[:openai_cutoff])

    # Cut prompt depending on minimal length limit
    number_of_chars = min(number_of_chars_anthropic, number_of_chars_openai)

    if number_of_chars == len(prompt):
        return number_of_chars, max(nb_tokens_anthropic, nb_tokens_openai)
    elif number_of_chars == number_of_chars_anthropic:
        return number_of_chars, min(nb_tokens_anthropic, anthropic_cutoff)
    else:
        return number_of_chars, min(nb_tokens_openai, openai_cutoff)


def get_model(provider="openai", model_name="gpt-4", temp=0.01, max_tokens=2048) -> BaseChatModel:

    if provider == "openai":
        return ChatOpenAI(model_name=model_name, temperature=temp, max_tokens=max_tokens)

    elif provider == "anthropic":
        return ChatAnthropic(model=model_name, temperature=temp, max_tokens_to_sample=max_tokens, anthropic_api_key=os.environ["ANTHROPIC_API_KEY"])

    else:
        return None


def get_answer(
    provider: str,
    model: BaseChatModel,
    eval_mode: str,
    question: str,
    context: str,
    retriever: RetrievalQA,
    timestamp: float,
    token_usage: Deque[Tuple[float, int]],
) -> str:

    timestamp = sleep_if_reach_RPM(RPM[provider], timestamp)

    retrieved_documents = []
    chars_cutoff = 0
    if eval_mode == "closedBook":
        prompt = f"Answer this question: {question}"
        answer = model.invoke(prompt)

    elif eval_mode in ["inContext", "inContext_reverse"]:

        # Context Cutoff to satisfy max tokens
        max_number_of_chars, context_tokens = get_max_context_length(context)
        chars_cutoff = len(context) - max_number_of_chars
        context = context[:max_number_of_chars]

        if eval_mode == "inContext":
            prompt = f"Answer this question: {question} \nHere is the relevant filing that you need to answer the question:\n[START OF FILING] {context} [END OF FILING]"
        else:
            prompt = f"Context:\n[START OF FILING] {context} [END OF FILING]\n\n Answer this question: {question}\n"

        sleep_if_reach_TPM(TPM[provider], context_tokens, token_usage)
        timestamp = time.time()

        answer = model.invoke(prompt)

        token_usage.appendleft((time.time(), context_tokens))

    elif eval_mode == "oracle":
        prompt = f"Answer this question: {question} \nHere is the relevant evidence that you need to answer the question:\n[START OF FILING] {context} [END OF FILING]"
        answer = model.invoke(prompt)

    elif eval_mode == "oracle_reverse":

        prompt = f"Context:\n[START OF FILING] {context} [END OF FILING\n\n Answer this question: {question} \n"
        answer = model.invoke(prompt)

    elif eval_mode == "singleStore" or eval_mode == "sharedStore":

        # Retrieval-only mode if model=None (No LLM calls, only queries in VectorDB)
        if not model:
            prompt = f"{question}"
            s = retriever.invoke(prompt)
            return ("", s)

        else:

            # Don't add a question prefix as RetrievalQA will do some automatic prompt wrapping
            # --> This can replace by more advanced Retrieval Strategies
            prompt = f"{question}"
            qa = RetrievalQA.from_chain_type(
                llm=model,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
            )
            s = qa(prompt)

            answer = s["result"]
            retrieved_documents = s["source_documents"]
    return (answer, retrieved_documents, timestamp, chars_cutoff)
