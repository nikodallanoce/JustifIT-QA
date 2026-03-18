import re
from typing import Any

import datasets
import pandas as pd


def remove_nones_from_dict(inp_dict_lst: list[dict[str, Any]]) -> dict:
    for i in range(len(inp_dict_lst)):
        a_dict = inp_dict_lst[i]
        keys_to_remove = [k for k, v in a_dict.items() if v is None]
        for k in keys_to_remove:
            del a_dict[k]
        inp_dict_lst[i] = a_dict

    return {"relevant_document_sections": inp_dict_lst}


def get_text_from_dict_indices(corpus_df: pd.DataFrame, rel_docs_dict: dict[str, list[tuple[int, int]]]):
    text_docs = {}
    for doc_id in rel_docs_dict:
        doc_id = int(doc_id)
        doc_content = corpus_df[corpus_df['id'] == doc_id]['content'].item()
        # doc_content = corpus_df.loc[doc_id]['content']
        text_lst = []
        for ind_tuple in rel_docs_dict[str(doc_id)]:
            text_lst.append(doc_content[ind_tuple[0]:ind_tuple[1]])
        text_docs[doc_id] = text_lst  # "\n\n".join(text_lst)
    return text_docs


def retrieve_doc_txt_given_ids_lst(questions_df: pd.DataFrame, corpus_df: pd.DataFrame):
    return questions_df['relevant_doc_ids'].map(
        lambda idx_lst: [corpus_df[corpus_df['id'] == idx]['content'].item() for idx in idx_lst])


def get_rel_docs(questions_df: pd.DataFrame, text_span_bounds_feat_name: str):
    return questions_df[text_span_bounds_feat_name].map(
        lambda a_dict: [int(k) for k, v in a_dict.items() if v is not None])


def get_cleaned_question_dataset(questions_ds: datasets.Dataset, corpus_ds: datasets.Dataset) -> pd.DataFrame:
    questions_ds = questions_ds.map(remove_nones_from_dict, input_columns=['relevant_document_sections'],
                                    batched=True)
    questions_df = questions_ds.to_pandas()
    questions_df['relevant_document_sections'] = questions_df['relevant_document_sections'].map(
        lambda a_dict: {k: v for k, v in a_dict.items() if v is not None})
    corpus_df: pd.DataFrame = corpus_ds.to_pandas()
    questions_df['relevant_text_sections'] = questions_df['relevant_document_sections'].map(
        lambda a_dict: get_text_from_dict_indices(corpus_df, a_dict))
    questions_df['relevant_doc_ids'] = get_rel_docs(questions_df, 'relevant_document_sections')
    questions_df["relevant_doc_txt"] = retrieve_doc_txt_given_ids_lst(questions_df, corpus_df)
    questions_df.rename(columns={'relevant_document_sections': 'relevant_document_spans'}, inplace=True)
    return questions_df


def extract_evaluation(text: str) -> tuple[str, float]:
    """
    Extracts the value associated with the "evaluation" key from a possibly malformed JSON-like string.
    Returns the value as a string, or None if not found.
    """
    # Regex explanation:
    # - (?i) makes it case-insensitive (matches "Evaluation", "evaluation", etc.)
    # - looks for "evaluation" followed by ":" and optional spaces
    # - captures the value between quotes (either single or double)
    result_mapping = {"compliant": 1.0, "semi-compliant": 0.5, "error": 0.0, "non-compliant": 0.0,
                      "conforme": 1.0, "parzialmente conforme": 0.5, "non conforme": 0.0}
    pattern = r'(?i)"evaluation"\s*:\s*["\']([^"\']+)["\']'

    match = re.search(pattern, text)
    result = "ERROR"
    if match:
        result = match.group(1).strip().lower()
    num_out = result_mapping.get(result, 0)
    if result == "ERROR" or result not in result_mapping:
        print("ERROR IN EXTRACTING EVALUATION!")
    return (result, float(num_out))
