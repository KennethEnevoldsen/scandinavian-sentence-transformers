"""
The dataset creator for the scandinavian sentence transformers.
"""

from functools import partial
from pathlib import Path

import spacy
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer  # type: ignore


def is_non_duplicate(example: dict):
    didnt_pass_quality_filter = example["is_13_gram_duplicate"] is None
    if didnt_pass_quality_filter:
        return False
    is_duplicate = example["is_13_gram_duplicate"]["duplicate"]
    return not is_duplicate


def example_to_paragraphs(doc, tokenizer, max_length: int) -> list[str]:
    tokens = tokenizer(doc.text)
    if len(tokens[0]) < max_length:
        return [doc.text]

    paragraphs = []
    min_idx = 0
    for sent in doc.sents:
        paragraph = doc[min_idx : sent.end]
        tokens = tokenizer(paragraph.text)

        # if the paragraph is longer than the max length
        # we use the previous paragraph and start a new one
        if len(tokens[0]) > max_length:
            sent_is_longer_than_max_length = sent.start == min_idx
            if sent_is_longer_than_max_length:
                min_idx = sent.end  # skip the sentence
                continue
            paragraphs.append(doc[min_idx : sent.start].text)
            min_idx = sent.start

    # add the last paragraph
    paragraph = doc[min_idx:].text
    if paragraph.strip():
        paragraphs.append(paragraph)
    return paragraphs


def examples_to_paragraph(
    examples: dict, lang: str, model_name: str, max_token_length: int
) -> dict:
    nlp = spacy.blank(lang)
    nlp.add_pipe("sentencizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    new_examples = {
        "doc_id": [],
        "source": [],
        "paragraph_id": [],
        "text": [],
    }
    docs = nlp.pipe(examples["text"])
    for doc_id, source, doc in zip(examples["doc_id"], examples["source"], docs):
        paragraphs = example_to_paragraphs(
            doc=doc, tokenizer=tokenizer, max_length=max_token_length
        )

        for n, paragraph in enumerate(paragraphs):
            new_examples["doc_id"].append(doc_id)
            new_examples["source"].append(source)
            new_examples["paragraph_id"].append(n)
            new_examples["text"].append(paragraph)

    return new_examples


if __name__ == "__main__":
    lang = "da"
    dataset = "DDSC/dagw_reddit_filtered_v1.0.0"

    repo_path = Path(__file__).parent.parent.parent
    data_path = repo_path / "data"

    data_path.mkdir(exist_ok=True)
    dagw: DatasetDict = load_dataset(dataset, cache_dir=str(data_path))  # type: ignore
    dagw: Dataset = dagw["train"]  # type: ignore
    model_name = "vesteinn/DanskBERT"
    max_sent_tokens = 128

    dagw = dagw.filter(is_non_duplicate, num_proc=8)

    examples = dagw[0:2]

    test = examples_to_paragraph(
        examples, lang=lang, model_name=model_name, max_token_length=max_sent_tokens
    )

    _examples_to_paragraph = partial(
        examples_to_paragraph,
        lang=lang,
        model_name=model_name,
        max_token_length=max_sent_tokens,
    )

    # remove all columns but text, doc_id and source
    columns_to_keep = ["text", "doc_id", "source"]
    columns_to_remove = set(dagw.column_names) - set(columns_to_keep)
    dagw = dagw.remove_columns(columns_to_remove)


    dagw = dagw.map(
        _examples_to_paragraph,
        batched=True,
        batch_size=1000,
        num_proc=8,
    )