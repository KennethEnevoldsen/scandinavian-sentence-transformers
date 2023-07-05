"""
The dataset creator for the scandinavian sentence transformers.
"""

from datasets import load_dataset
import spacy

lang = "da"
dataset = "DDSC/dagw_reddit_filtered_v1.0.0"
nlp = spacy.blank(lang)
dagw = load_dataset(dataset)
