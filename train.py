from pathlib import Path

import nltk
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, datasets, losses, models
from torch.utils.data import DataLoader
import wandb
# nltk.download("punkt")

wandb.init(project="dfm-sentence-encoder")

repo_dir = Path(__file__).resolve().parent
dataset_path = repo_dir / "data" / "dfm_paragraphs"

model_name = "vesteinn/DanskBERT"
word_embedding_model = models.Transformer(model_name)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


paragraphs = load_dataset("KennethEnevoldsen/dfm-paragraphs", split="train")
# shuffle
paragraphs = paragraphs.shuffle()
train_sentences = paragraphs[:100_000]["text"]


# Create the special denoising dataset that adds noise on-the-fly
train_dataset = datasets.DenoisingAutoEncoderDataset(train_sentences)

# DataLoader to batch your data
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Use the denoising auto-encoder loss
train_loss = losses.DenoisingAutoEncoderLoss(
    model, decoder_name_or_path=model_name, tie_encoder_decoder=True
)

# Call the fit method
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    weight_decay=0,
    scheduler="constantlr",
    optimizer_params={"lr": 3e-5},
    show_progress_bar=True,
    checkpoint_save_steps=100,

)

models_path = repo_dir / "models"
model.save(models_path / "dfm-sentence-encoder-medium")
