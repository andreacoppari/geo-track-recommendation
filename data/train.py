import torch
import torch.nn as nn

from transformers import BertModel, BertTokenizer

from torch.optim import Adam
from .model import MTS

EPOCHS = 10

bert_model_name = "bert-base-uncased"
bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert_encoder = BertModel.from_pretrained(bert_model_name)

text_embedding_dim = bert_encoder.config.hidden_size
geodata_embedding_dim = 7  # (?)

model = MTS(bert_encoder, text_embedding_dim, geodata_embedding_dim)

# track_pairs = [
#     {"text": "track description 1", "geodata": geodata_tensor_1},
#     {"text": "track description 2", "geodata": geodata_tensor_2},
#     ...
# ]

criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=0.001)


for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for pair in track_pairs:
        text_input = pair["text"]
        geodata_input = pair["geodata"]

        text_tokens = bert_tokenizer.encode(text_input, add_special_tokens=True, padding=True, truncation=True)
        text_embeddings = {"input_ids": torch.tensor(text_tokens).unsqueeze(0)}

        text_embeddings, geodata_embeddings = model(text_embeddings, geodata_input.unsqueeze(0))

        similarity_score = torch.cosine_similarity(text_embeddings, geodata_embeddings)

        target_similarity = 1.0
        loss = criterion(similarity_score, target_similarity)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss}")