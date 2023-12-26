import torch
import torch.nn as nn


class MTS(nn.Module):
    def __init__(self, bert_encoder, text_embedding_dim, geodata_embedding_dim):
        super(MTS, self).__init__()
        self.bert_encoder = bert_encoder
        self.projection_text = nn.Linear(text_embedding_dim, text_embedding_dim)  # Linear layer for text projection
        self.projection_geodata = nn.Linear(geodata_embedding_dim, text_embedding_dim)  # Linear layer for geodata projection

    def forward(self, text_embeddings, geodata_embeddings):
        text_encoded = self.bert_encoder(**text_embeddings).last_hidden_state

        projected_text = self.projection_text(text_encoded)
        projected_geodata = self.projection_geodata(geodata_embeddings)

        return projected_text, projected_geodata


