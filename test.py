import torch
from transformers import BertTokenizer

# Initialize a BERT model and tokenizer (replace with your specific BERT model)
bert_model_name = "bert-base-uncased"
bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)

# Load your trained model (replace 'your_model_path' with the actual path to your trained model checkpoint)
model = torch.load('your_model_path')

# Define a function to retrieve similar tracks based on a given text description
def retrieve_similar_tracks(text_description, geodata_entries, top_k=5):
    # Tokenize text input
    text_tokens = bert_tokenizer.encode(text_description, add_special_tokens=True, padding=True, truncation=True)
    text_embeddings = {"input_ids": torch.tensor(text_tokens).unsqueeze(0)}

    # Calculate text embeddings using the model
    text_embeddings, _ = model(text_embeddings, geodata_entries)

    # Calculate cosine similarities between text embeddings and geodata embeddings for all entries
    similarities = []
    for geodata_entry in geodata_entries:
        _, geodata_embeddings = model(text_embeddings, geodata_entry)
        similarity_score = torch.cosine_similarity(text_embeddings, geodata_embeddings)
        similarities.append(similarity_score.item())

    # Sort entries by similarity and return the top-k similar entries
    similar_tracks = [entry for entry, similarity in sorted(zip(geodata_entries, similarities), key=lambda x: x[1], reverse=True)[:top_k]]

    return similar_tracks

# Example geodata entries (replace with your actual geodata entries)
geodata_entries = [
    torch.tensor([1.0, 2.0, 0.5, 30.0, 15.0, 0.0, 24.0]),
    torch.tensor([0.5, 1.0, 0.25, 20.0, 10.0, 5.0, 15.0]),
    # Add more geodata entries as needed
]

# Example text description
text_description = "A scenic hiking trail with moderate elevation gain and beautiful views."

# Retrieve similar tracks based on the text description
similar_tracks = retrieve_similar_tracks(text_description, geodata_entries, top_k=5)

# Print the similar tracks
print("Top 5 Similar Tracks:")
for track in similar_tracks:
    print(track)
