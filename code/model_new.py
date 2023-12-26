import torch
from transformers import BertTokenizer, BertModel, AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import json

class TrackMatcher:
    def __init__(self, bert_model_name="distilbert-base-multilingual-cased", decoder_model_name="andrea-coppari/mistral-7b-geodata-finetuning-eng-1500"):
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert_model = BertModel.from_pretrained(bert_model_name)
        self.decoder_tokenizer = AutoTokenizer.from_pretrained(decoder_model_name)
        self.decoder_model = AutoModelForCausalLM.from_pretrained(decoder_model_name)

    def load_json_objects_from_file(self, file_path):
        with open(file_path, "r+") as file:
            data = json.load(file)
        return [json.dumps(obj) for obj in data["tracks"]]

    def decode_json_objects(self, json_objects):
        decoded_sentences = []
        for json_obj in json_objects:
            input_ids = self.decoder_tokenizer.encode(json_obj, return_tensors='pt', max_length=128, truncation=True, padding=True)
            with torch.no_grad():
                decoded_output = self.decoder_model.generate(input_ids)
            decoded_text = self.decoder_tokenizer.decode(decoded_output[0], skip_special_tokens=True)
            decoded_sentences.append(decoded_text)
        return decoded_sentences

    def encode_sentence(self, sentence):
        tokens = self.bert_tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            output = self.bert_model(**tokens)
        return output.last_hidden_state.mean(dim=1)

    def calculate_similarity(self, input_sentence, sentence_list):
        input_embedding = self.encode_sentence(input_sentence)
        similarities = []
        for sentence in sentence_list:
            sentence_embedding = self.encode_sentence(sentence)
            similarity = cosine_similarity(input_embedding, sentence_embedding) #to optimize
            similarities.append(similarity)
        return similarities

    def find_best_match(self, input_sentence, json_objects):
        decoded_sentences = self.decode_json_objects(json_objects)
        similarities = self.calculate_similarity(input_sentence, decoded_sentences)
        best_match_index = similarities.index(max(similarities))
        best_match = decoded_sentences[best_match_index]
        return best_match, best_match_index


if __name__ == "__main__":
    matcher = TrackMatcher()
    
    input_sentence = "Track X ..."
    json_objects = matcher.load_json_objects_from_file("track_description.geojson")
    
    best_match, best_match_index = matcher.find_best_match(input_sentence, json_objects)
    
    print(f"Best match: {best_match}")
