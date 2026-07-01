# !pip install bertviz transformers -q
from bertviz import head_view, model_view
from transformers import BertTokenizer, BertModel
import torch

# Load BERT-base
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name, output_attentions=True)

# Visualise attention for a sentence
sentence = "The Prime Minister of Pakistan visited Lahore"
inputs = tokenizer(sentence, return_tensors="pt", add_special_tokens=True)
outputs = model(**inputs)
attention = outputs.attentions
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

# Show head view — interactive in Colab
head_view(attention, tokens)