import torch
from transformers import BertTokenizer, BertModel

def text_to_vector(text="Helloo"):
    embeddings = None
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    # Tokenize the input text
    input_ids = tokenizer.encode(text, add_special_tokens=True)
    # Convert the token IDs to a tensor
    input_ids_tensor = torch.tensor(input_ids)
    # Obtain BERT embeddings
    with torch.no_grad():
        embeddings = model(input_ids_tensor.unsqueeze(0))[0]
        print(embeddings.shape)
        return embeddings

print(text_to_vector("t this point, embeddings will contain the tensor representation of your input text based on the BERT model. You can use this tensor for various natural language understanding tasks or further analysis."))