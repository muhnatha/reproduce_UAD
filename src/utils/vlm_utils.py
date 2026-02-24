"""
Utility functions for VLM
"""
from openai import OpenAI

import os
import ast
import base64
import requests
import logging
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from sentence_transformers import SentenceTransformer
sentence_transformer_model = SentenceTransformer("all-MiniLM-L6-v2")


##############################################
## Text embedding
##############################################
def get_text_embedding_options(option="embeddings_oai"):
    """
    Get text embedding function based on the option
    """
    if option == "embeddings_oai":
        return get_text_embedding
    elif option == "embeddings_st":
        return get_text_embedding_sentence_transformer
    else:
        raise ValueError(f"Invalid option: {option}")

def get_text_embedding(text, model="text-embedding-3-large", dim=1024):
    """
    Get openai text embedding with specified dimension
    """
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    embedding = np.array(client.embeddings.create(input=[text], model=model, dimensions=dim).data[0].embedding)
    return embedding

def get_text_embedding_sentence_transformer(text, model_name="all-MiniLM-L6-v2"):
    """
    Get text embedding with sentence transformer
    """
    # model = SentenceTransformer(model_name)
    embedding = sentence_transformer_model.encode(text) # shape (D,)
    return embedding

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def create_payload_gpt4(messages):
    """
    Create payload for GPT-4.
    Input:
        messages: list of dicts, each in form {"role": str, "content": list of content dicts}
    """
    payload = {
        "model": "gpt-4o",
        "messages": messages,
    }
    return payload