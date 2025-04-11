import torch
from transformers import GPT2Tokenizer, GPT2Model
from sklearn.decomposition import PCA
import plotly.express as px
import numpy as np
import plotly.graph_objects as go


# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")

# Input text
text = "The The The"

# Tokenize input
inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)

# Get token embeddings
with torch.no_grad():
    outputs = model(**inputs)  # Forward pass
    embeddings = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_dim)

# Convert tokens
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

# Print token embeddings
for token, vector in zip(tokens, embeddings[0]):
    print(f"{token}: {vector[:5].tolist()}...")  # Show first 5 values

# Contextual embeddings (corrected)
with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[0]  # (seq_len, hidden_dim)

#input_ids = inputs['input_ids']
#embedding_layer = model.get_input_embeddings()  # Same as model.transformer.wte
#embeddings = embedding_layer(input_ids).squeeze(0) 


# PCA
pca = PCA(n_components=3)
reduced_embeddings = pca.fit_transform(embeddings.detach().cpu().numpy())
#reduced_embeddings = pca.fit_transform(embeddings.cpu().numpy())

# Create a DataFrame for Plotly
#import pandas as pd
#df = pd.DataFrame(reduced_embeddings, columns=["PC1", "PC2", "PC3"])
#df["token"] = [token.replace("Ġ", "") for token in tokens]
#df["token"] = tokens

reduced = pca.fit_transform(embeddings.detach().cpu().numpy())  # (seq_len, 3)

clean_tokens = [token.replace("Ġ", "") for token in tokens]


# Plot: use line segments from (0, 0, 0) to each embedding vector
fig = go.Figure()

# Add each arrow as a 3D line
for i, (x, y, z) in enumerate(reduced):
    fig.add_trace(go.Scatter3d(
        x=[0, x], y=[0, y], z=[0, z],
        mode='lines+text',
        line=dict(width=4),
        text=[None, clean_tokens[i]],
        textposition='top center',
        name=clean_tokens[i],
        showlegend=False
    ))

# Layout
fig.update_layout(
    title="3D Token Embedding Vectors from Origin",
    scene=dict(
        xaxis_title='PC1',
        yaxis_title='PC2',
        zaxis_title='PC3'
    ),
    width=800,
    height=700
)

fig.show()