
"""

# Using FAISS with Google Generative AI Embeddings for Text Similarity Search

## Introduction

In this guide, we'll walk through the process of using FAISS (Facebook AI Similarity Search) in conjunction with `GoogleGenerativeAIEmbeddings` to perform efficient similarity searches on text data. FAISS is a powerful library designed for similarity search and clustering of dense vectors, making it a great tool for applications like search engines and recommendation systems.

## Prerequisites

Before we start, ensure you have the following Python libraries installed:
- `faiss-cpu` or `faiss-gpu` (depending on whether you have a GPU)
- `langchain-google-genai` for embeddings
- `python-dotenv` for environment variable management
- `numpy` for numerical operations

You can install these libraries using pip:

```bash
pip install faiss-cpu langchain-google-genai python-dotenv numpy
```

## Setting Up the Environment

1. **Create a `.env` File**

   Store your Google API key in a `.env` file to keep it secure and avoid hardcoding sensitive information in your script.

   ```env
   GOOGLE_API_KEY=your_google_api_key
   ```

2. **Load Environment Variables**

   Use `dotenv` to load the API key from the `.env` file.

"""

## Code Example

from dotenv import load_dotenv
import os
import numpy as np
import faiss
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key
api_key = os.getenv("GOOGLE_API_KEY")

# Initialize the embeddings with your API key
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=api_key)

# Define some sample documents
documents = ["hello, world!", "machine learning is fun", "deep learning is a subset of machine learning"]

# Embed the documents
vectors = np.array([embeddings.embed_query(doc) for doc in documents])

# Convert the list of vectors to a NumPy array
vectors = np.vstack(vectors).astype(np.float32)

# Create a FAISS index
dimension = vectors.shape[1]  # Dimensionality of the vectors
index = faiss.IndexFlatL2(dimension)

# Add the vectors to the index
index.add(vectors)

# Query vector
query_vector = np.array(embeddings.embed_query("hello")).astype(np.float32).reshape(1, -1)

# Search for the closest vectors
distances, indices = index.search(query_vector, k=1)  # k is the number of nearest neighbors to retrieve

# Print results
print("Nearest document index:", indices[0][0])
print("Distance:", distances[0][0])
print("Nearest document:", documents[indices[0][0]])

"""

### Explanation

1. **Loading Environment Variables**

   ```python
   load_dotenv()
   api_key = os.getenv("GOOGLE_API_KEY")
   ```

   Loads the API key from the `.env` file for secure access.

2. **Initializing Embeddings**

   ```python
   embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=api_key)
   ```

   Sets up the embeddings model using the Google API key.

3. **Embedding Documents**

   ```python
   vectors = np.array([embeddings.embed_query(doc) for doc in documents])
   ```

   Converts each document into a vector representation.

4. **Preparing FAISS Index**

   ```python
   vectors = np.vstack(vectors).astype(np.float32)
   dimension = vectors.shape[1]
   index = faiss.IndexFlatL2(dimension)
   index.add(vectors)
   ```

   Creates a FAISS index and adds the document vectors to it.

5. **Querying the Index**

   ```python
   query_vector = np.array(embeddings.embed_query("hello")).astype(np.float32).reshape(1, -1)
   distances, indices = index.search(query_vector, k=1)
   ```

   Converts the query into a vector and searches the index for the most similar document.

6. **Printing Results**

   ```python
   print("Nearest document index:", indices[0][0])
   print("Distance:", distances[0][0])
   print("Nearest document:", documents[indices[0][0]])
   ```

   Displays the index, distance, and content of the nearest document.

## Conclusion

Using FAISS with `GoogleGenerativeAIEmbeddings` allows you to efficiently index and search through dense vector representations of text data. This setup is ideal for creating powerful search and recommendation systems by leveraging similarity search capabilities. By following the steps outlined above, you can integrate these technologies into your applications and achieve high-performance text similarity searches.
"""