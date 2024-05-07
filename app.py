import streamlit as st
from sentence_transformers import SentenceTransformer, util

# Initialize the SentenceTransformer model
model = SentenceTransformer('Supabase/gte-small')

# Predefined documents
docs = [
    "A man is eating food.",
    "A man is eating a piece of bread.",
    "The girl is carrying a baby.",
    "A man is riding a horse.",
    "A woman is playing violin.",
    "Two men pushed carts through the woods.",
    "A man is riding a white horse on an enclosed ground.",
    "A monkey is playing drums.",
    "A cheetah is running behind its prey.",
]

# Convert documents to embeddings
docs_embeddings = model.encode(docs, convert_to_tensor=True)

# Streamlit UI
st.title("Semantic Search with Sentence Transformers")
st.write("Input your query and find relevant sentences from the document set.")

# Query input field
query = st.text_input("Enter your search query:")

# If a query is entered, perform semantic search
if query:
    # Convert query to an embedding
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Perform semantic search
    hits = util.semantic_search(query_embedding, docs_embeddings, top_k=3)

    # Display results
    st.write(f"Results for: '{query}'")
    for hit in hits[0]:
        st.write(f"{docs[hit['corpus_id']]} (Score: {hit['score']:.4f})")
 
