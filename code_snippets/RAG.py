from transformers import pipeline
from faiss import IndexFlatL2
from langchain.llms import OpenAI
from langchain.question_answering import RetrieverRAG

# OpenAI Setup
openai_model = OpenAI("text-embedding-3-small")  # Use "text-embedding-3-small" model
embed_fn = pipeline("feature-extraction", model=openai_model.pretrained_model, device=0)  # Specify GPU device (optional)

# Local Vector Store using FAISS
loader = DirectoryLoader(DOCUMENT_DIRECTORY, glob="*.html")
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(loader.load())
docs_as_str = [doc.page_content for doc in documents]

# Create FAISS index using OpenAI embeddings
d = openai_model.get_embedding_dim()  # Get embedding dimension from OpenAI model
index = IndexFlatL2(d)
index.add(embed_fn(docs_as_str, return_tensors="pt")["embeddings"].cpu().detach().numpy())

# Building the RAG System with Retriever using FAISS
retriever = Retriever(embed_fn=embed_fn, index=index)
rag = RetrieverRAG(retriever=retriever, answerer=openai_model)


def get_context(query):
  # Use FAISS to search for similar passages in context documents
  query_embedding = embed_fn(query, return_tensors="pt")["embeddings"][0].unsqueeze(0).cpu().detach().numpy()
  D, I = index.search(query_embedding, k=5)

  # Retrieve top K most similar passages
  similar_passages = [docs_as_str[i] for i in I.ravel()]

  # Construct context from retrieved similar passages
  context = "".join(similar_passages)

  max_length = 40000
  if len(context) > max_length:
      context = context[:max_length]

  prompt = f"""The following is a conversation between an employee and an AI. The AI provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know. Use the following pieces of context to answer the question at the end.

      {context}

      employee: {query}
      AI Assistant:"""

  return prompt

# Example usage
query = "What is the capital of France?"
context_prompt = get_context(query)
answer = rag.answer(query, context_prompt)
print(answer)