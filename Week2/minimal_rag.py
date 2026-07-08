from sentence_transformers import SentenceTransformer
from transformers import pipeline
from dotenv import load_dotenv
from groq import Groq
import numpy as np
import argparse
import os

class RAG:
    def __init__(self, file_path: str, 
                chunk_size: int = 200, 
                overlap: int = 40, 
                embedding_model_name: str = 'all-MiniLM-L6-v2'):
        
        self.file_path  = file_path
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                self.text = f.read()
        except FileNotFoundError:
            print("Please provide valid supported file")
            exit(1)

        self.chunk_size = chunk_size
        self.overlap    = overlap
        if self.overlap >= self.chunk_size:
            print("overlap must be smaller than chunk_size")
            exit(1)
        self.chunks = None
        self.chunk_text()

        self.embedding_model_name = embedding_model_name
        self.model = SentenceTransformer(self.embedding_model_name) 
        self.chunk_embeddings = None 
        self.embed_text()

        self.generator = pipeline("text2text-generation", model='google/flan-t5-base')

    def chunk_text(self):
        if not self.text:
            print("No text in chunking stage")
            exit(1)

        words = self.text.strip().split()
        self.chunks = []

        start = 0
        while start < len(words):
            end = min(len(words), start + self.chunk_size)
            self.chunks.append(' '.join(words[start:end]))

            if end == len(words):
                break

            start = end - self.overlap

    def embed_text(self):
        self.chunk_embeddings = self.model.encode(self.chunks, show_progress_bar = True) 
        self.chunk_embeddings = self.chunk_embeddings / np.linalg.norm(self.chunk_embeddings, axis=1, keepdims=True) # (N, d)

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:    
        query_embedding = self.model.encode(query) # (1, d) 
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        scores = self.chunk_embeddings @ query_embedding.T
        top_indices = np.argsort(scores)[-top_k:][::-1] 
        return [self.chunks[i] for i in top_indices]

    def rag_prompt(self, query:str) -> str: 
        hypothetical_answer = self.generator(
                                query,
                                max_new_tokens=50
                            )[0]["generated_text"]

        chunks = self.retrieve(hypothetical_answer, top_k = 3)
        context = "\n\n".join(chunk for chunk in chunks)
        prompt = f"""
                Answer the question using ONLY the context below in less than 100 words. 
                If the answer is not in the context, say 'I don't know'.
            
                context: {context}
                Question: {query}
                Answer: """
        return prompt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=str)
    args = parser.parse_args()
    file_path = args.file_path

    load_dotenv()
    client = Groq(api_key= os.getenv("GROQ_API_KEY"))

    rag_object = RAG(file_path)

    while True:
        print("Enter your query (q to quit): ", end='')
        query = input().strip()

        if query.lower() == 'q':
            print("Exiting...")
            break

        prompt = rag_object.rag_prompt(query)

        chat_completion = client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": prompt
            }],
            model="llama-3.3-70b-versatile",
            max_completion_tokens=400
        )

        print("Chatbot:", chat_completion.choices[0].message.content)
        print()