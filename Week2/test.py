import matplotlib.pyplot as plt
from dotenv import load_dotenv
from minimal_rag import *
from groq import Groq
import numpy as np
import argparse
import os

class TWEAKED_RAG(RAG):
    def retrieve(self, query: str, top_k: int = 3) ->  tuple[list[str], float]:    
        query_embedding = self.model.encode(query)  
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        scores = self.chunk_embeddings @ query_embedding.T
        top_indices = np.argsort(scores)[-top_k:][::-1] 
        return [self.chunks[i] for i in top_indices], scores[top_indices[0]]

    def rag_prompt(self, query:str): 
        hypothetical_answer = self.generator(
                                query,
                                max_new_tokens=50
                            )[0]["generated_text"]

        chunks, max_similarity  = self.retrieve(hypothetical_answer, top_k = 3)
        context = "\n\n".join(chunk for chunk in chunks)
        prompt  = f"""
                Answer the question using ONLY the context below in less than 100 words. 
                If the answer is not in the context, say 'I don't know'.
            
                context: {context}
                Question: {query}
                Answer: """
        
        top_chunk = chunks[0] if chunks else ""

        return prompt, top_chunk, max_similarity

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=str)
    args = parser.parse_args()
    file_path = args.file_path

    load_dotenv()
    client = Groq(api_key= os.getenv("GROQ_API_KEY"))
    
    rag_object = TWEAKED_RAG(file_path)

    queries = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]
    questionaire = [
        "Why is management training considered important in organisations?",
        "What three types of skills do effective managers need according to the essay?",
        "Can managers be trained to become leaders according to the essay?",
        "What is the time complexity of quicksort in the average case?",
        "Who invented the Python programming language?",
        "How does blockchain prevent double spending?"
    ]
    colors = [
        "green",
        "green",
        "green",
        "red",
        "red",
        "red"
    ]

    LLM_RESPONSES  = []
    TOP_CHUNK      = []
    SIMILARITY     = []

    for i, query in enumerate(questionaire):
        prompt, top_chunk, max_similarity = rag_object.rag_prompt(query)
        
        chat_completion = client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": prompt
            }],
            model="llama-3.3-70b-versatile",
            max_completion_tokens=400
        )

        LLM_RESPONSES.append(chat_completion.choices[0].message.content)
        print(f"answerable: {colors[i]} | LLM: {LLM_RESPONSES[-1]}", end="\n\n")
        TOP_CHUNK.append(top_chunk)
        SIMILARITY.append(max_similarity)

    plt.figure(figsize=(8,4))
    plt.bar(queries, SIMILARITY, color=colors)
    plt.xlabel("Queries")
    plt.ylabel("Cosine Similarity")
    plt.title("RAG Retrieval Similarity Scores")
    plt.ylim(0,1)
    plt.savefig("rag_scores.png", dpi=300, bbox_inches="tight")
    print("Saved plot as rag_scores.png")