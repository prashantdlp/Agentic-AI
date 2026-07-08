from sentence_transformers import SentenceTransformer
import numpy as np
import argparse
import re

def fixed_chunk(text: str, size: int = 50, overlap: int = 10) -> list[str]:
    if not text:
        print("No text in chunking stage")
        exit(1)

    if overlap >= size:
        print("overlap must be smaller than size")
        exit(1)

    words = text.strip().split()
    chunks = []

    start = 0
    while start < len(words):
        end = min(len(words), start + size)
        chunks.append(' '.join(words[start:end]))

        if end == len(words):
            break

        start = end - overlap
    
    return chunks

def sentence_chunk(text: str) -> list[str]:
    if not text:
        print("No text in chunking stage")
        exit(1)

    raw_sentences = [s.strip() for s in re.split(r'[.?!]', text.strip()) if s.strip()]
    chunks = []
    for start in range(0, len(raw_sentences), 5):
        end = min(len(raw_sentences), start + 5) # [start, end)
        chunk = " ".join(raw_sentences[start:end])
        chunks.append(chunk)
    return chunks

def sliding_window_chunk(text: str, window: int = 100, step: int = 50) -> list[str]:
    if not text:
        print("No text in chunking stage")
        exit(1)

    raw_words = text.strip().split() 
    chunks = []

    start = 0
    while start < len(raw_words):
        end = min(len(raw_words), start + window)
        chunk = " ".join(raw_words[start:end])
        chunks.append(chunk)

        if end == len(raw_words):
            break

        start += step

    return chunks

def A(text: str):
    chunk1 = fixed_chunk(text)
    N = len(chunk1)
    assert N != 0
    mean = sum(len(chunk.split()) for chunk in chunk1) / N
    mini = min(len(chunk.split()) for chunk in chunk1)
    maxi = max(len(chunk.split()) for chunk in chunk1)
    print(f"Strategy: fixed_chunk")
    print(f"number of chunks: {N}")
    print(f"mean chunk length: {mean}")
    print(f"min chunk length: {mini}")
    print(f"max chunk length: {maxi}")
    print("-" * 50)

    return chunk1

def B(text: str):
    chunk2 = sentence_chunk(text)
    N = len(chunk2)
    assert N != 0
    mean = sum(len(chunk.split()) for chunk in chunk2) / N
    mini = min(len(chunk.split()) for chunk in chunk2)
    maxi = max(len(chunk.split()) for chunk in chunk2)
    print(f"Strategy: sentence_chunk")
    print(f"number of chunks: {N}")
    print(f"mean chunk length: {mean}")
    print(f"min chunk length: {mini}")
    print(f"max chunk length: {maxi}")
    print("-" * 50) 

    return chunk2

def C(text: str):
    chunk3 = sliding_window_chunk(text)
    N = len(chunk3)
    assert N != 0
    mean = sum(len(chunk.split()) for chunk in chunk3) / N
    mini = min(len(chunk.split()) for chunk in chunk3)
    maxi = max(len(chunk.split()) for chunk in chunk3)
    print(f"Strategy: sliding_window_chunk")
    print(f"number of chunks: {N}")
    print(f"mean chunk length: {mean}")
    print(f"min chunk length: {mini}")
    print(f"max chunk length: {maxi}")
    print("-" * 50)

    return chunk3

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=str)
    args = parser.parse_args()
    file_path = args.file_path
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        print("Please provide valid supported file")
        exit(1)

    chunk1 = A(text)
    chunk2 = B(text)
    chunk3 = C(text)

    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    chunk1_embeddings = embedder.encode(chunk1, show_progress_bar = True) 
    chunk1_embeddings = chunk1_embeddings / np.linalg.norm(chunk1_embeddings, axis=1, keepdims=True)

    chunk2_embeddings = embedder.encode(chunk2, show_progress_bar = True) 
    chunk2_embeddings = chunk2_embeddings / np.linalg.norm(chunk2_embeddings, axis=1, keepdims=True)

    chunk3_embeddings = embedder.encode(chunk3, show_progress_bar = True) 
    chunk3_embeddings = chunk3_embeddings / np.linalg.norm(chunk3_embeddings, axis=1, keepdims=True)

    def retrieve_top_indices(query: str, chunk_embeddings, top_k: int = 1) -> list[str]:    
        global embedder

        query_embedding = embedder.encode(query)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        scores = chunk_embeddings @ query_embedding.T
        top_indices = np.argsort(scores)[-top_k:][::-1] 
        return top_indices
    
    questionaire = {
        1: {
            "question": "What are the five elements of emotional intelligence according to Goleman?",
            "answer": [
                "self-awareness",
                "self-regulation",
                "motivation",
                "empathy",
                "social skills"
            ]
        },

        2: {
            "question": "Which leadership styles are associated with Kurt Lewin's behavioural theory?",
            "answer": [
                "autocratic",
                "democratic",
                "laissez-faire"
            ]
        },

        3: {
            "question": "Which company in France studied emotional intelligence in its workforce?",
            "answer": [
                "sanofi"
            ]
        },

        4: {
            "question": "Which company uses peer reviews for manager feedback?",
            "answer": [
                "google"
            ]
        },

        5: {
            "question": "Which car company is mentioned as investing in leadership training?",
            "answer": [
                "bentley motors"
            ]
        }
    }
    total = len(questionaire)

    for i, chunk_embeddings in enumerate([chunk1_embeddings, chunk2_embeddings, chunk3_embeddings]):
        Hits = 0
        for item in questionaire.values():
            query = item["question"]
            answers = item["answer"]

            top_indices = retrieve_top_indices(query, chunk_embeddings)

            if i == 0:
                top_k_chunks =  [chunk1[idx] for idx in top_indices]
            elif i == 1:
                top_k_chunks =  [chunk2[idx] for idx in top_indices]
            else:
                top_k_chunks =  [chunk3[idx] for idx in top_indices]

            context = " ".join(top_k_chunks).lower()
            # print(f"Strategy {i+1}: context: {context}") # for debugging

            if all(ans.lower() in context for ans in answers):
                Hits += 1
        
        print(f"Strategy {i+1}: Hit Rate = {Hits}/{total} = {Hits/total:.2f}")