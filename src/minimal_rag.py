""" 
This script implements a minimal Retrieval-Augmented Generation (RAG) pipeline. It retrieves  relevant documents 
from a predefined knowledge base using TF-IDF vectorization and cosine similarity, then  generates a contextual 
response to a user-provided question using a GPT-2 language model. The script evaluates the  quality of the 
response based on relevance, references, and length, providing a critique and confidence score.  It can be run 
from the command line with options to specify the number of documents to retrieve and  output format. 
""" 
# minimal rag 
import sys 
import numpy as np 
import re 
import json
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity 
from transformers import pipeline 
import argparse
import warnings
warnings.filterwarnings('ignore')

def load_model( model="gpt2"):
  try: 
    print("loading model...") 
    llm = pipeline("text-generation", model=model, max_new_tokens=256 )  
    print(f"model {model} loaded") 
    return llm
  except Exception as e: 
    print("Error loading model: {e}") 
    sys.exit(1)

# knoledge base
docs = [ 
  {"id": "1", "txt": "Python is a high-level, interpreted programming language. It was created by  Guido van Rossum and first released in 1991. Python emphasizes code readability with its  notable use of significant whitespace."}, 
  {"id": "2", "txt": "FastAPI is a modern, fast web framework for building APIs with Python 3.7+.  It is based on standard Python type hints and provides automatic API documentation. FastAPI is  one of the fastest Python frameworks available."}, 
  {"id": "3", "txt": "Machine learning is a subset of artificial intelligence that enables systems to  learn and improve from experience. It uses algorithms to parse data, learn from it, and make  predictions or decisions without being explicitly programmed."}, 
  {"id": "4", "txt": "RAG (Retrieval-Augmented Generation) combines information retrieval with  text generation. It retrieves relevant documents from a knowledge base and uses them to  generate informed, contextual answers. This approach improves accuracy and reduces  hallucinations."}, 
  {"id": "5", "txt": "Vector databases store data as high-dimensional vectors and enable efficient  similarity search. They are essential for modern AI applications, particularly in semantic search  and retrieval systems. Common examples include Pinecone, Weaviate, and Chroma."} 
  ]

def vectorize_document(docs):
  vectorizer = TfidfVectorizer() 
  vectors = vectorizer.fit_transform([x["txt"] for x in docs]) 
  return vectorizer, vectors


# retrieved_docs = [] 
# answer = "" 
# problems = [] 
# score = 1.0 
# is_valid = False 
# temp_storage = {} 

# # i = 2 
# # while i < len(sys.argv): 
# # if sys.argv[i] == "-k": 
# # k = int(sys.argv[i+1]) 
# # i += 2 
# # elif sys.argv[i] == "--json":
# # use_json = True 
# # i += 1 
# # else: 
# # i += 1 
# # retrieved_docs = [] 
# # answer = "" 
# # problems = [] 
# # score = 1.0 
# temp_storage["question"] = question 

def retreive_docs(question, vectorizer, vectors, num_retrives):
    query_vec = vectorizer.transform([question]) 
    similarities = cosine_similarity(query_vec, vectors)[0] 
    #order docs by similarirty score
    indices = np.argsort(similarities)[-num_retrives:][::-1] 
    retrieved_docs = []
    for i in indices: 
        if similarities[i] > 0 and len(retrieved_docs) < num_retrives: 
            retrieved_docs.append({'id': str(i), "doc": docs[i], "score": float(similarities[i])}) 
            # if : 
            #   break
    return retrieved_docs


# def get_data(): 
# global retrieved_docs, temp_storage 

# temp_storage["sims"] = similarities 

 
# get_data()

def generate_answer(question, retrieved_docs, llm):
    # problem = []
    # score = 1.0
    # is_valid = False
    if len(retrieved_docs) == 0: 
        answer = "idk" 
        # problems.append("no docs") 
        # score = score * 0.4 
        # is_valid = False 
        # print("\nQ: {}".format(question)) 
        # print(f"\nA: {answer}\n") 
        # print("Citations: None") 
        # print("\nCritique: ISSUES (confidence: {:.2f})".format(score)) 
        # print("Problems: %s" % ', '.join(problems)) 

    context = '\n'.join([f"[{doc['doc']['id']}] {doc['doc']['txt']}"  for i,doc in enumerate(retrieved_docs)])
    # print(retreive_docs)
    prompt = """
    You are an expert in information retrieval and text generation. Your task is to answer questions strictly based on the provided context.

    Instructions:
    1.  **Strict Context Adherence**: Answer the question using *only* the information found in the `Context:` section. Do not use any outside knowledge.
    2.  **Mandatory Citations**: Every piece of information or claim you make in your answer *must* be immediately followed by an explicit citation to its source from the `Context:` using the format `[X]`, where `X` is the corresponding context number. If a claim draws from multiple sources, cite all applicable sources (e.g., `[1][3]`).
    3.  **Conciseness**: Keep your answer as brief and to the point as possible, while still being informative.
    4.  **Handling Unknowns**: If the answer cannot be found *within the provided context*, state explicitly, "I don't know based on the provided context." Do not guess or infer.

    Example:
    Contest: [4] RAG (Retrieval-Augmented Generation) combines information retrieval with text generation. It retrieves relevant documents from a knowledge base and uses them to generate informed, contextual answers. This approach improves accuracy and reduces hallucinations.
    [1] Python is a high-level, interpreted programming language.
    Question: What is RAG?
    Answer: RAG combines information retrieval with text generation [4].

    Context:
    {}

    Question: {}
    
    Answer:""".format(context,  question)
    # prompt = "Context:\n{}\nQuestion: {}\n\nAnswer the question based only on the context  above. Cite sources using [1], [2], etc. Be concise.\n\nAnswer:".format(context,  question)
    # print(prompt)
    output = llm(prompt) 
    # print(output)
    answer = output[0]['generated_text'].split("Answer:")[-1].strip() 
    if answer == "": 
        answer = "idk" 
    return answer

# def call_model(): 
# global answer, temp_storage 
# ctx = "" 
# counter = 0 
# for r in retrieved_docs: 
# counter += 1 
# ctx += f"[{counter}] {r['doc']['txt']}\n\n" 
# temp_storage["context"] = ctx 
# prompt = "Context:\n{}\nQuestion: {}\n\nAnswer the question based only on the context  above. Cite sources using [1], [2], etc. Be concise.\n\nAnswer:".format(ctx,  temp_storage['question'])
# output = llm(prompt) 
# answer = output[0]['generated_text'].split("Answer:")[-1].strip() 

# call_model() 
def validate_output(answer, retrieved_docs): 
    problems = []
    score = 1.0
    # is_valid = False

    if "idk" in answer.lower(): 
        problems.append("no info") 
        score = score * 0.5 

    if len(answer) < 20: 
        problems.append("short") 
        score = score * 0.3 

    refs = re.findall(r'\[(\d+)\]', answer) 
    if len(retrieved_docs) > 0 and not refs: 
        problems.append("no refs")
        score = score * 0.7 
    if retrieved_docs:
        avg_score = np.mean([doc['score'] for doc in retrieved_docs])
    else:
        avg_score = 0

    if avg_score < 0.1: 
        problems.append("low relevance") 
        score = score * 0.6 

    if score > 0.6 and len(problems) < 3: 
        is_valid = True 
    else: 
        is_valid = False 
    return problems, score, is_valid

def main():
    # if len(sys.argv) < 2: 
    #   print("usage: python minimal-rag.py <question> [-k NUM] [--json]") 
    #   print("example: python minimal-rag.py 'What is RAG?' -k 2 --json") 
    #   sys.exit(1) 
    # question = sys.argv[1] 
    # k = 3 
    # use_json = False  
    parser = argparse.ArgumentParser(description="usage: python minimal-rag.py -q <question> [-k NUM] [-j Bool] [-m Model] \n example: python minimal-rag.py 'What is RAG?'")
    parser.add_argument('-q', '--question', type=str, default = "What is RAG?", help = "Question string")
    parser.add_argument('-k', type=str, default = 3, help = "Number of documents to retrieve")
    parser.add_argument('-j', '--json', type = int, default = 1, help = 'Require JSON output or not, 1 or 0')
    parser.add_argument('-m', '--model', type = str, default = 'gpt2', help = 'HuggingFace model to choose ')
    args = parser.parse_args()
    print(args)

    question, num_retrives, use_json, model = args.question, int(args.k), bool(args.json), args.model

    llm = load_model( model=model)

    vectorizer, vectors = vectorize_document(docs)

    retrieved_docs = retreive_docs(question, vectorizer, vectors, num_retrives)
    # print('retrived docs: ', retrieved_docs)
    # for doc in retrieved_docs: 
    #     print(doc['id'], doc['doc']['id'], doc['doc']['txt'], doc['score'], )

    answer = generate_answer(question, retrieved_docs, llm)

    problems, score, is_valid = validate_output(answer, retrieved_docs)

    result = { 
    "answer": answer, 
    "citations": retrieved_docs, 
    "critique": {"ok": is_valid, "score": score, "problems": problems} 
    } 
    if use_json: 
        print(json.dumps(result, indent=2)) 
    else:
        print("\nQ: {}".format(question)) 
        print(f"\nA: {answer}\n") 
        print("Citations:") 

        for r in retrieved_docs:
            doc_text = r['doc']['txt'] 
            if len(doc_text) > 80: 
                doc_text = doc_text[:80] + "..." 
            print(" [{}] (score: {:.3f}) {}".format(r['doc']['id'], r['score'], doc_text)) 
        if is_valid: 
            print("\nCritique: OK (confidence: {:.2f})".format(score)) 
        else: 
            print(f"\nCritique: ISSUES (confidence: {score:.2f})") 
        if len(problems) > 0: 
            print("Problems: %s" % ', '.join(problems))  
    sys.exit(0)

if __name__ == "__main__":
    main()
