import logging
import torch, json, os, numpy as np, faiss

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import T5Tokenizer, T5ForConditionalGeneration

from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("rag_pipeline.log", mode="w"),  # Save logs to a file
        logging.StreamHandler()  # Print logs to the console
    ]
)
logger = logging.getLogger(__name__)

# Disable parallel tokenization
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def process_document(path: str) -> list:
    logger.info("Processing document...")
    try:
        with open(path) as f:
            text = f.read()

        embedder = SentenceTransformer('all-MiniLM-L6-v2')

        def token_count(text: str) -> int:
            return len(embedder.tokenizer.tokenize(text))
    
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=32,
            chunk_overlap=8,
            length_function=token_count,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(text)

        logger.info("SUCCESS - Document processed into chunks.")
        return chunks
    except Exception as e:
        logger.error(f"FAILED - Error processing document: {e}")
        return []

def create_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    logger.info("Creating FAISS index...")
    try:
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        logger.info("SUCCESS - FAISS index created.")
        return index
    except Exception as e:
        logger.error(f"FAILED - Error creating FAISS index: {e}")
        return None

def retrieve_context(query: str, index: faiss.Index, chunks: list, embedder, k=3):
    logger.info("Retrieving context...")
    try:
        query_embed = embedder.encode([query]).reshape(1, -1)
        distances, indices = index.search(query_embed, k)
        
        if indices.size == 0:
            logger.info("No relevant chunks found.")
            return [], []
        
        valid_indices = [i for i in indices[0] if i < len(chunks)]
        logger.info(f"SUCCESS - Retrieved {len(valid_indices)} relevant chunks.")
        return valid_indices, [chunks[i] for i in valid_indices]
    except Exception as e:
        logger.error(f"FAILED - Error retrieving context: {e}")
        return [], []

# Answer Generation
def truncate_context(contexts: List[str], tokenizer, max_tokens: int=80) -> str:
    """Truncate concatenated context to fit LLM's token limit."""
    combined = " ".join(contexts)
    tokens = tokenizer.encode(combined, truncation=True, max_length=max_tokens)
    return tokenizer.decode(tokens, skip_special_tokens=True)

def generate_answer(query: str, context: str, model, tokenizer, device):
    logger.info("Generating answer...")
    try:
        # prompt = f"Answer based ONLY on this context. If unsure, say 'I don't know'.\nContext: {context}\nQuestion: {query}\nAnswer:"
        prompt = (
            f"Answer the question based ONLY on the context below. "
            f"If the context does not provide enough information, say 'I don't know.'\n\n"
            f"Context: {context}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )
        print("Tokenizing prompt...")
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512
                           ).to(device)
        outputs = model.generate(**inputs, max_length=512)
        # answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Decode the output and extract only the answer (remove the prompt)
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the part after "Answer:"
        # answer = full_output.split("Answer:")[-1].strip()

        logger.info("SUCCESS - Answer generated.")
        return full_output
    except Exception as e:
        logger.error(f"FAILED - Error generating answer: {e}")
        return "I don't know."
    
def main():
    logger.info("Starting RAG pipeline...")
    
    # Step 1: Process document
    logger.info("Step 1: Loading and processing document...")
    chunks = process_document("input/medicare_comparison.md")
    if not chunks:
        logger.error("FAILED - Exiting pipeline.")
        return
    
    # Step 2: Create FAISS index
    logger.info("Step 2: Generating embeddings and creating FAISS index...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedder.encode(chunks)
    index = create_faiss_index(embeddings)
    if index is None:
        logger.error("FAILED - Exiting pipeline.")
        return
    
    # Step 3: Load queries
    logger.info("Step 3: Loading queries...")
    try:
        with open("input/queries.json") as f:
            queries = json.load(f)
        logger.info("SUCCESS - Queries loaded.")
    except Exception as e:
        logger.error(f"FAILED - Error loading queries: {e}")
        return
    
    # Step 4: Load LLM
    logger.info("Step 4: Loading LLM...")
    try:
        device = torch.device("cpu")
        # tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
        # model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base").to(device)

        tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
        model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

        logger.info("SUCCESS - LLM loaded.")
    except Exception as e:
        logger.error(f"FAILED - Error loading LLM: {e}")
        return
    
    # Step 5: Process queries
    logger.info("Step 5: Processing queries...")
    results = []
    for q in queries:
        try:
            q_text, q_id = q["text"], q["id"]
        except KeyError as e:
            logger.error(f"FAILED - Missing key in query object: {e}")
            continue
        
        logger.info(f"\nProcessing Query {q_id}: {q_text}")
        
        # Step 5.1: Retrieve context
        logger.info(f"Step 5.1 (Query {q_id}): Retrieving context...")
        indices, retrieved_chunks = retrieve_context(q_text, index, chunks, embedder)
        
        if not retrieved_chunks:
            answer = "I don't know."
            source_chunks = []
            source_texts = []
        else:
            # Step 5.2: Generate answer
            logger.info(f"Step 5.2 (Query {q_id}): Generating answer...")
            context = truncate_context(retrieved_chunks, tokenizer, 80)
            answer = generate_answer(q_text, context, model, tokenizer, device)
            source_chunks = [f"chunk_{i}" for i in indices]
            source_texts = retrieved_chunks
        
        results.append({
            "query_id": q_id,
            "query_text": q_text,
            "answer": answer,
            "source_chunks": source_chunks,
            "source_text": source_texts
        })
    
    # Step 6: Save results
    logger.info("Step 6: Saving results...")
    try:
        os.makedirs("output", exist_ok=True)
        with open("output/answers.json", 'w') as f:
            json.dump(results, f, indent=2)
        logger.info("SUCCESS - Results saved to output/answers.json.")
    except Exception as e:
        logger.error(f"FAILED - Error saving results: {e}")

if __name__ == "__main__":
    main()