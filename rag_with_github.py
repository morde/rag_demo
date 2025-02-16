import os

import boto3
import git
import numpy as np
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_aws import BedrockEmbeddings
from langchain_community.chat_models import BedrockChat
from langchain_core.documents import Document

from constants import GITHUB_REPOS

# AWS Bedrock client setup
bedrock = boto3.client("bedrock-runtime")

# AWS Titan Embedding Model
embedding_model = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1")

# GitHub Repos to clone

# Persistent storage directory for FAISS
FAISS_INDEX_PATH = "faiss_vectorstore"

# Temporary directory to store cloned repos
temp_dir = "GITHUB_REPOS"

def clone_github_repos():
    """Clones the specified GitHub repositories."""
    print("Cloning repositories...")
    repo_dirs = []
    for repo_url in GITHUB_REPOS:
        repo_name = repo_url.split("/")[-1].replace(".git", "")
        repo_path = os.path.join(temp_dir, repo_name)
        if not os.path.exists(repo_path):
            print(f"Cloning {repo_url}...")
            git.Repo.clone_from(repo_url, repo_path)
        repo_dirs.append(repo_path)
    return repo_dirs

def extract_text_from_repos(repo_dirs):
    """Extracts text content from README, markdown, and code comments, with metadata."""
    extracted_documents = []
    for repo in repo_dirs:
        repo_name = os.path.basename(repo)
        for root, _, files in os.walk(repo):
            for file in files:
                if file.endswith((".md", ".py", ".js", ".ts", ".java", ".go", ".rs", ".cs", ".vb", ".fs")):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                        metadata = {
                            "project": repo_name,
                            "component": os.path.relpath(root, repo),
                            "filename": file,
                            "filepath": file_path
                        }
                        extracted_documents.append(Document(page_content=content, metadata=metadata))
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
    return extracted_documents


def create_vectorstore(documents):
    """Creates a FAISS vectorstore from extracted documents and saves it persistently."""
    if not documents:
        print("No documents found for embedding. Skipping vectorstore creation.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(documents)

    if not split_docs:
        print("No split documents found. Skipping vectorstore creation.")
        return None

    vectors = embedding_model.embed_documents([doc.page_content for doc in split_docs])

    if not vectors or len(vectors) == 0:
        print("No embeddings generated. Skipping FAISS indexing.")
        return None

    vectors_np = np.array(vectors, dtype=np.float32)

    if vectors_np.size == 0 or len(vectors_np.shape) < 2:
        print("Embeddings array is empty or malformed. Skipping FAISS indexing.")
        return None

    dimension = vectors_np.shape[1]
    index_path = os.path.join(FAISS_INDEX_PATH, "github_repos")

    # Load existing FAISS index if it exists
    if os.path.exists(index_path):
        print("Loading existing FAISS index from storage...")
        _vectorstore = FAISS.load_local(index_path, embedding_model)
    else:
        print("Creating new FAISS index...")
        _vectorstore = FAISS.from_documents(split_docs, embedding_model)
        _vectorstore.save_local(index_path)  # Save FAISS to disk

    return _vectorstore

def create_rag_chain(vectorstore):
    """Creates a RAG pipeline with similarity search."""
    bedrock_client = boto3.client("bedrock-runtime", region_name="us-west-2")

    # Initialize LLM with Claude 3 Sonnet (a newer model)
    llm = BedrockChat(
        client=bedrock_client,
        model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
        model_kwargs={
            "temperature": 0,
            "max_tokens": 500,
            "anthropic_version": "bedrock-2023-05-31"
        }
    )
    # Use similarity search to retrieve the most relevant chunks
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # Create Retrieval-Augmented Generation (RAG) pipeline
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

if __name__ == "__main__":
    repos = clone_github_repos()
    index_path = os.path.join(FAISS_INDEX_PATH, "github_repos")

    # Load existing FAISS index if it exists
    if not os.path.exists(index_path):
        documents = extract_text_from_repos(repos)
        vectorstore = create_vectorstore(documents)
    else:
        print("Loading existing FAISS index from storage...")
        vectorstore = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
    rag_chain = create_rag_chain(vectorstore)

    while True:
        user_question = input("Ask about your GitHub projects: ")
        if user_question.lower() in ["exit", "quit"]:
            break
        answer = rag_chain.run(user_question)
        print("\nClaude's Answer:\n", answer)
