import os
import re
import html
import boto3
from atlassian import Confluence
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings
from langchain_community.chat_models import BedrockChat
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
import warnings

# 🔹 Load environment variables
load_dotenv()

# 🔹 Configuration
AWS_REGION = os.getenv("AWS_REGION", "us-west-2")
VECTORSTORE_PATH = "faiss_confluence_index"

CONFLUENCE_USER = os.getenv("CONFLUENCE_USER")
CONFLUENCE_API_TOKEN = os.getenv("CONFLUENCE_API_TOKEN")
CONFLUENCE_SPACEKEY = os.getenv("CONFLUENCE_SPACEKEY")
CONFLUENCE_URL = os.getenv("CONFLUENCE_URL")

# 🔹 Initialize Confluence client
confluence = Confluence(url=CONFLUENCE_URL, username=CONFLUENCE_USER, password=CONFLUENCE_API_TOKEN)


def clean_html(raw_html):
    """Removes HTML tags and decodes HTML entities."""
    clean_text = re.sub(r"<[^>]+>", "", raw_html)  # Remove HTML tags
    return html.unescape(clean_text)  # Convert entities (e.g., &amp; → &)


def get_all_child_pages_recursive(parent_page_id, max_pages=50, depth=10):
    """Recursively fetches all pages under a parent page."""
    pages = list(confluence.get_child_pages(parent_page_id))  # Convert generator to list
    documents = []

    for page in pages[:max_pages]:  # Limit results
        page_id = page["id"]
        title = page["title"]
        content = confluence.get_page_by_id(page_id, expand="body.storage")["body"]["storage"]["value"]

        cleaned_content = clean_html(content)
        documents.append(Document(page_content=cleaned_content, metadata={"title": title, "depth": depth}))

        print(f"{'  ' * depth}📄 Found page: {title} (ID: {page_id})")

        # 🔹 **Recursive Call: Fetch child pages of this page**
        sub_documents = get_all_child_pages_recursive(page_id, max_pages, depth + 1)
        documents.extend(sub_documents)

    return documents


def create_faiss_vector_store(documents):
    """Splits text and stores it in FAISS for retrieval."""
    if not documents:
        print("⚠️ No documents found. Skipping vector store creation.")
        return None

    # 🔹 Split text into smaller chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = splitter.split_documents(documents)

    # 🔹 Initialize Bedrock embeddings
    bedrock_client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    embeddings = BedrockEmbeddings(client=bedrock_client, model_id="amazon.titan-embed-text-v1")

    # 🔹 Store in FAISS
    vectorstore = FAISS.from_documents(text_chunks, embeddings)
    vectorstore.save_local(VECTORSTORE_PATH)

    print("✅ Confluence content indexed successfully in FAISS with chunking.")
    return vectorstore


def load_or_create_vector_store(parent_page_id, max_pages=200):
    """Loads FAISS index if available, otherwise fetches child pages from a specific Confluence section."""
    if os.path.exists(f"{VECTORSTORE_PATH}/index.faiss") and os.path.exists(f"{VECTORSTORE_PATH}/index.pkl"):
        try:
            print("🔍 Found existing FAISS index. Loading...")
            bedrock_client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
            embeddings = BedrockEmbeddings(client=bedrock_client, model_id="amazon.titan-embed-text-v1")
            vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
            print("✅ Loaded existing FAISS index.")
            return vectorstore

        except Exception as e:
            print(f"⚠️ Failed to load FAISS index: {str(e)}. Rebuilding index...")

    # 🔹 Fetch pages under the specific parent page
    documents = get_all_child_pages_recursive(parent_page_id, max_pages)
    if documents:
        return create_faiss_vector_store(documents)
    else:
        print("❌ No Confluence data available. Exiting.")
        return None


def create_rag_chain(vectorstore):
    """Creates a RAG pipeline with similarity search."""
    bedrock_client = boto3.client("bedrock-runtime", region_name=AWS_REGION)

    # Initialize LLM with Claude 3 Sonnet (a newer model)
    llm = BedrockChat(
        client=bedrock_client,
        model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",  # Correct model ID
        model_kwargs={
            "temperature": 0,
            "max_tokens": 500,
            "anthropic_version": "bedrock-2023-05-31"
        }
    )
    # 🔹 Use **similarity search** to retrieve the most relevant chunks
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # 🔹 Create Retrieval-Augmented Generation (RAG) pipeline
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)


def main():
    """Main function to load vector store and run Q&A loop."""
    parent_page_id = CONFLUENCE_SPACEKEY

    # 🔹 Load or create the FAISS index
    vectorstore = load_or_create_vector_store(parent_page_id)

    if not vectorstore:
        print("❌ No vector store available. Exiting.")
        return

    # 🔹 Create RAG system
    qa_chain = create_rag_chain(vectorstore)

    print("\n✅ System ready! Type 'quit' to exit.")

    # Start Q&A loop
    while True:
        question = input("\nAsk a question: ")
        if question.lower() == "quit":
            break

        result = qa_chain.invoke({"query": question})
        print("\n🔹 Answer:", result["result"])


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
