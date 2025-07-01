import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from openai import OpenAI
import time

# Set Streamlit page settings
st.set_page_config(page_title="🌐 Website RAG Chatbot", layout="centered")
st.title("🌐 Website RAG Chatbot")

# Set up OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Crawl function
def crawl_site(start_url, max_pages=20):
    visited = set()
    to_visit = [start_url]
    documents = []

    st.info("🌐 Crawling website...")

    progress_placeholder = st.empty()
    progress_bar = progress_placeholder.progress(0)
    crawl_counter = 0

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue

        try:
            res = requests.get(url, timeout=10)
            if not res.ok:
                continue

            visited.add(url)
            crawl_counter += 1

            soup = BeautifulSoup(res.text, "html.parser")
            text = soup.get_text(separator=" ", strip=True)

            documents.append(Document(
                page_content=text,
                metadata={"source": url}
            ))

            # ✅ Generalized link filter
            for tag in soup.find_all("a", href=True):
                link = urljoin(url, tag['href'])
                parsed_base = urlparse(start_url)
                parsed_link = urlparse(link)

                if (
                    parsed_link.netloc == parsed_base.netloc and
                    link not in visited and
                    link not in to_visit
                ):
                    to_visit.append(link)

            st.write(f"✅ Crawled: {url}")

            progress = crawl_counter / max_pages
            progress_bar.progress(min(progress, 1.0))

            time.sleep(0.3)

        except Exception as e:
            st.warning(f"⚠️ Failed to crawl {url}: {e}")

    progress_placeholder.empty()
    return documents



# Session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# Input field
url_input = st.text_input("Enter a website URL to crawl (e.g., https://docs.streamlit.io):")

if url_input and st.button("🚀 Crawl and Index Site"):
    with st.spinner("Processing site..."):
        raw_docs = crawl_site(url_input, max_pages=100)
        st.success(f"✅ Done crawling {len(raw_docs)} pages!")

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = splitter.split_documents(raw_docs)

        # Create embedding model
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",  # use correct model for your embedding size
            api_key=st.secrets["OPENAI_API_KEY"]
        )

        # Create Qdrant client manually
        qdrant_client = QdrantClient(
            url=st.secrets["QDRANT_CLOUD_URL"],
            api_key=st.secrets["QDRANT_API_KEY"],
            timeout=60
        )

        # Recreate collection
        qdrant_client.recreate_collection(
            collection_name="website_vectors",
            vectors_config=VectorParams(size=3072, distance=Distance.COSINE)
        )

        # Initialize LangChain vector store
        vector_store = Qdrant(
            client=qdrant_client,
            collection_name="website_vectors",
            embeddings=embeddings
        )
        # Add in batches with progress bar
        BATCH_SIZE = 10
        num_batches = (len(split_docs) + BATCH_SIZE - 1) // BATCH_SIZE
        upload_progress = st.progress(0, text="📤 Uploading embeddings to Qdrant...")

        for i in range(0, len(split_docs), BATCH_SIZE):
            batch = split_docs[i:i + BATCH_SIZE]
            vector_store.add_documents(batch)
            st.write(f"✅ Uploaded batch {i // BATCH_SIZE + 1}")

            progress = (i + BATCH_SIZE) / len(split_docs)
            upload_progress.progress(min(progress, 1.0),text=f"📤 Uploading batch {i // BATCH_SIZE + 1}/{num_batches}...")
            time.sleep(0.3)

        upload_progress.empty()

        st.session_state.vector_store = vector_store
        st.session_state.messages = []
        st.success("✅ Website processed! You can now ask questions.")

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Ask a new question
if st.session_state.vector_store:
    query = st.chat_input("Ask a question about the website")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.spinner("🔍 Retrieving relevant content..."):
            results = st.session_state.vector_store.similarity_search(query, k=4)
            context = "\n\n\n".join([
                f"Page Content: {doc.page_content[:1000]}...\nURL: {doc.metadata['source']}"
                for doc in results
            ])

        SYSTEM_PROMPT = f"""
        You are an assistant answering questions using content from a website.

        Only use the context below. If the answer is not present, say so.

        Context:
        {context}
        """

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ] + st.session_state.messages

        with st.chat_message("assistant"):
            try:
                response = client.chat.completions.create(
                    model="gpt-4.1",
                    messages=messages
                )
                ai_reply = response.choices[0].message.content
            except Exception as e:
                ai_reply = f"⚠️ OpenAI error: {str(e)}"
                st.error(ai_reply)

            st.markdown(ai_reply)

        st.session_state.messages.append({"role": "assistant", "content": ai_reply})

# Clear chat button
if st.session_state.messages:
    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []

# Footer
st.markdown(
    """
    <hr style="margin-top: 2rem;">
    <div style='text-align: center; color: gray; font-size: 0.9rem'>
        🚀 Built by <b>Manunjay Bhardwaj</b> | Powered by LangChain, OpenAI & Qdrant
    </div>
    """,
    unsafe_allow_html=True
)
