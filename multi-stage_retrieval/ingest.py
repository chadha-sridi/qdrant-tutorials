import uuid
import httpx
import logging
import asyncio
from typing import List, Dict, Any
from qdrant_client import models
from langchain_core.documents import Document
from langchain_community.document_loaders import ArxivLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import qdrant_client, dense_model, colbert_model, COLLECTION_NAME

logger = logging.getLogger(__name__)

# TEXT SPLITTER
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MIN_CHUNK_LENGTH = 200
SEPARATORS = ["\n\n", "\n", ".", ";", ",", " "]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=SEPARATORS,
)

async def get_paper_stats(arxiv_id: str) -> Dict[str, Any]:
    """
    Fetches citation count and publication year from Semantic Scholar.
    Essential for the 'Score Boosting' stage of retrieval.
    """
    url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}"
    params = {"fields": "citationCount,year"}
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
    except Exception as e:
        logger.warning(f"Metadata enrichment failed for {arxiv_id}: {e}")
    return {"citationCount": 0, "year": None}

def preprocess(user_id: str, doc: Document, arxiv_id: str) -> List[Document]:
    """
    Clean a document by removing references and splitting into chunks.
    Returns a list of Document objects (chunks).
    """
    content = doc.page_content
    if "References" in content:
        content = content[:content.index("References")]
    doc.page_content = content
    chunks = []
    for c in text_splitter.split_documents([doc]):
        if len(c.page_content) > MIN_CHUNK_LENGTH: # Filter out tiny chunks
            chunks.append(c)
    return chunks

async def ingest_paper(user_id: str, arxiv_id: str):
    """
    The main ingestion worker. 
    Downloads -> Chunks -> Local Embedding -> Cloud Batch Upsert.
    """
    try:
        # 1. Load Data
        logger.info(f"📥 Fetching {arxiv_id}...")
        docs = await ArxivLoader(query=arxiv_id).aload()
        if not docs:
            raise ValueError("ArXiv returned no content.")
        
        # 2. Process & Chunk
        doc = docs[0]
        chunks = preprocess(user_id, doc, arxiv_id) 
        texts = [c.page_content for c in chunks]

        # 3. Enrich with Stats : citation count 
        stats = await get_paper_stats(arxiv_id)
        pub_year = stats.get("year") 

        # 4. Generate Embeddings 
        logger.info(f"Encoding {len(texts)} chunks...")
        # BGE-Base (Dense) + ColBERT (Multi-vector)
        dense_embeds = list(dense_model.embed(texts))
        colbert_embeds = list(colbert_model.embed(texts))

        # 5. Construct Qdrant Points
        points = []
        for i, text in enumerate(texts):
            points.append(models.PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "bge_dense": dense_embeds[i].tolist(),
                    "colbert": colbert_embeds[i].tolist() # Matrix: [tokens x 128]
                },
                payload={
                    "page_content": text,
                    "user_id": user_id,
                    "paper_id": arxiv_id,
                    "citation_count": stats.get("citationCount", 0),
                    "published_year": pub_year
                }
            ))

        # 6. Batch Upsert to Qdrant Cloud (Avoiding the 32MB JSON Limit)
        # 25 chunks is the 'Safe Zone' for ColBERT multi-vectors
        BATCH_SIZE = 25 
        for i in range(0, len(points), BATCH_SIZE):
            batch = points[i : i + BATCH_SIZE]
            await qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=batch,
                wait=True # Ensures data is searchable immediately
            )
            logger.info(f"📤 Uploaded batch {i//BATCH_SIZE + 1}/{(len(points)-1)//BATCH_SIZE + 1}")

        logger.info(f"✅ Ingestion Successful: {arxiv_id}")
        return True

    except Exception as e:
        logger.error(f"❌ Critical Ingestion Error: {str(e)}")
        return False

if __name__ == "__main__":
    asyncio.run(ingest_paper("demo_user", "1810.04805"))