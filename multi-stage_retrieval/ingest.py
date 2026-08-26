import uuid
import httpx
import logging
import asyncio
import re
from typing import List, Dict, Any
from qdrant_client import models
from langchain_core.documents import Document
from langchain_community.document_loaders import ArxivLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import (
    COLLECTION_NAME,
    colbert_model,
    dense_model,
    ensure_collection,
    qdrant_client,
)

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
        async with httpx.AsyncClient(timeout=10) as client:
            for attempt in range(3):
                response = await client.get(url, params=params)
                if response.status_code == 200:
                    return response.json()
                if response.status_code != 429 or attempt == 2:
                    response.raise_for_status()
                await asyncio.sleep(2**attempt)
    except httpx.HTTPError as error:
        logger.warning("Metadata enrichment failed for %s: %s", arxiv_id, error)
    return {"citationCount": 0, "year": None}


def preprocess(doc: Document) -> List[Document]:
    """
    Clean a document by removing references and splitting into chunks.
    Returns a list of Document objects (chunks).
    """
    content = doc.page_content
    reference_heading = re.search(r"(?im)^\s*references\s*$", content)
    if reference_heading:
        content = content[:reference_heading.start()]
    doc.page_content = content
    chunks = []
    for c in text_splitter.split_documents([doc]):
        if len(c.page_content) > MIN_CHUNK_LENGTH:
            chunks.append(c)
    return chunks

async def ingest_paper(user_id: str, arxiv_id: str):
    """
    The main ingestion worker. 
    Downloads -> Chunks -> Local Embedding -> Cloud Batch Upsert.
    """
    try:
        await ensure_collection()

        # 1. Load Data
        logger.info("Fetching %s...", arxiv_id)
        docs = await ArxivLoader(query=arxiv_id).aload()
        if not docs:
            raise ValueError("ArXiv returned no content.")
        
        # 2. Process & Chunk
        doc = docs[0]
        chunks = preprocess(doc)
        texts = [c.page_content for c in chunks]
        if not texts:
            raise ValueError("The paper produced no chunks after preprocessing.")

        # 3. Enrich with Stats : citation count 
        stats = await get_paper_stats(arxiv_id)
        pub_year = stats.get("year") or 2000

        # 4. Generate Embeddings 
        logger.info("Encoding %s chunks...", len(texts))
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
                    "colbert": colbert_embeds[i].tolist()
                },
                payload={
                    "page_content": text,
                    "user_id": user_id,
                    "paper_id": arxiv_id,
                    "citation_count": stats.get("citationCount") or 0,
                    "published_year": pub_year,
                    }
            ))

        # 6. Batch Upsert to Qdrant Cloud (Avoiding the 32MB JSON Limit)
        BATCH_SIZE = 25 
        for i in range(0, len(points), BATCH_SIZE):
            batch = points[i : i + BATCH_SIZE]
            await qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=batch,
                wait=True # Ensures data is searchable immediately
            )
            logger.info(
                "Uploaded batch %s/%s",
                i // BATCH_SIZE + 1,
                (len(points) - 1) // BATCH_SIZE + 1,
            )

        logger.info("Ingestion successful: %s", arxiv_id)
        return True

    except Exception as e:
        logger.error("Critical ingestion error: %s", e)
        return False

if __name__ == "__main__":
    asyncio.run(ingest_paper("demo_user", "1810.04805"))