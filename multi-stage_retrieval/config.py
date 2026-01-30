import os
import logging
import asyncio
from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient, models
from fastembed import TextEmbedding, LateInteractionTextEmbedding 
logger = logging.getLogger(__name__)
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
qdrant_client = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=300)

# Embedding models 
dense_model = TextEmbedding(model_name="BAAI/bge-base-en-v1.5")
colbert_model = LateInteractionTextEmbedding(model_name="colbert-ir/colbertv2.0")
# === Collection creation ===
COLLECTION_NAME = "demo_collection"
async def init_vectdb():
    collections_response = await qdrant_client.get_collections()
    exists = any(c.name == COLLECTION_NAME for c in collections_response.collections)

    if not exists:
        logger.info(f"Creating collection: {COLLECTION_NAME}")
        
        # Create a collection with Named Vectors
        await qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "bge_dense": models.VectorParams(
                    size=768, # BGE embedding dimension
                    distance=models.Distance.COSINE
                ),
                "colbert": models.VectorParams(
                    size=128, # ColBERT token dimension
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    )
                )
            },
        )
        # Create payload indexes 
        await qdrant_client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="user_id",
            field_schema=models.KeywordIndexParams(
                type=models.KeywordIndexType.KEYWORD,
                is_tenant=True,  # Multi-tenant setup based on user_id
            ),
        )
        other_payloads = {
            "citation_count": models.PayloadSchemaType.INTEGER,
            "published_year": models.PayloadSchemaType.INTEGER,
            "paper_id": models.PayloadSchemaType.KEYWORD,
        }
        for field, schema in other_payloads.items():
            await qdrant_client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field,
                field_schema=schema,
            )
        logger.info("Vector Database initialized.")

if __name__ == "__main__":
    asyncio.run(init_vectdb())