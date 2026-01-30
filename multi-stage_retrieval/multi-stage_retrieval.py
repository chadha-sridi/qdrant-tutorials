import logging
import asyncio
from qdrant_client import models
from config import qdrant_client, dense_model, colbert_model, COLLECTION_NAME

logger = logging.getLogger(__name__)

async def advanced_discovery_search(user_id: str, query_text: str, top_k: int = 5):
    # Generate query embeddings
    query_dense = list(dense_model.embed([query_text]))[0].tolist()
    query_multivector = list(colbert_model.embed([query_text]))[0].tolist()
    response = await qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query_filter=models.Filter(
            must=[models.FieldCondition(
                key="user_id", 
                match=models.MatchValue(value=user_id)
            )]
        ),
        prefetch=[
            models.Prefetch(
                prefetch=[
                    # --- STAGE 1: Semantic Search ---
                    models.Prefetch(
                        query=query_dense,
                        using="bge_dense",
                        limit=50, # Get a healthy pool of candidates
                    )
                ],
                # --- STAGE 2: Score Boosting ---
                # Apply custom business logic boosting on the 50 semantic candidates
                query=models.FormulaQuery(
                    formula=models.SumExpression(
                        sum=[
                            "$score",  # Reference Stage 1 score
                            # Boost by Citation: 0.1 * ln(1 + citation_count)
                            models.MultExpression(mult=[
                                0.1, 
                                models.LnExpression(ln=models.SumExpression(sum=[1.0, "citation_count"]))
                            ]),
                            # Boost by Year: 
                            models.MultExpression(mult=[
                                0.05, 
                                models.SumExpression(sum=["published_year", -2000.0])
                            ])
                        ]
                    )
                ),
                limit=30, # Narrow down to top 30 boosted papers for the re-ranker
            )
        ],
        # --- STAGE 3: ColBERT Re-ranking ---
        # Run on the 30 papers that survived both similarity and boosting
        query=query_multivector,
        using="colbert",
        limit=top_k,
        with_payload=True
    )
    return response.points

if __name__ == "__main__":
    asyncio.run(advanced_discovery_search(user_id="demo_user", query_text="what is bidirectional attention"))