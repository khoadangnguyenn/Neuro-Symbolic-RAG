import sys
import os
sys.path.insert(0, os.path.abspath("."))
from exact_pipeline.knowledge.knowledge import get_physics_knowledge_index
from exact_pipeline.core.config import Settings

settings = Settings.from_env()
db = get_physics_knowledge_index(
    db_path=str(settings.project_root / "dataset" / "chromadb_data"),
    graph_path=str(settings.graph_dir / "physics_graph.graphml"),
    alpha=settings.alpha
)

query = "Three electric charges, q1 = q2 = q3 = 2.4 × 10^-19 C, are placed at the three vertices of an equilateral triangle ABC with side length 16 cm in air. Determine the net electric force vector acting on q3"
print(f"Using alpha={db.alpha}")
print("Vector Hits:")
for hit in db.vector_db.search(query, k=5):
    print(f" - {db.id_fn(hit.item)}: {hit.score}")

print("\nHybrid Hits:")
for hit in db.search(query, k=5):
    print(f" - {db.id_fn(hit.item)}: {hit.score} (vec={hit.vector_score}, pr={hit.pagerank_score})")

