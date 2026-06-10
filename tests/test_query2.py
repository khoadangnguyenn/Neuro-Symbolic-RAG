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

print(f"Collection count: {db.vector_db.collection.count()}")
print(f"Items in python: {len(db.vector_db.items)}")
