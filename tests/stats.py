import sys
import os
import networkx as nx
import chromadb
from pathlib import Path

# Fix python path if needed
sys.path.insert(0, os.path.abspath("."))
from exact_pipeline.core.config import Settings

settings = Settings.from_env()

print("="*50)
print("📊 EXACT PIPELINE DATABASE STATISTICS")
print("="*50)

# 1. ChromaDB Stats
chroma_path = settings.project_root / "dataset" / "chromadb_data"
try:
    client = chromadb.PersistentClient(path=str(chroma_path))
    collections = client.list_collections()
    print("\n🗄️ VECTOR DB (ChromaDB) - Semantic Search Index")
    print(f"Path: {chroma_path}")
    print("-" * 40)
    for col in collections:
        print(f" - Collection '{col.name}': {col.count()} items")
except Exception as e:
    print(f"Error reading ChromaDB: {e}")

# 2. GraphDB Stats
graph_dir = settings.graph_dir
physics_graph_path = graph_dir / "physics_graph.graphml"
logic_graph_path = graph_dir / "logic_graph.graphml"

print("\n🕸️ GRAPH DB (NetworkX) - Symbolic Reasoning Graph")
print(f"Path: {graph_dir}")
print("-" * 40)

def print_graph_stats(path, name):
    if os.path.exists(path):
        try:
            G = nx.read_graphml(path)
            nodes = G.number_of_nodes()
            edges = G.number_of_edges()
            static = sum(1 for _, data in G.nodes(data=True) if data.get('is_dynamic') == 'False' or data.get('is_dynamic') == False)
            dynamic = nodes - static
            print(f"\n{name} Graph:")
            print(f" - Total Nodes (Formulas/Rules): {nodes}")
            print(f"   ↳ Static (Hardcoded): {static}")
            print(f"   ↳ Dynamic (LLM Extracted): {dynamic}")
            print(f" - Total Edges (Relationships): {edges}")
            
            if "physics" in name.lower():
                print(f"   (Edges represent 'shares_variable' relationships)")
            else:
                print(f"   (Edges represent 'feeds_into' relationships)")
                
        except Exception as e:
            print(f"Error reading {name} Graph: {e}")
    else:
        print(f"\n{name} Graph: File not found ({path})")

print_graph_stats(physics_graph_path, "Physics")
print_graph_stats(logic_graph_path, "Logic")

print("\n" + "="*50)
