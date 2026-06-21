import os
import tempfile
import unittest
from unittest.mock import patch

from exact_pipeline.knowledge.retrieval import VectorDBIndex


class RetrievalOfflineTest(unittest.TestCase):
    def test_missing_embedding_model_falls_back_without_network(self):
        items = ["capacitor voltage energy", "employee training badge"]
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"EXACT_RETRIEVAL_MODE": "auto"}, clear=False
        ), patch(
            "exact_pipeline.knowledge.retrieval.embedding_functions."
            "SentenceTransformerEmbeddingFunction",
            side_effect=OSError("model is not cached"),
        ):
            index = VectorDBIndex.from_items(
                "offline_test",
                items,
                text_fn=lambda item: item,
                id_fn=lambda item: item,
                db_path=directory,
            )

        hits = index.search("training employee", k=2, rerank_top_k=2)
        self.assertEqual(hits[0].item, "employee training badge")
        self.assertIsNone(index.embedding_function)


if __name__ == "__main__":
    unittest.main()
