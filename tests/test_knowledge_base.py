
import unittest
from pathlib import Path
import tempfile
import shutil
from src.services.knowledge_base import KnowledgeBaseService

class TestKnowledgeBaseService(unittest.TestCase):
    def setUp(self):
        # Crear un directorio temporal para documentos de prueba
        self.test_dir = Path(tempfile.mkdtemp())
        
        # Crear algunos archivos markdown de prueba
        (self.test_dir / "doc1.md").write_text("Hello world", encoding="utf-8")
        (self.test_dir / "doc2.md").write_text("The quick brown fox jumps over the lazy dog", encoding="utf-8")
        (self.test_dir / "doc3.md").write_text("Atheria is a simulation of quantum cellular automata", encoding="utf-8")
        (self.test_dir / "ignored.txt").write_text("This should be ignored", encoding="utf-8")
        
        self.kb = KnowledgeBaseService(docs_path=str(self.test_dir))

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_load_documents(self):
        self.kb.load_documents()
        self.assertEqual(len(self.kb.documents), 3)
        filenames = [d['filename'] for d in self.kb.documents]
        self.assertIn("doc1.md", filenames)
        self.assertIn("doc2.md", filenames)
        self.assertIn("doc3.md", filenames)
        self.assertNotIn("ignored.txt", filenames)

    def test_search_exact(self):
        self.kb.build_index()
        results = self.kb.search("Atheria")
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0]['filename'], "doc3.md")

    def test_search_relevance(self):
        self.kb.build_index()
        results = self.kb.search("quick fox")
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0]['filename'], "doc2.md")

    def test_empty_search(self):
        self.kb.build_index()
        results = self.kb.search("nonexistentword12345")
        self.assertEqual(len(results), 0)

if __name__ == '__main__':
    unittest.main()
