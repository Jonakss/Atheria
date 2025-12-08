
import os
import re
from pathlib import Path
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi

class KnowledgeBaseService:
    """
    Servicio para indexar y buscar documentos en la Knowledge Base (docs/) usando BM25.
    """
    
    def __init__(self, docs_path: str = None):
        if docs_path is None:
            # Por defecto busca docs/ en la raíz del proyecto
            # Asumimos que este archivo está en src/services/
            project_root = Path(__file__).parent.parent.parent
            self.docs_path = project_root / "docs"
        else:
            self.docs_path = Path(docs_path)
            
        self.documents = []
        self.bm25 = None
        self.is_ready = False
        
    def _tokenize(self, text: str) -> List[str]:
        """Tokenización simple: minúsculas y palabras alfanuméricas."""
        text = text.lower()
        # Eliminar caracteres especiales y números sueltos si se desea, 
        # pero para código a veces es útil mantenerlos.
        # Aquí usamos una tokenización básica.
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def load_documents(self):
        """Carga recursiva de archivos Markdown desde docs_path."""
        if not self.docs_path.exists():
            print(f"⚠️  Knowledge Base path not found: {self.docs_path}")
            return

        self.documents = []
        
        # Recorrer recursivamente
        for file_path in self.docs_path.rglob("*.md"):
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                
                # Ignorar archivos muy pequeños o vacíos
                if len(content.strip()) < 10:
                    continue
                    
                # Crear entrada de documento
                doc = {
                    "path": str(file_path),
                    "relative_path": str(file_path.relative_to(self.docs_path.parent)),
                    "content": content,
                    "filename": file_path.name
                }
                self.documents.append(doc)
            except Exception as e:
                print(f"⚠️  Error reading {file_path}: {e}")
                
        print(f"📚 Loaded {len(self.documents)} documents from {self.docs_path}")

    def build_index(self):
        """Construye el índice BM25 con los documentos cargados."""
        if not self.documents:
            self.load_documents()
            
        if not self.documents:
            print("⚠️  No documents to index.")
            return

        tokenized_corpus = [self._tokenize(doc["content"]) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.is_ready = True
        print("✅ Knowledge Base Index built successfully.")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Busca documentos relevantes para la query."""
        if not self.is_ready:
            self.build_index()
            
        if not self.bm25:
            return []
            
        tokenized_query = self._tokenize(query)
        # Obtenemos scores y filtramos
        scores = self.bm25.get_scores(tokenized_query)
        
        # Crear lista de (score, doc)
        scored_docs = []
        for i, score in enumerate(scores):
            if score > 0:
                doc = self.documents[i].copy()
                doc["score"] = score
                scored_docs.append(doc)
                
        # Ordenar por score descendente
        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        
        return scored_docs[:top_k]
