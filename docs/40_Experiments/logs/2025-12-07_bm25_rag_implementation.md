# 2025-12-07: BM25 RAG for Knowledge Base

**Fecha:** 2025-12-07  
**Tipo:** Feature  
**Tags:** #feature #rag #cli #knowledge-base

## Contexto

La documentación de Atheria en `docs/` es extensa. Para mejorar la eficiencia del desarrollo y la consulta de información, se implementó un sistema RAG (Retrieval Augmented Generation) ligero basado en BM25.

## Implementación

### Nuevo Servicio: `src/services/knowledge_base.py`

- **`KnowledgeBaseService`**: Clase que maneja la indexación y búsqueda.
  - Carga recursiva de archivos `.md` desde `docs/`.
  - Tokenización simple y eficiente.
  - Índice BM25 usando `rank_bm25`.
  - Filtrado por score > 0 para evitar resultados irrelevantes.

### Modificación del CLI: `src/cli.py`

- Agregado argumento global `-q` / `--query`.
- Integración con `KnowledgeBaseService`.
- Resultados muestran: filename, ruta relativa, y snippet.

### Dependencia

- Agregado `rank_bm25>=0.2.2` a `requirements.txt`.

## Verificación

- **Tests unitarios**: `tests/test_knowledge_base.py` (4 tests, todos OK).
- **Prueba manual**: `ath -q "Harmonic Engine"` retorna documentos relevantes.

## Uso

```bash
atheria -q "your search query"
# OR
ath -q "your search query"
```

## Relacionado

- [[CLI_TOOL]]: Documentación del CLI
- [[00_KNOWLEDGE_BASE]]: Knowledge Base principal
