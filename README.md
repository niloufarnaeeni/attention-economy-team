# attention-economy-team

This repository contains the datasets and codes used for studying team formation and actor retrieval in incentivized attention economies.

The `data/` folder includes three datasets: `cookie_fun`, `giverep`, and `kaito`.

Each dataset contains:

### dataset/
Structured metadata used for graph construction and attention-aware modeling, including:

- **attention_scores.csv** – Actor-level attention signals.
- **creator_details.csv** – Creator metadata (project associations and profile information).
- **creators.csv** – Canonical list of creators/actors.
- **skills.csv** – Extracted skill taxonomy.
- **gpt5_skills.csv** – LLM-enhanced skill annotations (`id`, `keyword`, `level`, `categories`, `one_sentence_definition`) providing semantic grounding for skill-based modeling and retrieval.
- **rootdata_descriptions.csv** – Project-level descriptions obtained from RootData.

These files collectively define the full structured dataset and can be used to build both graph-based representations and text-based models.

### ranker_input/
- **ranker_corpus.jsonl** – Supervised ranking instances used as input for text-based ranking models.

### text_retrieval/

Text-based actor retrieval using cross-encoder ranking models, including MiniLM (`cross-encoder/ms-marco-MiniLM-L-6-v2`) and BGE Reranker (`BAAI/bge-reranker-base`). This module supports training and evaluation pipelines.

### graph_retrieval/

Graph-based actor retrieval using GNN models such as MetaPath2Vec, GCN, and GraphSAGE for representation learning and recommendation.

### skill_extraction/

GPT-based structured skill extraction pipeline for building and enriching the project skill taxonomy.

