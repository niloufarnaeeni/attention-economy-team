# attention-economy-team

This repository contains the datasets and codes used for studying team formation and actor retrieval in incentivized attention economies.

---

## Data Structure

The `data/` folder contains three datasets:

- `cookie_fun`
- `giverep`
- `kaito`

Each dataset includes two main folders:

---

### dataset/

This folder contains the structured metadata used for graph construction, feature extraction, and attention-aware modeling.

It includes:

**attention_scores.csv**  
Actor-level attention signals.

**creator_details.csv**  
Creator metadata (project associations, profile information).

**creators.csv**  
Canonical list of creators/actors.

**skills.csv**  
Extracted skill taxonomy.

**gpt5_skills.csv**  
LLM-enhanced skill annotations. Each skill includes:

- `id`
- `keyword`
- `level`
- `categories`
- `one_sentence_definition`

Example:

```
S1 | Layer 1 blockchain | Generic | ['Layer 1 / Core Protocols'] | A base-layer public blockchain that provides consensus, settlement, and execution for native assets and smart contracts.
```

**rootdata_descriptions.csv**  
Project-level descriptions obtained from RootData.

These files collectively define the full structured dataset and can be used to build graph-based representations.

---

### ranker_input/

This folder contains the input for text-based ranking models.

**ranker_corpus.jsonl**  
Supervised ranking instances used for training and evaluating text-based rankers.
