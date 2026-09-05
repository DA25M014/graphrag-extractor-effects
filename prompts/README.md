# Prompts

`extract_graph.txt` and `summarize_descriptions.txt` are Microsoft
GraphRAG's own default prompt templates, copied verbatim from a workspace
created by `graphrag init` (graphrag 3.1.2, MIT licensed, copyright
Microsoft Corporation).

They are committed rather than generated for two reasons. A clone can run
the LLM extraction path without a separate `graphrag init` step, and the
exact prompt that produced `results/extraction_models.csv` is pinned, which
matters because every finding in this repository is conditional on it and
no prompt ablation was run.

`setup_index.py` copies this directory into each LLM workspace.
