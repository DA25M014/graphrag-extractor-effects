"""Build one GraphRAG workspace per corpus, with the pipeline trimmed to
the steps this study actually needs.

What is measured here is the entity graph itself. That needs neither
community summaries nor text embeddings nor claim extraction, and those
are the steps that cost money. Trimming them is not a shortcut:
`create_community_reports` cannot change the graph it summarizes, so
dropping it leaves the compared object identical.

Two extraction modes:
  nlp   `extract_graph_nlp`, regex_english analyzer. No LLM, no API key,
        zero cost. Produces a real entity graph from a real corpus.
  llm   `extract_graph`, the default LLM extraction GraphRAG ships and
        the one the headline claim is about. Costs money; see
        cost_model.py.
  local Same LLM extraction workflow as `llm`, but pointed at an
        OpenAI-compatible endpoint on localhost (Ollama). Zero cost, no
        API key, no card. The extracted graph is still LLM-built, which
        is the property this study's claim depends on; only the model
        differs, and the model is recorded in the CSV.

Both modes leave `cluster_graph.max_cluster_size` at GraphRAG's default
of 10, because that setting is part of what is being compared.
"""

import os
import shutil
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
CORPORA = os.path.join(HERE, 'corpora')

# Everything up to and including community assignment, and nothing after.
WORKFLOWS = {
    'llm': ['load_input_documents', 'create_base_text_units',
            'create_final_documents', 'extract_graph', 'finalize_graph',
            'create_communities'],
    'local': ['load_input_documents', 'create_base_text_units',
              'create_final_documents', 'extract_graph',
              'finalize_graph', 'create_communities'],
    'nlp': ['load_input_documents', 'create_base_text_units',
            'create_final_documents', 'extract_graph_nlp',
            'prune_graph', 'finalize_graph', 'create_communities'],
}

SETTINGS = """\
completion_models:
  default_completion_model:
    model_provider: openai
    model: {model}{api_base_line}{local_caps}
    auth_method: api_key
    api_key: ${{GRAPHRAG_API_KEY}}
    retry:
      type: exponential_backoff

embedding_models:
  default_embedding_model:
    model_provider: openai
    model: text-embedding-3-small
    auth_method: api_key
    api_key: ${{GRAPHRAG_API_KEY}}
    retry:
      type: exponential_backoff

input:
  type: text

chunking:
  type: tokens
  size: 1200
  overlap: 100
  encoding_model: o200k_base

input_storage:
  type: file
  base_dir: "input"

output_storage:
  type: file
  base_dir: "output"

reporting:
  type: file
  base_dir: "logs"

cache:
  type: json
  storage:
    type: file
    base_dir: "cache"

vector_store:
  type: lancedb
  db_uri: output/lancedb

# Only the steps this study needs. Community reports, text embeddings and
# claim extraction are omitted: none of them can change the entity graph
# or its community assignment, and they are where the cost is.
workflows: {workflows}

extract_graph:
  completion_model_id: default_completion_model
  prompt: "prompts/extract_graph.txt"
  entity_types: [organization,person,geo,event]
  # Gleanings is a follow-up round telling the model it MISSED entities.
  # A 7B-8B local model can loop on it indefinitely: an observed run
  # stalled for 20+ minutes on a single chunk. It is also half the calls.
  # Kept at GraphRAG's default of 1 for paid backends.
  max_gleanings: {max_gleanings}

summarize_descriptions:
  completion_model_id: default_completion_model
  prompt: "prompts/summarize_descriptions.txt"
  max_length: 500

extract_graph_nlp:
  text_analyzer:
    extractor_type: regex_english

# GraphRAG's own default. Part of what is being compared; do not change.
cluster_graph:
  max_cluster_size: 10

extract_claims:
  enabled: false

snapshots:
  graphml: true
  embeddings: false
"""


# Ollama exposes an OpenAI-compatible API here. litellm talks to it as
# an 'openai' provider with api_base overridden.
LOCAL_API_BASE = 'http://localhost:11434/v1'


def build(corpus, mode, model='gpt-4o-mini'):
    name = os.path.splitext(os.path.basename(corpus))[0]
    root = os.path.join(HERE, 'idx', f'{name}_{mode}')
    os.makedirs(os.path.join(root, 'input'), exist_ok=True)
    shutil.copy(corpus, os.path.join(root, 'input',
                                     os.path.basename(corpus)))

    # LLM extraction needs the shipped prompt files; NLP does not.
    probe_prompts = os.path.join(HERE, 'idx', '_probe', 'prompts')
    if mode in ('llm', 'local') and os.path.isdir(probe_prompts):
        dst = os.path.join(root, 'prompts')
        if not os.path.isdir(dst):
            shutil.copytree(probe_prompts, dst)

    is_local = mode == 'local'
    api_base_line = (f"\n    api_base: {LOCAL_API_BASE}"
                     if is_local else '')
    # Cap generation so one rambling chunk cannot stall the whole run.
    local_caps = "\n    max_tokens: 2048" if is_local else ''
    with open(os.path.join(root, 'settings.yaml'), 'w') as f:
        f.write(SETTINGS.format(model=model, api_base_line=api_base_line,
                                local_caps=local_caps,
                                max_gleanings=0 if is_local else 1,
                                workflows=WORKFLOWS[mode]))
    with open(os.path.join(root, '.env'), 'w') as f:
        # Ollama ignores the key but graphrag requires one to be set.
        f.write("GRAPHRAG_API_KEY=ollama-local-no-key-needed\n"
                if mode == 'local'
                else "GRAPHRAG_API_KEY=${GRAPHRAG_API_KEY}\n")
    print(f"[setup] {root}  mode={mode}  "
          f"workflows={len(WORKFLOWS[mode])} steps")
    return root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['nlp', 'llm', 'local'],
                    default='nlp')
    ap.add_argument('--model', default='gpt-4o-mini')
    ap.add_argument('--corpora', nargs='*')
    a = ap.parse_args()

    files = a.corpora or sorted(
        os.path.join(CORPORA, f) for f in os.listdir(CORPORA)
        if f.endswith('.txt'))
    for c in files:
        build(c, a.mode, a.model)


if __name__ == '__main__':
    main()
