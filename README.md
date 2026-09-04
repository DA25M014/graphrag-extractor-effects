# graphrag-extractor-effects

**The entity extractor determines the graph, and papers reporting
GraphRAG results almost never say what the resulting graph looked like.**

Microsoft GraphRAG turns a corpus into an entity graph, partitions that
graph, and summarizes the partitions. Everything downstream depends on the
graph. But the graph is not a property of the corpus: it is a property of
the corpus *and* whatever extracted the entities.

This repository holds the same corpus through the same GraphRAG pipeline,
the same chunker and the same clusterer, changing only the extractor. The
resulting graphs are not small variations on each other. They differ by an
order of magnitude in edge count, by a factor of seven in density, and in
one case the extractor silently broke entity resolution by translating
entity names into another language.

## The result

A Christmas Carol, indexed four ways. From
`results/extraction_models.csv`:

| Extractor | entities | nodes (LCC) | edges | mean degree | off the LCC | non-ASCII titles | entity types |
|---|---|---|---|---|---|---|---|
| `regex_english` (no LLM) | 161 | 161 | 1,466 | 18.2 | 0% | 0% | 1 |
| llama3.1:8b (local) | 291 | 97 | 125 | 2.6 | 67% | 0% | 14 |
| qwen2.5:7b-instruct (local) | 181 | 104 | 145 | 2.8 | 43% | 8% | 11 |

Three things are worth stating plainly.

**The graphs differ by an order of magnitude.** The regex path yields
11.7x as many edges as llama3.1 and 10.1x as many as qwen2.5, at 7.1x the
mean degree. This is not a quality gap in either direction. GraphRAG's
`extract_graph_nlp` path links noun phrases that co-occur, so it produces
a dense co-occurrence graph. LLM extraction emits only relations the model
states explicitly, so it produces a sparse assertion graph. They are
different objects that happen to share a file format.

**Most extracted entities never reach the graph.** llama3.1 extracted 291
entities and left 67% of them off the largest connected component, so two
thirds of what it found plays no part in any community. qwen2.5 leaves
43% off. The regex path leaves none. An entity that is extracted but
disconnected costs tokens at extraction time and contributes nothing after
it.

**qwen2.5 translated the entity names.** It emitted Chinese renderings of
English proper nouns (Charles Dickens, Scrooge) alongside untranslated
ones, and a type vocabulary mixing Chinese and English labels. 8.3% of its
entity titles contain non-ASCII letters. The translated and untranslated
forms of one entity become two nodes, so entity resolution fails silently:
nothing errors, the pipeline completes, and the graph is wrong. The
`nonascii_title_frac` column exists to catch exactly this, and it is
cheap enough that any pipeline could compute it.

![extractor effects](results/f1_extractor_effects.png)

## Why this matters

If you read a paper reporting that some method achieves some number on
"the GraphRAG entity graph for Cora", you cannot reconstruct the object it
was measured on. Papers do generally name the extraction model. What they
do not report is the resulting graph's structural statistics, and the
table above is the reason that gap matters: two reasonable extractor
choices give graphs that differ by 10x in edge count on the same text.

The practical recommendation is one line long. **Report n, m, mean degree
and the fraction of extracted entities off the largest connected component
alongside any result computed on an LLM-built knowledge graph.** All four
are free to compute and they pin down the object.

## The other corpora

Two further Project Gutenberg corpora are indexed through the no-LLM path,
which is the arm cheap enough to run at every size:

| Corpus | nodes (LCC) | edges | mean degree |
|---|---|---|---|
| A Christmas Carol | 161 | 1,466 | 18.2 |
| The Adventures of Sherlock Holmes | 669 | 9,299 | 27.8 |
| Moby Dick | 1,888 | 37,452 | 39.7 |

## Reproducing this

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt

# Build one GraphRAG workspace per corpus.
./.venv/bin/python setup_index.py --mode nlp

# The no-LLM path needs no API key and costs nothing.
GRAPHRAG_API_KEY=unused ./.venv/bin/python -m graphrag index \
    --root idx/christmas_carol_nlp --skip-validation

# Record the shape of every index that exists.
./.venv/bin/python exp_extraction_models.py
./.venv/bin/python make_fig.py
```

The local LLM arms use Ollama, which exposes an OpenAI-compatible API, so
they run at zero cost:

```bash
ollama pull llama3.1:8b
./.venv/bin/python setup_index.py --mode local
```

One practical note. `max_gleanings` is a follow-up round that tells the
model it missed entities. GraphRAG's default is 1, and llama3.1 tolerates
it, but qwen2.5:7b stalled for more than 20 minutes generating on a single
chunk, so `setup_index.py` allows it to be set to 0 for small local models.
Gleanings are also roughly half the LLM calls, so the setting matters for
cost as well as for recall.

## Cost

`cost_model.py` estimates GraphRAG indexing spend before you commit to it,
reading chunk size, overlap and `max_gleanings` from graphrag's own
installed defaults so the estimate tracks the version you have. Prompt and
output sizes per chunk are stated ASSUMPTIONS, marked as such in the
source, because they depend on the corpus.

```bash
./.venv/bin/python cost_model.py --corpus-tokens 250000 --model gpt-4o-mini
```

## Verification

```bash
./.venv/bin/python verify_numbers.py
```

Every number in this README is checked against `results/*.csv` by that
script, which names the file, row and column each must come from. Derived
figures such as the 11.7x edge ratio read **both** operands from the CSV
rather than hand-typing either, since a gate that types its own operands
proves nothing. It exits non-zero on any mismatch.

## Limitations

- **Three corpora, one genre, one language.** All three are 19th-century
  English literary prose from Project Gutenberg. Nothing here establishes
  that the same spread appears on news, code, clinical notes or
  multilingual text, and the qwen2.5 translation failure in particular may
  behave differently on a corpus that is not entirely English.
- **The four-way comparison exists on one corpus.** Only A Christmas
  Carol was indexed by every extractor. The other two are the no-LLM path
  only, so the density trend with corpus size is a single-extractor
  observation.
- **One run per cell.** GraphRAG's extraction is an LLM call, so it is not
  deterministic across runs, and no arm here is repeated to estimate that
  variance. The gaps reported above are much larger than any plausible
  run-to-run variation, but the variation is not measured.
- **This is descriptive.** It reports what the extractors produce. It does
  not evaluate answer quality, retrieval, or which graph is better for any
  downstream task, and nothing here should be read as saying the dense
  co-occurrence graph is preferable to the sparse assertion graph.

## Layout

| Path | Contents |
|---|---|
| `setup_index.py` | Builds one GraphRAG workspace per corpus per extraction mode |
| `gr_adapter.py` | GraphRAG parquet output to an edge list; largest component, simple, self-loop free |
| `exp_extraction_models.py` | Records the shape of every index that exists; writes the CSV |
| `cost_model.py` | Pre-spend indexing cost estimate, read from graphrag's own defaults |
| `make_fig.py` | Regenerates the figure from the CSV and nothing else |
| `verify_numbers.py` | Gate: checks every number in this README against the CSVs |
| `corpora/` | Public-domain source texts (Project Gutenberg, boilerplate stripped) |
| `results/extraction_models.csv` | Every number this repository reports |

## License

MIT, see `LICENSE`. The corpora are public domain via Project Gutenberg.
