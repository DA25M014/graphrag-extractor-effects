# graphrag-extractor-effects

**In Microsoft GraphRAG, the entity extractor decides how dense the graph
is, and a better LLM does not make it denser.**

GraphRAG turns a corpus into an entity graph, partitions that graph, and
summarizes the partitions. Everything downstream depends on the graph. But
the graph is not a property of the corpus: it is a property of the corpus
*and* whatever extracted the entities.

This repository holds three corpora through GraphRAG, changing the
extractor and holding the chunker and clusterer fixed. Four extractors,
from a regex with no LLM at all up to gpt-4o-mini.

Not affiliated with or endorsed by Microsoft, OpenAI, Meta or Alibaba
Cloud. GraphRAG is Microsoft's, MIT licensed.

## The result

Mean degree, across graphs spanning 97 to 1,888 nodes. From
`results/extraction_models.csv`:

| Extractor | A Christmas Carol | Sherlock Holmes | Moby Dick |
|---|---|---|---|
| `regex_english` (no LLM) | 18.2 | 27.8 | **39.7** |
| gpt-4o-mini (paid API) | 3.01 | 3.09 | **3.30** |
| llama3.1:8b (local) | 2.58 | | |
| qwen2.5:7b (local) | 2.79 | | |

**Three instruction-tuned models spanning 7B to a production API landed
between 2.58 and 3.30 mean degree. The co-occurrence path, over the same
corpora, produced 18 to 40.** The regex graph densifies as the corpus
grows, rising 18.2 to 39.7 while its node count grows 11.7x; gpt-4o-mini
rises only 3.01 to 3.30 over the same range.

The mechanism is not model quality. `extract_graph_nlp` links noun phrases
that co-occur, so more text gives entities more chances to appear together
and the graph densifies. LLM extraction emits only relations a model
states explicitly, and that count stays roughly proportional to the number
of entities.

![extractor effects](results/f1_extractor_effects.png)

## What a better model does buy

Model quality is not irrelevant. It moves how much of the corpus ends up
connected, and it moves whether the extractor follows instructions. It
just does not move density.

| A Christmas Carol | llama3.1:8b | qwen2.5:7b | gpt-4o-mini |
|---|---|---|---|
| entities extracted | 291 | 181 | 201 |
| nodes in the largest component | 97 | 104 | **158** |
| extracted entities left off it | 67% | 43% | **21%** |
| non-ASCII entity titles | 0% | **8.3%** | 0% |
| distinct entity types emitted | 14 | 11 | **5** |

The regex path, for reference, extracts 161 entities and connects all of
them.

Two readings that are easy to get backwards. First, llama3.1 extracted
*more* raw entities than the regex path did, 291 against 161; its problem
is that two thirds of them never joined the main component. So its node
count being 60% of the regex graph's is a statement about connectivity,
not about recall, and no code here checks whether the two extractors found
the *same* entities. Every percentage in this repository compares counts,
never sets.

Second, the LLM arms were configured with a four-type vocabulary
(`organization, person, geo, event`; `setup_index.py`). gpt-4o-mini
emitting 5 types is close to obedience. llama3.1 emitting 14 and qwen2.5
emitting 11 means those models ignored the configured list. That row
measures instruction-following, not descriptive richness.

**qwen2.5 translated the entity names in this run.** It emitted Chinese
renderings of English proper nouns alongside untranslated ones, and a type
vocabulary mixing Chinese and English labels. 8.3% of its entity titles
contain non-ASCII letters. The translated and untranslated forms of one
entity become two nodes, so entity resolution fails silently: nothing
errors, the pipeline completes, and the graph is wrong. The
`nonascii_title_frac` column exists to catch exactly this. gpt-4o-mini
shows 0% here and 0.3% on the two larger corpora.

## The gap widens with corpus size

Because the regex graph densifies with n and the LLM graph does not, the
two diverge as corpora grow. gpt-4o-mini against `regex_english`, as count
ratios:

| Corpus | nodes, as % of the regex graph | edges, as % | mean-degree ratio |
|---|---|---|---|
| A Christmas Carol | 98% | 16% | 6.0x |
| Sherlock Holmes | 84% | 9% | 9.0x |
| Moby Dick | 48% | 4% | 12.0x |

A conclusion drawn on one of these graphs does not transfer to the other,
and the size of the discrepancy depends on how large your corpus is.

## Why this matters

If you read a result reported on "the GraphRAG entity graph" for some
corpus, you cannot reconstruct the object it was measured on. Papers
generally name the extraction model. What they do not report is the
resulting graph's structural statistics, and the tables above are the
reason that matters: two reasonable extractor choices give graphs that
differ by 12x in mean degree on the same text.

The practical recommendation is one line long. **Report n, m, mean degree
and the fraction of extracted entities off the largest connected component
alongside any result computed on an LLM-built knowledge graph.** All four
are free to compute and they pin down the object. A fifth, the fraction of
entity titles that are non-ASCII, costs nothing and catches a silent
entity-resolution failure that a 7B model produced here on its first
attempt.

## What is NOT held constant

The headline says the extractor is what changes. Three things also differ
between arms, and each weakens the comparison somewhere:

- **`max_gleanings`.** Gleanings are a follow-up round telling the model
  it missed entities. `setup_index.py` sets it to 0 for both local arms
  and 1 for gpt-4o-mini, because qwen2.5:7b stalled for more than 20
  minutes generating on a single chunk. So gpt-4o-mini got a second
  extraction pass that llama3.1 and qwen2.5 did not, and the connectivity
  comparison between them is not clean. It does not touch the density
  finding, which holds at both settings.
- **`prune_graph`.** The `nlp` arm runs GraphRAG's `prune_graph` step and
  the LLM arms do not, following each path's own defaults.
- **The entity-type vocabulary.** The LLM arms are given four types; the
  regex arm has no vocabulary and emits one. See above.

Chunk size is fixed at 1,200 tokens with 100 overlap across every arm and
is never swept, so the regex path's density is partly a function of a
constant nobody varied here.

## Reproducing this

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt

# Build one GraphRAG workspace per corpus for the no-LLM path.
./.venv/bin/python setup_index.py --mode nlp

# Index all three. No API key, no cost.
for c in christmas_carol sherlock_holmes moby_dick; do
    GRAPHRAG_API_KEY=unused ./.venv/bin/python -m graphrag index \
        --root idx/${c}_nlp --skip-validation
done

# Record the shape of every index that exists, then plot.
./.venv/bin/python exp_extraction_models.py
./.venv/bin/python make_fig.py
```

`exp_extraction_models.py` rewrites `results/extraction_models.csv` from
whatever indexes are present, so a partial rebuild will shrink the shipped
CSV. Pass `--out` to write elsewhere while experimenting.

The paid arm indexes all three corpora with gpt-4o-mini and took about 16
minutes. Estimate the spend first with `cost_model.py`:

```bash
./.venv/bin/python setup_index.py --mode llm
export GRAPHRAG_API_KEY=sk-...
for c in christmas_carol sherlock_holmes moby_dick; do
    ./.venv/bin/python -m graphrag index --root idx/${c}_llm --skip-validation
done
```

The local arms use Ollama, which exposes an OpenAI-compatible API, so they
run at zero cost:

```bash
ollama pull llama3.1:8b
./.venv/bin/python setup_index.py --mode local
```

The qwen2.5 arm in the shipped CSV was produced by pointing the `local`
mode at `qwen2.5:7b-instruct` and renaming the workspace by hand; there is
no `--mode qwen`. That row is reported as measured but is not reproducible
from a flag in this repository.

## Cost

`cost_model.py` estimates GraphRAG indexing spend before you commit to it,
reading chunk size, overlap and `max_gleanings` from graphrag's own
installed defaults so the estimate tracks the version you have. Prompt and
output sizes per chunk are stated ASSUMPTIONS, marked as such in the
source, because they depend on the corpus.

```bash
./.venv/bin/python cost_model.py --corpus-tokens 250000 --model gpt-4o-mini
```

For the three corpora here (482,251 tokens, 438 chunks) it estimates $0.71
on gpt-4o-mini for extraction. That is a model, not a measurement. Check
your provider's usage dashboard for what a run actually cost; GraphRAG does
not record token usage in a form this repository can read back.

## Verification

```bash
./.venv/bin/python verify_numbers.py
```

Every number in this README that is **derived from `results/*.csv`** is
checked by that script, which names the file, row and column each must
come from, and reports how many claims it checked. Derived figures such as
the 12.0x mean-degree ratio read **both** operands from the CSV rather than
hand-typing either, since a gate that types its own operands proves
nothing. It exits non-zero on any mismatch.

Out of its scope, and therefore unchecked: the $0.71 cost estimate, the
token and chunk counts behind it, the "about 16 minutes" runtime, and
configuration values quoted from `setup_index.py` such as chunk size and
`max_gleanings`.

## Limitations

- **Three corpora, one genre, one language.** All three are 19th-century
  English literary prose. Nothing here establishes that the same spread
  appears on news, code, clinical notes or multilingual text, and the
  qwen2.5 translation failure in particular may behave differently on a
  corpus that is not entirely English.
- **The four-way comparison exists on one corpus.** Only A Christmas Carol
  was indexed by all four extractors, so the claim that the local models
  sit in the same density band as gpt-4o-mini rests on a single point each.
- **One run per cell.** LLM extraction is not deterministic and no arm is
  repeated, so run-to-run variance is unmeasured.
- **One prompt, one vocabulary, one chunk size.** No prompt, `entity_types`
  or chunking ablation is run anywhere here. The exact prompt used is
  committed under `prompts/` so the claim can be checked rather than taken
  on trust. The finding is about three
  models under one configuration, not about LLM extraction in general, and
  a prompt that explicitly asked for co-occurrence edges might well close
  the gap.
- **This is descriptive.** It reports what the extractors produce. It does
  not evaluate answer quality, retrieval, or which graph is better for any
  downstream task, and nothing here says the dense co-occurrence graph is
  preferable to the sparse assertion graph. They are different objects and
  the right one depends on the question being asked.

## Layout

| Path | Contents |
|---|---|
| `setup_index.py` | Builds one GraphRAG workspace per corpus per extraction mode |
| `gr_adapter.py` | GraphRAG parquet output to an edge list; largest component, simple, self-loop free |
| `exp_extraction_models.py` | Records the shape of every index that exists; writes the CSV |
| `cost_model.py` | Pre-spend indexing cost estimate, read from graphrag's own defaults |
| `make_fig.py` | Regenerates the figure from the CSV and nothing else |
| `verify_numbers.py` | Gate: checks every CSV-derived number in this README |
| `prompts/` | GraphRAG's own default extraction prompts, committed so the LLM path is reproducible and the prompt is pinned |
| `corpora/` | Public domain source texts, see `CORPORA_NOTICE.md` |
| `results/extraction_models.csv` | Every measured number this repository reports |

## License

Code, figures and result files: MIT, see `LICENSE`. The texts in
`corpora/` are public domain and are not covered by that license; see
`CORPORA_NOTICE.md`.
