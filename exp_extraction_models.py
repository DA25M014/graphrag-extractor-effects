"""Compare entity graphs produced by different extraction backends.

Same corpus, same GraphRAG pipeline, same chunker, same clusterer. The
only thing that varies is what extracts the entities. If the resulting
graphs differ structurally, then any claim about "the entity graph" is
conditional on the extractor, and a result reported on one extractor
does not transfer to another.

This is cheap: it only reads the indexes that already exist and records
their shape. It makes no LLM call of its own.

Writes `results/extraction_models.csv`.
"""

import os
import sys
import csv
import glob
import argparse
import unicodedata

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gr_adapter as ga

HERE = os.path.dirname(os.path.abspath(__file__))

# Which model produced each index. `nlp` is GraphRAG's LLM-free
# extract_graph_nlp path; the rest are LLM extraction.
BACKEND = {
    'nlp': ('regex_english (no LLM)', 'none'),
    'qwen': ('qwen2.5:7b-instruct via Ollama', 'local'),
    'local': ('llama3.1:8b via Ollama', 'local'),
    'llm': ('gpt-4o-mini', 'paid api'),
}


def non_ascii_fraction(titles):
    """Fraction of entity titles containing non-ASCII letters.

    Flags an extractor that renders entity names in another script.
    A model that translates entity names silently breaks entity
    resolution, because the translated and untranslated forms of one
    entity become two nodes.
    """
    if len(titles) == 0:
        return float('nan')
    bad = sum(1 for t in titles
              if any(unicodedata.category(ch).startswith('L')
                     and ord(ch) > 127 for ch in str(t)))
    return bad / len(titles)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default=os.path.join(HERE, 'results',
                                                  'extraction_models.csv'),
                    help='where to write; defaults to the shipped CSV')
    ap.add_argument('--force', action='store_true',
                    help='allow overwriting with FEWER rows than exist now')
    args = ap.parse_args()
    rows = []
    for d in sorted(glob.glob(os.path.join(HERE, 'idx', '*', 'output'))):
        ws = os.path.basename(os.path.dirname(d))
        corpus, _, mode = ws.rpartition('_')
        if mode not in BACKEND or not corpus:
            continue
        ents_p = os.path.join(d, 'entities.parquet')
        if not os.path.exists(ents_p):
            continue

        ents = pd.read_parquet(ents_p)
        try:
            E, n, meta = ga.load_entity_graph(d, verbose=False)
        except Exception as exc:
            print(f"[skip] {ws}: {exc}")
            continue

        model, kind = BACKEND[mode]
        types = ents['type'].astype(str)
        rows.append({
            'corpus': corpus, 'mode': mode, 'model': model,
            'cost_kind': kind,
            'entities_raw': len(ents),
            'relationships_raw': int(
                len(pd.read_parquet(os.path.join(d,
                                                 'relationships.parquet')))),
            'n_lcc': n, 'm_lcc': int(E.shape[0]),
            'dbar': round(meta['dbar'], 2),
            'dmax': meta['dmax'],
            'frac_dropped_off_lcc': round(meta['n_dropped_off_lcc']
                                          / max(1, len(ents)), 3),
            'n_entity_types': int(types.nunique()),
            'nonascii_title_frac': round(
                non_ascii_fraction(ents['title'].tolist()), 3),
        })
        print(f"[{ws}] n={n} m={int(E.shape[0])} dbar={meta['dbar']:.2f} "
              f"off-LCC={rows[-1]['frac_dropped_off_lcc']:.0%} "
              f"non-ASCII titles={rows[-1]['nonascii_title_frac']:.0%} "
              f"types={rows[-1]['n_entity_types']}")

    if not rows:
        print('no indexes found under idx/*/output; nothing written')
        return
    out = args.out
    # A partial rebuild would otherwise silently shrink the shipped CSV and
    # break every downstream claim, which is the first thing a fresh clone
    # would do.
    if os.path.exists(out) and not args.force:
        existing = len(list(csv.DictReader(open(out))))
        if len(rows) < existing:
            print(f"refusing to overwrite {out}: it has {existing} rows and "
                  f"this run found only {len(rows)}. Index the missing arms, "
                  f"or pass --out to write elsewhere, or --force.")
            return
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {out}")


if __name__ == '__main__':
    main()
