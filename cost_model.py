"""Estimate GraphRAG indexing cost before spending anything.

Every parameter below is read from graphrag's own defaults where one
exists (chunk size 1200, overlap 100, max_gleanings 1), so the estimate
tracks the version actually installed. Prompt and output sizes are
stated assumptions, marked ASSUMED, because they depend on the corpus.

Calibration check: at 1M corpus tokens with gpt-4o this model gives a
figure in the $20-40 range that Microsoft's own cost guidance reports,
which is the only external anchor available.

Usage:
    python cost_model.py --corpus-tokens 250000 --model gpt-4o-mini
"""

import argparse

from graphrag.config.defaults import graphrag_config_defaults as D

# USD per 1M tokens (input, output). Verify against current pricing
# before relying on a number; these move.
PRICES = {
    'gpt-4o-mini': (0.15, 0.60),
    'gpt-4o': (2.50, 10.00),
}

# ASSUMED, not read from config: the extract_graph prompt carries
# instructions plus few-shot examples. Measured prompts in this family
# run 1500-2500 tokens; 2000 is the midpoint.
PROMPT_OVERHEAD = 2000
# ASSUMED: tuples emitted per chunk.
OUT_PER_CHUNK = 1000
# ASSUMED: a gleaning round re-sends context on a fraction of chunks.
GLEANING_FACTOR = 1.0 + 0.5 * D.extract_graph.max_gleanings

# Community reports: ASSUMED per-report prompt and output sizes.
REPORT_IN, REPORT_OUT = 2500, 800


def estimate(corpus_tokens, model='gpt-4o-mini', n_communities=0,
             with_reports=False):
    size, overlap = D.chunking.size, D.chunking.overlap
    stride = size - overlap
    chunks = max(1, round(corpus_tokens / stride))

    ex_in = chunks * (PROMPT_OVERHEAD + size) * GLEANING_FACTOR
    ex_out = chunks * OUT_PER_CHUNK * GLEANING_FACTOR

    rp_in = rp_out = 0
    if with_reports:
        rp_in = n_communities * REPORT_IN
        rp_out = n_communities * REPORT_OUT

    pin, pout = PRICES[model]
    cost_ex = (ex_in * pin + ex_out * pout) / 1e6
    cost_rp = (rp_in * pin + rp_out * pout) / 1e6
    return {'chunks': chunks, 'extract_in': ex_in, 'extract_out': ex_out,
            'reports_in': rp_in, 'reports_out': rp_out,
            'cost_extract': cost_ex, 'cost_reports': cost_rp,
            'total': cost_ex + cost_rp}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus-tokens', type=int, default=250_000)
    ap.add_argument('--model', default='gpt-4o-mini', choices=PRICES)
    ap.add_argument('--communities', type=int, default=0)
    ap.add_argument('--with-reports', action='store_true')
    a = ap.parse_args()

    r = estimate(a.corpus_tokens, a.model, a.communities, a.with_reports)
    print(f"corpus {a.corpus_tokens:,} tokens -> {r['chunks']:,} chunks "
          f"(size {D.chunking.size}, overlap {D.chunking.overlap}, "
          f"gleanings {D.extract_graph.max_gleanings})")
    print(f"  extract_graph : {r['extract_in']:>12,.0f} in  "
          f"{r['extract_out']:>10,.0f} out   ${r['cost_extract']:.2f}")
    if a.with_reports:
        print(f"  community rpt : {r['reports_in']:>12,.0f} in  "
              f"{r['reports_out']:>10,.0f} out   ${r['cost_reports']:.2f}")
    print(f"  TOTAL on {a.model}: ${r['total']:.2f}")


if __name__ == '__main__':
    main()
