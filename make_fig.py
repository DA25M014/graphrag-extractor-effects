"""Regenerate the figure from results/extraction_models.csv and nothing else.

One corpus (A Christmas Carol) is indexed by every extractor, so it is the
only place a like-for-like comparison exists. The other corpora appear in
the CSV under the NLP path and are reported in the README table, not here.

Writes results/f1_extractor_effects.pdf (and .png).
"""

import os
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, 'results', 'extraction_models.csv')
OUT = os.path.join(HERE, 'results', 'f1_extractor_effects')

# Colorblind-safe (Okabe-Ito). The no-LLM path is grey so the LLM
# extractors read as a group against it.
GREY = '#999999'
BLUE = '#0072B2'
ORANGE = '#E69F00'
GREEN = '#009E73'

# Display order and label, keyed by the `mode` column.
ORDER = [
    ('nlp', 'regex_english\n(no LLM)', GREY),
    ('llm', 'gpt-4o-mini\n(paid API)', BLUE),
    ('local', 'llama3.1:8b\n(local)', ORANGE),
    ('qwen', 'qwen2.5:7b\n(local)', GREEN),
]
CORPUS = 'christmas_carol'


def load():
    rows = {}
    for r in csv.DictReader(open(CSV)):
        if r['corpus'] == CORPUS:
            rows[r['mode']] = r
    return rows


def main():
    rows = load()
    present = [(m, lab, c) for m, lab, c in ORDER if m in rows]
    if len(present) < 2:
        raise SystemExit(f"need at least 2 extractors for {CORPUS}, "
                         f"found {list(rows)}")
    missing = [m for m, _, _ in ORDER if m not in rows]
    if missing:
        print(f"note: no row yet for {missing}; plotting {len(present)} arms")

    labels = [lab for _, lab, _ in present]
    colors = [c for _, _, c in present]
    x = np.arange(len(present))

    n = [int(rows[m]['n_lcc']) for m, _, _ in present]
    mm = [int(rows[m]['m_lcc']) for m, _, _ in present]
    off = [float(rows[m]['frac_dropped_off_lcc']) for m, _, _ in present]
    nonascii = [float(rows[m]['nonascii_title_frac']) for m, _, _ in present]

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))

    # Panel 1: graph size. Log scale, because m spans an order of magnitude.
    ax = axes[0]
    w = 0.38
    ax.bar(x - w / 2, n, w, color=colors, edgecolor='black', linewidth=0.5)
    ax.bar(x + w / 2, mm, w, color=colors, edgecolor='black', linewidth=0.5,
           hatch='///')
    ax.set_yscale('log')
    ax.set_ylabel('count (log scale)')
    ax.set_title('Graph size\nsolid: nodes, hatched: edges', fontsize=9)
    for xi, (a, b) in enumerate(zip(n, mm)):
        ax.text(xi - w / 2, a * 1.12, str(a), ha='center', fontsize=7)
        ax.text(xi + w / 2, b * 1.12, str(b), ha='center', fontsize=7)

    # Panel 2: mean degree, the density the graph actually has.
    ax = axes[1]
    dbar = [float(rows[m]['dbar']) for m, _, _ in present]
    ax.bar(x, dbar, 0.6, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('mean degree')
    ax.set_title('Density of the extracted graph', fontsize=9)
    for xi, v in zip(x, dbar):
        ax.text(xi, v + max(dbar) * 0.02, f'{v:.1f}', ha='center', fontsize=7)

    # Panel 3: entities that never reach the main component, plus the
    # translated-names failure marked where it occurs.
    ax = axes[2]
    ax.bar(x, [v * 100 for v in off], 0.6, color=colors,
           edgecolor='black', linewidth=0.5)
    ax.set_ylabel('% of extracted entities off the LCC')
    ax.set_title('Entities the graph never connects', fontsize=9)
    ax.set_ylim(0, max(100 * max(off) * 1.35, 10))
    for xi, (v, na) in enumerate(zip(off, nonascii)):
        ax.text(xi, v * 100 + max(off) * 100 * 0.04, f'{v:.0%}',
                ha='center', fontsize=7)
        if na > 0:
            ax.text(xi, v * 100 * 0.5,
                    f'{na:.0%} of titles\nnon-ASCII', ha='center',
                    fontsize=6.5, color='white', fontweight='bold')

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle('A Christmas Carol: same corpus, same chunker, same '
                 'clusterer, different entity extractor', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT + '.pdf')
    fig.savefig(OUT + '.png', dpi=200)
    print(f"wrote {OUT}.pdf (+ .png) from {len(present)} extractors")


if __name__ == '__main__':
    main()
