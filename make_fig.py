"""Regenerate the figure from results/extraction_models.csv and nothing else.

Three panels, left to right:
  1. Mean degree against corpus size. The regex path climbs with n; every
     LLM extractor sits flat near 3 regardless of model or corpus size.
     This is the result the repository exists to report.
  2. What a better LLM buys, on the one corpus every extractor saw:
     node recall against edge recall, relative to the regex graph.
  3. Entities that never reach the largest connected component.

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

# Okabe-Ito, colorblind safe. Grey is the no-LLM path so the LLM
# extractors read as a family against it.
GREY, BLUE, ORANGE, GREEN = '#999999', '#0072B2', '#E69F00', '#009E73'
STYLE = {
    'nlp':   ('regex_english (no LLM)', GREY),
    'llm':   ('gpt-4o-mini', BLUE),
    'local': ('llama3.1:8b', ORANGE),
    'qwen':  ('qwen2.5:7b', GREEN),
}
CORPUS_ORDER = ['christmas_carol', 'sherlock_holmes', 'moby_dick']
CORPUS_LABEL = {'christmas_carol': 'A Christmas\nCarol',
                'sherlock_holmes': 'Sherlock\nHolmes',
                'moby_dick': 'Moby Dick'}


def load():
    rows = {}
    for r in csv.DictReader(open(CSV)):
        rows[(r['corpus'], r['mode'])] = r
    return rows


def main():
    R = load()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))

    # Panel 1: mean degree vs corpus size, the headline.
    ax = axes[0]
    for mode in ('nlp', 'llm'):
        pts = [(int(R[(c, mode)]['n_lcc']), float(R[(c, mode)]['dbar']))
               for c in CORPUS_ORDER if (c, mode) in R]
        if len(pts) < 2:
            continue
        pts.sort()
        lab, col = STYLE[mode]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], 'o-', color=col,
                label=lab, linewidth=2, markersize=6)
        for x, y in pts:
            ax.annotate(f'{y:.1f}', (x, y), textcoords='offset points',
                        xytext=(0, 7), ha='center', fontsize=7)
    for mode in ('local', 'qwen'):
        if ('christmas_carol', mode) not in R:
            continue
        r = R[('christmas_carol', mode)]
        lab, col = STYLE[mode]
        ax.plot(int(r['n_lcc']), float(r['dbar']), 's', color=col,
                label=lab, markersize=7)
    ax.set_xscale('log')
    ax.set_xlabel('nodes in the graph (log scale)')
    ax.set_ylabel('mean degree')
    ax.set_title('Density is set by the extraction paradigm,\nnot by the '
                 'model', fontsize=9)
    ax.legend(fontsize=6.5, loc='upper left', framealpha=0.9)
    ax.set_ylim(0, 45)

    # Panel 2: node recall against edge recall, relative to the regex graph.
    ax = axes[1]
    x = np.arange(len(CORPUS_ORDER))
    w = 0.38
    nrec = [100 * int(R[(c, 'llm')]['n_lcc']) / int(R[(c, 'nlp')]['n_lcc'])
            for c in CORPUS_ORDER]
    erec = [100 * int(R[(c, 'llm')]['m_lcc']) / int(R[(c, 'nlp')]['m_lcc'])
            for c in CORPUS_ORDER]
    ax.bar(x - w / 2, nrec, w, color=BLUE, edgecolor='black', linewidth=0.5,
           label='nodes')
    ax.bar(x + w / 2, erec, w, color=BLUE, edgecolor='black', linewidth=0.5,
           hatch='///', label='edges')
    for xi, (a, b) in enumerate(zip(nrec, erec)):
        ax.text(xi - w / 2, a + 2, f'{a:.0f}%', ha='center', fontsize=7)
        ax.text(xi + w / 2, b + 2, f'{b:.0f}%', ha='center', fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([CORPUS_LABEL[c] for c in CORPUS_ORDER], fontsize=7.5)
    ax.set_ylabel('% of the regex graph')
    ax.set_ylim(0, 112)
    ax.set_title('gpt-4o-mini recovers the entities,\nnot the edges',
                 fontsize=9)
    ax.legend(fontsize=7, loc='upper right')

    # Panel 3: entities that never reach the main component, on the corpus
    # every extractor saw.
    ax = axes[2]
    modes = [m for m in ('nlp', 'llm', 'local', 'qwen')
             if ('christmas_carol', m) in R]
    off = [100 * float(R[('christmas_carol', m)]['frac_dropped_off_lcc'])
           for m in modes]
    na = [float(R[('christmas_carol', m)]['nonascii_title_frac'])
          for m in modes]
    cols = [STYLE[m][1] for m in modes]
    xs = np.arange(len(modes))
    ax.bar(xs, off, 0.6, color=cols, edgecolor='black', linewidth=0.5)
    for xi, (v, a) in enumerate(zip(off, na)):
        ax.text(xi, v + 2, f'{v:.0f}%', ha='center', fontsize=7)
        if a > 0:
            ax.annotate(f'{a:.0%} of titles\nnon-ASCII', (xi, v * 0.55),
                        ha='center', va='center', fontsize=6.5,
                        color='white', fontweight='bold',
                        annotation_clip=False)
    ax.set_xticks(xs)
    ax.set_xticklabels([STYLE[m][0].split(' (')[0] for m in modes],
                       fontsize=7, rotation=12, ha='right')
    ax.set_ylabel('% of extracted entities off the LCC')
    ax.set_ylim(0, 80)
    ax.set_title('Entities the graph never connects\n(A Christmas Carol)',
                 fontsize=9)

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT + '.pdf')
    fig.savefig(OUT + '.png', dpi=200)
    print(f"wrote {OUT}.pdf (+ .png)")


if __name__ == '__main__':
    main()
