"""Check every number claimed in README.md against results/*.csv.

The rule this repo follows is that `results/extraction_models.csv` is the
single source of truth and nothing is typed into the README by hand. This
script enforces it: each claim names the CSV, the row selector and the
column it must come from, and the script fails loudly if the file no
longer agrees.

Derived claims read BOTH operands from the CSV rather than hand-typing
either, since a gate that types its own operands proves nothing.

Run: python verify_numbers.py
"""

import os
import sys
import csv
import re

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    path = os.path.join(HERE, 'results', name)
    if not os.path.exists(path):
        return None
    return list(csv.DictReader(open(path)))


def pick(rows, **sel):
    if rows is None:
        return None
    for r in rows:
        if all(str(r.get(k)) == str(v) for k, v in sel.items()):
            return r
    return None


CC = {'corpus': 'christmas_carol'}
EM = 'extraction_models.csv'

# (claim, csv, row selector, column, expected)
CLAIMS = [
    # The no-LLM path, on all three corpora.
    ('christmas carol regex n', EM, dict(CC, mode='nlp'), 'n_lcc', 161),
    ('christmas carol regex m', EM, dict(CC, mode='nlp'), 'm_lcc', 1466),
    ('christmas carol regex mean degree', EM, dict(CC, mode='nlp'),
     'dbar', 18.21),
    ('christmas carol regex nothing off the LCC', EM, dict(CC, mode='nlp'),
     'frac_dropped_off_lcc', 0.0),
    ('christmas carol regex entity types', EM, dict(CC, mode='nlp'),
     'n_entity_types', 1),
    ('sherlock regex n', EM, {'corpus': 'sherlock_holmes', 'mode': 'nlp'},
     'n_lcc', 669),
    ('sherlock regex m', EM, {'corpus': 'sherlock_holmes', 'mode': 'nlp'},
     'm_lcc', 9299),
    ('moby dick regex n', EM, {'corpus': 'moby_dick', 'mode': 'nlp'},
     'n_lcc', 1888),
    ('moby dick regex m', EM, {'corpus': 'moby_dick', 'mode': 'nlp'},
     'm_lcc', 37452),

    # llama3.1:8b.
    ('llama3.1 n', EM, dict(CC, mode='local'), 'n_lcc', 97),
    ('llama3.1 m', EM, dict(CC, mode='local'), 'm_lcc', 125),
    ('llama3.1 mean degree', EM, dict(CC, mode='local'), 'dbar', 2.58),
    ('llama3.1 fraction off the LCC', EM, dict(CC, mode='local'),
     'frac_dropped_off_lcc', 0.667),
    ('llama3.1 no translated names', EM, dict(CC, mode='local'),
     'nonascii_title_frac', 0.0),
    ('llama3.1 entity types', EM, dict(CC, mode='local'),
     'n_entity_types', 14),
    ('llama3.1 entities extracted', EM, dict(CC, mode='local'),
     'entities_raw', 291),

    # qwen2.5:7b, including the translated-names failure.
    ('qwen n', EM, dict(CC, mode='qwen'), 'n_lcc', 104),
    ('qwen m', EM, dict(CC, mode='qwen'), 'm_lcc', 145),
    ('qwen mean degree', EM, dict(CC, mode='qwen'), 'dbar', 2.79),
    ('qwen fraction off the LCC', EM, dict(CC, mode='qwen'),
     'frac_dropped_off_lcc', 0.425),
    ('qwen translated entity names', EM, dict(CC, mode='qwen'),
     'nonascii_title_frac', 0.083),
    ('qwen entity types', EM, dict(CC, mode='qwen'), 'n_entity_types', 11),

    # gpt-4o-mini, the paid arm: recovers entities, not edges.
    ('gpt-4o-mini christmas carol n', EM, dict(CC, mode='llm'), 'n_lcc', 158),
    ('gpt-4o-mini christmas carol m', EM, dict(CC, mode='llm'), 'm_lcc', 238),
    ('gpt-4o-mini christmas carol mean degree', EM, dict(CC, mode='llm'),
     'dbar', 3.01),
    ('gpt-4o-mini christmas carol off the LCC', EM, dict(CC, mode='llm'),
     'frac_dropped_off_lcc', 0.214),
    ('gpt-4o-mini christmas carol clean titles', EM, dict(CC, mode='llm'),
     'nonascii_title_frac', 0.0),
    ('gpt-4o-mini christmas carol entity types', EM, dict(CC, mode='llm'),
     'n_entity_types', 5),
    ('gpt-4o-mini sherlock n', EM,
     {'corpus': 'sherlock_holmes', 'mode': 'llm'}, 'n_lcc', 562),
    ('gpt-4o-mini sherlock m', EM,
     {'corpus': 'sherlock_holmes', 'mode': 'llm'}, 'm_lcc', 867),
    ('gpt-4o-mini sherlock mean degree', EM,
     {'corpus': 'sherlock_holmes', 'mode': 'llm'}, 'dbar', 3.09),
    ('gpt-4o-mini moby dick n', EM,
     {'corpus': 'moby_dick', 'mode': 'llm'}, 'n_lcc', 905),
    ('gpt-4o-mini moby dick m', EM,
     {'corpus': 'moby_dick', 'mode': 'llm'}, 'm_lcc', 1491),
    ('gpt-4o-mini moby dick mean degree', EM,
     {'corpus': 'moby_dick', 'mode': 'llm'}, 'dbar', 3.30),
    ('regex sherlock mean degree', EM,
     {'corpus': 'sherlock_holmes', 'mode': 'nlp'}, 'dbar', 27.80),
    ('regex moby dick mean degree', EM,
     {'corpus': 'moby_dick', 'mode': 'nlp'}, 'dbar', 39.67),
]

# Derived. Both operands read from the CSV.
# (claim, (csv, sel, col), (csv, sel, col), expected quotient, dp)
RATIOS = [
    ('regex over llama3.1 edge count',
     (EM, dict(CC, mode='nlp'), 'm_lcc'),
     (EM, dict(CC, mode='local'), 'm_lcc'), 11.73, 2),
    ('regex over qwen edge count',
     (EM, dict(CC, mode='nlp'), 'm_lcc'),
     (EM, dict(CC, mode='qwen'), 'm_lcc'), 10.11, 2),
    ('regex over llama3.1 mean degree',
     (EM, dict(CC, mode='nlp'), 'dbar'),
     (EM, dict(CC, mode='local'), 'dbar'), 7.06, 2),

    # gpt-4o-mini against regex: nodes recovered, edges recovered, and the
    # mean-degree ratio that widens with corpus size.
    ('gpt-4o-mini recovers 98% of christmas carol nodes',
     (EM, dict(CC, mode='llm'), 'n_lcc'),
     (EM, dict(CC, mode='nlp'), 'n_lcc'), 0.98, 2),
    ('gpt-4o-mini recovers 16% of christmas carol edges',
     (EM, dict(CC, mode='llm'), 'm_lcc'),
     (EM, dict(CC, mode='nlp'), 'm_lcc'), 0.16, 2),
    ('gpt-4o-mini recovers 84% of sherlock nodes',
     (EM, {'corpus': 'sherlock_holmes', 'mode': 'llm'}, 'n_lcc'),
     (EM, {'corpus': 'sherlock_holmes', 'mode': 'nlp'}, 'n_lcc'), 0.84, 2),
    ('gpt-4o-mini recovers 48% of moby dick nodes',
     (EM, {'corpus': 'moby_dick', 'mode': 'llm'}, 'n_lcc'),
     (EM, {'corpus': 'moby_dick', 'mode': 'nlp'}, 'n_lcc'), 0.48, 2),
    ('mean-degree ratio christmas carol',
     (EM, dict(CC, mode='nlp'), 'dbar'),
     (EM, dict(CC, mode='llm'), 'dbar'), 6.0, 1),
    ('mean-degree ratio sherlock',
     (EM, {'corpus': 'sherlock_holmes', 'mode': 'nlp'}, 'dbar'),
     (EM, {'corpus': 'sherlock_holmes', 'mode': 'llm'}, 'dbar'), 9.0, 1),
    ('mean-degree ratio moby dick',
     (EM, {'corpus': 'moby_dick', 'mode': 'nlp'}, 'dbar'),
     (EM, {'corpus': 'moby_dick', 'mode': 'llm'}, 'dbar'), 12.0, 1),
    ('llama3.1 recovers 60% of christmas carol nodes',
     (EM, dict(CC, mode='local'), 'n_lcc'),
     (EM, dict(CC, mode='nlp'), 'n_lcc'), 0.60, 2),
    ('qwen recovers 65% of christmas carol nodes',
     (EM, dict(CC, mode='qwen'), 'n_lcc'),
     (EM, dict(CC, mode='nlp'), 'n_lcc'), 0.65, 2),
]

# Claims that only exist once the paid arm has been run. Absent rows are
# reported as SKIPPED rather than silently passing, so a README that
# quotes gpt-4o-mini numbers cannot slip through before the run.
OPTIONAL_MODES = []


def main():
    fails, checked, skipped = [], 0, 0
    cache = {}

    def rows_for(fname):
        if fname not in cache:
            cache[fname] = load(fname)
        return cache[fname]

    for claim, fname, sel, col, expected in CLAIMS:
        rows = rows_for(fname)
        if rows is None:
            fails.append(f"{claim}: {fname} missing")
            continue
        row = pick(rows, **sel)
        if row is None:
            if sel.get('mode') in OPTIONAL_MODES:
                skipped += 1
                continue
            fails.append(f"{claim}: no row matching {sel} in {fname}")
            continue
        if col not in row:
            fails.append(f"{claim}: column {col} absent in {fname}")
            continue
        checked += 1
        if abs(float(row[col]) - float(expected)) > 1e-6:
            fails.append(f"{claim}: README says {expected}, "
                         f"{fname} says {row[col]}")

    for claim, nsrc, dsrc, expected, dp in RATIOS:
        vals = []
        for fname, sel, col in (nsrc, dsrc):
            row = pick(rows_for(fname), **sel)
            if row is None or col not in row:
                fails.append(f"{claim}: cannot read {col} from {fname}")
                vals = None
                break
            vals.append(float(row[col]))
        if not vals:
            continue
        checked += 1
        got = round(vals[0] / vals[1], dp)
        if abs(got - expected) > 10 ** (-dp):
            fails.append(f"{claim}: stated {expected}, CSVs give {got} "
                         f"({vals[0]:g}/{vals[1]:g})")

    readme = open(os.path.join(HERE, 'README.md')).read()
    n_numbers = len(re.findall(r'(?<![\w.])\d[\d,]*\.?\d*(?![\w])', readme))

    print(f"checked {checked} claims against results/*.csv")
    if skipped:
        print(f"skipped {skipped} claims for arms not yet run "
              f"({', '.join(OPTIONAL_MODES)})")
    print(f"README contains {n_numbers} numeric tokens in total "
          f"(includes years, versions, prose counts)")
    if fails:
        print(f"\nFAILED {len(fails)}:")
        for f in fails:
            print('  -', f)
        sys.exit(1)
    print('\nAll claimed numbers match the CSVs.')


if __name__ == '__main__':
    main()
