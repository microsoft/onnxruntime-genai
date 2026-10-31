# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Recompute the ratios and percentages quoted in RESULTS.md from its own recorded rows.

RESULTS.md records measured numbers and quotes speedups, improvements and percentages next to
them. This script derives each quoted value again from the recorded rows, at the precision the
rows are documented (round half up), and fails when the text disagrees. The inputs are the
numbers reported in RESULTS.md; nothing is measured or re-run here.

    python check_results_ratios.py           # exit code 1 if a quoted value disagrees with its rows
    python check_results_ratios.py --print   # also list every derived value

The checks cover the static-graph and Windows RTX 5090 sections, which start at FIRST_HEADING.
"""

import argparse
import re
import sys
from collections import namedtuple
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

RESULTS = Path(__file__).with_name("RESULTS.md")
FIRST_HEADING = "## Static-graph backend next to the paged Engine"
LAST_HEADING = "## Detailed records and reproduction"
NUMBER = r"\d[\d,]*(?:\.\d+)?"
DASH = "\N{EN DASH}"

Item = namedtuple("Item", "ok where derived found")
Table = namedtuple("Table", "heading header line rows")


def num(text):
    return Decimal(text.replace(",", ""))


def plain(text):
    return re.sub(r"[`*$]", "", text)


def rounded(value, places):
    return value.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)


def ratio(new, old):
    return rounded(new / old, 2)


def gain(value):
    """Percent gain of a new/old ratio."""
    return rounded((value - 1) * 100, 1)


def span(low, high, unit=""):
    return f"{low}{unit}" if low == high else f"{low}{unit}{DASH}{high}{unit}"


def ttft_factors(baseline, session1, final):
    """Return (baseline / final, Session 1 / final) for a (lowest, highest) final TTFT in seconds."""
    low, high = final
    return (
        span(ratio(baseline, high), ratio(baseline, low), "x"),
        span(ratio(session1, high), ratio(session1, low), "x"),
    )


def find(pattern, text, where):
    match = re.search(pattern, plain(text))
    if match is None:
        raise ValueError(f"{where}: cannot read /{pattern}/ from '{plain(text)}'")
    return match


def seconds(cell, where):
    match = find(rf"^({NUMBER})(?:[{DASH}-]({NUMBER}))? s\b", cell, where)
    return num(match[1]), num(match[2] or match[1])


def memory(cell, where):
    match = find(rf"^({NUMBER}) MiB \(({NUMBER}) GiB\)", cell, where)
    return num(match[1]), num(match[2])


def read_tables(lines):
    tables, heading, table = [], "", None
    for number, line in enumerate(lines, 1):
        if line.startswith("#"):
            heading, table = line.lstrip("#").strip(), None
        elif line.startswith("|"):
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if table is None:
                table = Table(heading, cells, number, [])
                tables.append(table)
            elif not set(line) <= set("|-: "):
                table.rows.append((number, cells))
        else:
            table = None
    return tables


def find_table(tables, key):
    for table in tables:
        if key in table.heading or key in plain(" ".join(table.header)):
            return table
    raise ValueError(f"no table matching '{key}'")


def find_row(table, label):
    for line, cells in table.rows:
        if label in plain(cells[0]):
            return line, cells
    raise ValueError(f"no row containing '{label}' under '{table.heading}'")


class Report:
    def __init__(self):
        self.items = []

    def add(self, where, ok, derived="", found=""):
        self.items.append(Item(ok, where, derived, found))

    def equal(self, where, found, derived):
        self.add(where, plain(found).strip() == plain(derived), derived, plain(found))

    def contains(self, where, text, derived):
        self.add(where, plain(derived) in " ".join(plain(text).split()), derived, None)


def check_ttft(report, tables):
    table = find_table(tables, "TTFT by prompt length")
    lowest, highest = [], []
    for line, cells in table.rows:
        where = f"line {line}: TTFT at {plain(cells[0])} prompt tokens"
        baseline, session1 = seconds(cells[1], where)[0], seconds(cells[2], where)[0]
        final = seconds(cells[3], where)
        improvement, versus_session1 = ttft_factors(baseline, session1, final)
        report.equal(f"{where}, TTFT improvement", cells[4], improvement)
        report.equal(f"{where}, Session 1 / final", cells[5], versus_session1)
        lowest.append(ratio(baseline, final[1]))
        highest.append(ratio(baseline, final[0]))
    return span(min(lowest), max(highest), "x")


def check_summary(report, tables, ttft_range):
    table = find_table(tables, "Verified Windows RTX 5090")
    ratios = []
    for context in ("ctx=512", "ctx=2048"):
        line, cells = find_row(table, f"{context} median decode")
        where = f"line {line}: decode gain at {context}"
        baseline = num(find(rf"^({NUMBER}) tok/s", cells[1], where)[1])
        default = num(find(rf"({NUMBER}) tok/s \(default\)", cells[3], where)[1])
        promo = find(rf"({NUMBER})[{DASH}-]({NUMBER}) tok/s \(promo", cells[3], where)
        low, high = num(promo[1]), num(promo[2])
        report.equal(
            where,
            cells[4],
            f"{gain(default / baseline):+}% ({ratio(default, baseline)}x) default / "
            f"{gain(low / baseline):+}%{DASH}{gain(high / baseline):+}% "
            f"({ratio(low, baseline)}x{DASH}{ratio(high, baseline)}x) promo",
        )
        ratios += [default / baseline, low / baseline, high / baseline]

    line, cells = find_row(table, "Long-CoT decode")
    where = f"line {line}: long-CoT decode gain"
    baseline = num(find(rf"\(({NUMBER}) avg", cells[1], where)[1])
    final = num(find(rf"\(({NUMBER}) tok/s avg", cells[3], where)[1])
    report.equal(where, cells[4], f"{gain(final / baseline):+}% ({ratio(final, baseline)}x)")

    line, cells = find_row(table, "wall time")
    where = f"line {line}: shard wall time"
    baseline, session1 = seconds(cells[1], where)[0], seconds(cells[2], where)[0]
    final = seconds(cells[3], where)[0]
    speedup = ratio(baseline, final)
    report.contains(where, cells[3], f"{speedup}x faster than baseline, {gain(final / session1):+}% vs Session 1")
    report.contains(where, cells[4], f"{speedup}x end-to-end speedup")

    line, cells = find_row(table, "bench_ttft_engine.py")
    if ttft_range:
        report.contains(f"line {line}: TTFT improvement", cells[4], f"TTFT improvement {ttft_range}")
    factors = span(rounded(min(ratios), 2), rounded(max(ratios), 2), "x")
    return factors, f"{gain(min(ratios)):+}%{DASH}{gain(max(ratios)):+}%"


def check_intro(report, section, decode_range, decode_percent, ttft_range):
    paragraphs = [" ".join(plain(p).split()) for p in re.split(r"\n\s*\n", section)]
    intro = next((p for p in paragraphs if "decode throughput over the initial port" in p), "")
    for what, derived in (("decode factors", decode_range), ("decode gains", decode_percent), ("TTFT", ttft_range)):
        report.contains(f"intro of the RTX 5090 results: {what}", intro, derived)


def check_telemetry(report, tables, section):
    table = find_table(tables, "Cycle-Level Telemetry")
    line, cells = find_row(table, "Layer-to-Layer GPU Gap")
    where = f"line {line}: gap_in reduction"
    before = num(find(rf"^({NUMBER}) ms", cells[1], where)[1])
    after = num(find(rf"({NUMBER}) ms", cells[2], where)[1])
    reduction = gain(after / before)
    report.contains(where, cells[2], f"{reduction}%")
    report.contains(where, section, f"by {-reduction}%")

    line, cells = find_row(table, "t1 + t2 + t3")
    where = f"line {line}: t1 + t2 + t3"
    totals = []
    for column in (1, 2):
        match = find(rf"^({NUMBER}) ms \(({NUMBER}) \+ ({NUMBER}) \+ ({NUMBER}) ms\)", cells[column], where)
        total, *parts = (num(group) for group in match.groups())
        report.add(f"{where}, sum in column {column}", sum(parts) == total, str(sum(parts)), str(total))
        totals.append(total)
    after = num(find(rf"^({NUMBER}) ms", cells[3], where)[1])
    reduction = gain(after / totals[0])
    report.contains(where, cells[3], f"{reduction}%")
    report.contains(where, section, f"by {-reduction}%")


def check_packages(report, tables, section):
    table = find_table(tables, "Quantized and memory-reduced")
    _, baseline = find_row(table, "v12b baseline")
    base_mib, base_gib = memory(baseline[4], "v12b baseline peak")
    for line, cells in table.rows:
        if cells is baseline:
            continue
        where = f"line {line}: peak memory saved vs v12b"
        mib, gib = memory(cells[4], where)
        saved = num(find(rf"^(-{NUMBER}) GiB", cells[5], where)[1])
        derived = {rounded((mib - base_mib) / 1024, 2), gib - base_gib}
        report.add(where, saved in derived, " or ".join(f"{d} GiB" for d in sorted(derived)), plain(cells[5]))
        exact = re.search(rf"\((-{NUMBER}) MiB\)", plain(cells[5]))
        if exact:
            report.add(f"{where} (MiB)", num(exact[1]) == mib - base_mib, f"{mib - base_mib} MiB", exact[1])

    base_tps = num(plain(baseline[10]))
    for label, pattern in (("v12c Optimized", rf"^({NUMBER})$"), ("v12c + FP8 LM", rf"\(({NUMBER}) fused\)")):
        line, cells = find_row(table, label)
        where = f"line {line}: MMLU-Pro decode gain of {label}"
        tps = num(find(pattern, cells[10], where)[1])
        report.contains(where, section, f"{gain(tps / base_tps):+}% vs v12b")


def check_natural_text(report, tables, section):
    table = find_table(tables, "Decode tokens/s")
    decode, acceptance = [], []
    for _, cells in table.rows:
        decode.append(num(plain(cells[1])))
        acceptance += [num(value) for value in re.findall(r"\d\.\d+", plain(cells[3]))]
    where = f"line {table.line}: natural-text caveat"
    report.contains(where, section, f"decodes at {min(decode)}-{max(decode)} tokens/s")
    report.contains(where, section, f"accepts {min(acceptance)}-{max(acceptance)}")


FRACTION = re.compile(rf"(?<![\d.,])(\d[\d,]*)\s*/\s*(\d[\d,]*)[^\d%()=]{{0,30}}?[(=]\s*({NUMBER})%")
MEMORY = re.compile(rf"(?<![\d.,])(\d[\d,]*) MiB\s*(?:\(|/ )\s*({NUMBER}) GiB")
SUM = re.compile(r"(?<![\d.,])(\d[\d,]*) \+ (\d[\d,]*) = (\d[\d,]*)(?![\d,])")


def check_arithmetic(report, lines, first_line):
    """Fractions with a percentage, MiB with GiB, and integer sums, wherever the section quotes them."""
    for number, line in enumerate(lines, first_line):
        text = plain(line)
        for m in FRACTION.finditer(text):
            derived = rounded(num(m[1]) * 100 / num(m[2]), len(m[3].partition(".")[2]))
            report.add(f"line {number}: {m[1]}/{m[2]}", derived == num(m[3]), f"{derived}%", f"{m[3]}%")
        for m in MEMORY.finditer(text):
            derived = rounded(num(m[1]) / 1024, len(m[2].partition(".")[2]))
            report.add(f"line {number}: {m[1]} MiB", derived == num(m[2]), f"{derived} GiB", f"{m[2]} GiB")
        for m in SUM.finditer(text):
            derived = num(m[1]) + num(m[2])
            report.add(f"line {number}: {m[1]} + {m[2]}", derived == num(m[3]), str(derived), m[3])


def heading_index(lines, heading):
    for index, line in enumerate(lines):
        if line.startswith(heading):
            return index
    raise ValueError(f"heading '{heading}' not found")


def check_results(text):
    lines = text.splitlines()
    start, end = heading_index(lines, FIRST_HEADING), heading_index(lines, LAST_HEADING)
    section = "\n".join(lines[start:end])
    tables = [table for table in read_tables(lines) if start < table.line - 1 < end]
    report = Report()

    def run(check, *args):
        try:
            return check(report, *args)
        except ValueError as error:
            report.add(str(error), False)
            return None

    ttft_range = run(check_ttft, tables)
    summary = run(check_summary, tables, ttft_range)
    if summary and ttft_range:
        run(check_intro, section, *summary, ttft_range)
    run(check_telemetry, tables, section)
    run(check_packages, tables, section)
    run(check_natural_text, tables, section)
    run(check_arithmetic, lines[start:end], start + 1)
    return report


def describe(item):
    if not item.derived:
        return ""
    if item.ok:
        return f": derived '{item.derived}'"
    if item.found is None:
        return f": the text lacks '{item.derived}'"
    return f": derived '{item.derived}' but found '{item.found}'"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check that the ratios quoted in RESULTS.md follow from its rows.")
    parser.add_argument(
        "results", nargs="?", type=Path, default=RESULTS, help="defaults to the RESULTS.md next to this script"
    )
    parser.add_argument(
        "--print", dest="show", action="store_true", help="list every derived value, not only mismatches"
    )
    args = parser.parse_args(argv)
    items = check_results(args.results.read_text(encoding="utf-8")).items
    bad = [item for item in items if not item.ok]
    for item in items if args.show else bad:
        print(f"{'ok' if item.ok else 'MISMATCH'}: {item.where}{describe(item)}")
    print(f"{len(items) - len(bad)} of {len(items)} quoted values follow from the recorded rows in {args.results.name}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
