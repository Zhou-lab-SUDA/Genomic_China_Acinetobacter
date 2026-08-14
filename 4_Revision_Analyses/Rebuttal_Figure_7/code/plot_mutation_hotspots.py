"""Plot 10-kb lineage-associated SNP density and nonsynonymous subsets.

The two input tables contain all annotated lineage-associated SNPs and the
nonsynonymous subset.  Counts are aggregated into non-overlapping 10-kb bins;
the resulting figure-ready table is written alongside the plot.
"""

from pathlib import Path
import csv


FIGURE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = FIGURE_DIR / "data"
ALL_TSV = DATA_DIR / "mutation_density_all.tsv"
NONSYN_TSV = DATA_DIR / "mutation_density_nonsynonymous.tsv"
OUT = FIGURE_DIR / "RF7_mutation_hotspots_annotated_reproduced.pdf"
OUT_COMPAT = FIGURE_DIR / "RF7_mutation_hotspots_compatibility_reproduced.pdf"
OUT_CLEAN = FIGURE_DIR / "RF7_mutation_hotspots_reproduced.pdf"
OUT_TSV = DATA_DIR / "mutation_hotspots_10kb_figure_data_recalculated.tsv"
BLOCK_KB = 10

ORANGE_DARK = (0.91, 0.39, 0.16)
ORANGE_LIGHT = (0.98, 0.72, 0.50)
BLUE_DARK = (0.13, 0.14, 0.29)
BLUE_LIGHT = (0.58, 0.59, 0.72)
GREY = (0.86, 0.86, 0.86)
BLACK = (0.05, 0.05, 0.05)


def pdf_escape(text):
    return str(text).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def write_pdf(path, commands, width, height):
    stream = "\n".join(commands).encode("latin-1", errors="replace")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] "
            f"/Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> /Contents 6 0 R >>"
        ).encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    out = [b"%PDF-1.4\n"]
    offsets = [0]
    for i, obj in enumerate(objects, start=1):
        offsets.append(sum(len(x) for x in out))
        out.append(f"{i} 0 obj\n".encode())
        out.append(obj)
        out.append(b"\nendobj\n")
    xref = sum(len(x) for x in out)
    out.append(f"xref\n0 {len(objects) + 1}\n".encode())
    out.append(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        out.append(f"{off:010d} 00000 n \n".encode())
    out.append(f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    path.write_bytes(b"".join(out))


def ypdf(height, y):
    return height - y


def text(cmds, x, y, s, size=10, color=BLACK, bold=False, rotate=0):
    r, g, b = color
    font = "/F2" if bold else "/F1"
    s = pdf_escape(s)
    if rotate == 90:
        cmds.append(f"{r:.3f} {g:.3f} {b:.3f} rg BT {font} {size} Tf 0 1 -1 0 {x:.2f} {y:.2f} Tm ({s}) Tj ET")
    else:
        cmds.append(f"{r:.3f} {g:.3f} {b:.3f} rg BT {font} {size} Tf {x:.2f} {y:.2f} Td ({s}) Tj ET")


def stroke(cmds, color, width=1):
    r, g, b = color
    cmds.append(f"{r:.3f} {g:.3f} {b:.3f} RG {width:.2f} w")


def fill(cmds, color):
    r, g, b = color
    cmds.append(f"{r:.3f} {g:.3f} {b:.3f} rg")


def read_counts(path):
    data = {}
    max_kb = 0
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            group = row["group"]
            kb = int(row["kb_bin"])
            block = kb // BLOCK_KB
            key = (group, block)
            if key not in data:
                data[key] = {"density": 0.0, "count": 0}
            data[key]["density"] += float(row["density_percent"])
            data[key]["count"] += int(row["count"])
            max_kb = max(max_kb, kb)
    return data, max_kb // BLOCK_KB


def main():
    all_data, max1 = read_counts(ALL_TSV)
    nonsyn_data, max2 = read_counts(NONSYN_TSV)
    max_block = max(max1, max2)

    width, height = 1200, 600
    ml, mr = 88, 48
    plot_w = width - ml - mr
    mid = 295
    top = 82
    bottom = 500
    ymax_density = 8.0
    ymax_pct = 100.0
    cmds = ["1 1 1 rg 0 0 1200 600 re f"]

    def xs(block):
        return ml + block / max_block * plot_w

    def y_density(value, group):
        if group == "ESL2.5":
            return mid - value / ymax_density * (mid - top)
        return mid + value / ymax_density * (bottom - mid)

    text(cmds, 34, ypdf(height, 36), "All-SNP hotspots and nonsynonymous composition", 23, BLACK, True)
    text(cmds, 34, ypdf(height, 61), "Wide dark bars show all-SNP mutation density per 10 kb; narrow light bars show the nonsynonymous/nonsense fraction of the all-SNP density", 12, BLACK)

    stroke(cmds, BLACK, 1.4)
    cmds.append(f"{ml:.2f} {ypdf(height, mid):.2f} m {width - mr:.2f} {ypdf(height, mid):.2f} l S")
    cmds.append(f"{ml:.2f} {ypdf(height, top):.2f} m {ml:.2f} {ypdf(height, bottom):.2f} l S")

    for t in range(1, int(ymax_density) + 1):
        for group in ["ESL2.5", "non-ESL2.5"]:
            y = y_density(t, group)
            stroke(cmds, GREY, 0.55)
            cmds.append(f"{ml:.2f} {ypdf(height, y):.2f} m {width - mr:.2f} {ypdf(height, y):.2f} l S")
        text(cmds, ml - 25, ypdf(height, y_density(t, "ESL2.5") + 4), str(t), 12)
        text(cmds, ml - 25, ypdf(height, y_density(t, "non-ESL2.5") + 4), str(t), 12)

    for mb in range(0, 5):
        block = (mb * 1000) // BLOCK_KB
        if block > max_block:
            continue
        x = xs(block)
        stroke(cmds, GREY, 0.5)
        cmds.append(f"{x:.2f} {ypdf(height, top):.2f} m {x:.2f} {ypdf(height, bottom):.2f} l S")
        text(cmds, x - 17, ypdf(height, mid + 24), f"{mb}MB", 15)

    bar_w = max(1.2, plot_w / (max_block + 1) * 0.9)
    thin_w = max(0.45, bar_w * 0.32)

    for group in ["ESL2.5", "non-ESL2.5"]:
        dark = ORANGE_DARK if group == "ESL2.5" else BLUE_DARK
        light = ORANGE_LIGHT if group == "ESL2.5" else BLUE_LIGHT
        fill(cmds, dark)
        for block in range(max_block + 1):
            density = all_data.get((group, block), {}).get("density", 0.0)
            if density <= 0:
                continue
            x = xs(block) - bar_w / 2
            y = y_density(density, group)
            if group == "ESL2.5":
                cmds.append(f"{x:.2f} {ypdf(height, mid):.2f} {bar_w:.2f} {ypdf(height, y) - ypdf(height, mid):.2f} re f")
            else:
                cmds.append(f"{x:.2f} {ypdf(height, y):.2f} {bar_w:.2f} {ypdf(height, mid) - ypdf(height, y):.2f} re f")

        fill(cmds, light)
        for block in range(max_block + 1):
            all_count = all_data.get((group, block), {}).get("count", 0)
            ns_count = nonsyn_data.get((group, block), {}).get("count", 0)
            if all_count <= 0 or ns_count <= 0:
                continue
            density = all_data.get((group, block), {}).get("density", 0.0)
            ns_density = density * ns_count / all_count
            x = xs(block) - thin_w / 2
            y = y_density(ns_density, group)
            if group == "ESL2.5":
                cmds.append(f"{x:.2f} {ypdf(height, mid):.2f} {thin_w:.2f} {ypdf(height, y) - ypdf(height, mid):.2f} re f")
            else:
                cmds.append(f"{x:.2f} {ypdf(height, y):.2f} {thin_w:.2f} {ypdf(height, mid) - ypdf(height, y):.2f} re f")

    text(cmds, 34, ypdf(height, 225), "ESL2.5", 17, ORANGE_DARK, True)
    text(cmds, 34, ypdf(height, 375), "Non-2.5", 17, BLUE_DARK, True)
    text(cmds, 34, ypdf(height, 320), "All-SNP mutation density per 10 kb (%)", 18, BLACK, rotate=90)
    text(cmds, 505, ypdf(height, 345), "Chromosome", 16, BLACK)

    # Legend
    lx, ly = 760, 36
    fill(cmds, ORANGE_DARK)
    cmds.append(f"{lx:.2f} {ypdf(height, ly):.2f} 18 8 re f")
    text(cmds, lx + 24, ypdf(height, ly + 8), "All SNPs", 12, BLACK)
    fill(cmds, ORANGE_LIGHT)
    cmds.append(f"{lx + 105:.2f} {ypdf(height, ly):.2f} 8 8 re f")
    text(cmds, lx + 119, ypdf(height, ly + 8), "Non-syn fraction within all SNPs", 12, BLACK)
    fill(cmds, ORANGE_DARK)
    cmds.append(f"{lx + 295:.2f} {ypdf(height, ly):.2f} 9 9 re f")
    text(cmds, lx + 310, ypdf(height, ly + 8), "ESL2.5", 12, ORANGE_DARK)
    fill(cmds, BLUE_DARK)
    cmds.append(f"{lx + 370:.2f} {ypdf(height, ly):.2f} 9 9 re f")
    text(cmds, lx + 385, ypdf(height, ly + 8), "Non-2.5", 12, BLUE_DARK)

    write_pdf(OUT, cmds, width, height)
    write_pdf(OUT_COMPAT, cmds, width, height)
    clean_cmds = build_clean_pdf(all_data, nonsyn_data, max_block, width, height)
    write_pdf(OUT_CLEAN, clean_cmds, width, height)
    with OUT_TSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow([
            "group",
            "block_start",
            "block_end",
            "all_snp_count",
            "nonsyn_snp_count",
            "nonsyn_fraction",
            "all_snp_density_percent_per_10kb",
            "nonsyn_fraction_density_percent_per_10kb",
        ])
        for group in ["ESL2.5", "non-ESL2.5"]:
            for block in range(max_block + 1):
                all_count = all_data.get((group, block), {}).get("count", 0)
                ns_count = nonsyn_data.get((group, block), {}).get("count", 0)
                density = all_data.get((group, block), {}).get("density", 0.0)
                fraction = ns_count / all_count if all_count else 0.0
                writer.writerow([
                    group,
                    block * BLOCK_KB * 1000,
                    (block + 1) * BLOCK_KB * 1000 - 1,
                    all_count,
                    ns_count,
                    f"{fraction:.6g}",
                    f"{density:.6g}",
                    f"{density * fraction:.6g}",
                ])
    print(f"Done. Wrote {OUT}")
    print(f"Done. Wrote {OUT_COMPAT}")
    print(f"Done. Wrote {OUT_CLEAN}")


def build_clean_pdf(all_data, nonsyn_data, max_block, width=1200, height=600):
    ml, mr = 88, 48
    plot_w = width - ml - mr
    mid = 295
    top = 82
    bottom = 500
    ymax_density = 10.0
    cmds = ["1 1 1 rg 0 0 1200 600 re f"]

    def xs(block):
        return ml + block / max_block * plot_w

    def y_density(value, group):
        if group == "ESL2.5":
            return mid - value / ymax_density * (mid - top)
        return mid + value / ymax_density * (bottom - mid)

    stroke(cmds, BLACK, 1.4)
    cmds.append(f"{ml:.2f} {ypdf(height, mid):.2f} m {width - mr:.2f} {ypdf(height, mid):.2f} l S")
    cmds.append(f"{ml:.2f} {ypdf(height, top):.2f} m {ml:.2f} {ypdf(height, bottom):.2f} l S")

    for t in range(1, int(ymax_density) + 1):
        text(cmds, ml - 18, ypdf(height, y_density(t, "ESL2.5") + 4), str(t), 12)
        text(cmds, ml - 18, ypdf(height, y_density(t, "non-ESL2.5") + 4), str(t), 12)

    for mb in range(0, 5):
        block = (mb * 1000) // BLOCK_KB
        if block > max_block:
            continue
        x = xs(block)
        text(cmds, x - 17, ypdf(height, mid + 24), f"{mb}MB", 15)

    bar_w = max(1.2, plot_w / (max_block + 1) * 0.9)
    thin_w = max(0.45, bar_w * 0.32)

    for group in ["ESL2.5", "non-ESL2.5"]:
        dark = ORANGE_DARK if group == "ESL2.5" else BLUE_DARK
        light = ORANGE_LIGHT if group == "ESL2.5" else BLUE_LIGHT
        fill(cmds, dark)
        for block in range(max_block + 1):
            density = all_data.get((group, block), {}).get("density", 0.0)
            if density <= 0:
                continue
            x = xs(block) - bar_w / 2
            y = y_density(density, group)
            if group == "ESL2.5":
                cmds.append(f"{x:.2f} {ypdf(height, mid):.2f} {bar_w:.2f} {ypdf(height, y) - ypdf(height, mid):.2f} re f")
            else:
                cmds.append(f"{x:.2f} {ypdf(height, y):.2f} {bar_w:.2f} {ypdf(height, mid) - ypdf(height, y):.2f} re f")

        fill(cmds, light)
        for block in range(max_block + 1):
            all_count = all_data.get((group, block), {}).get("count", 0)
            ns_count = nonsyn_data.get((group, block), {}).get("count", 0)
            if all_count <= 0 or ns_count <= 0:
                continue
            density = all_data.get((group, block), {}).get("density", 0.0)
            ns_density = density * ns_count / all_count
            x = xs(block) - thin_w / 2
            y = y_density(ns_density, group)
            if group == "ESL2.5":
                cmds.append(f"{x:.2f} {ypdf(height, mid):.2f} {thin_w:.2f} {ypdf(height, y) - ypdf(height, mid):.2f} re f")
            else:
                cmds.append(f"{x:.2f} {ypdf(height, y):.2f} {thin_w:.2f} {ypdf(height, mid) - ypdf(height, y):.2f} re f")

    text(cmds, 34, ypdf(height, 320), "Mutation density per 10 kb (%)", 18, BLACK, rotate=90)
    text(cmds, 505, ypdf(height, 345), "Chromosome", 16, BLACK)

    # Inset legend.
    lx, ly, lw, lh = 780, 86, 340, 96
    fill(cmds, (1, 1, 1))
    cmds.append(f"{lx:.2f} {ypdf(height, ly + lh):.2f} {lw:.2f} {lh:.2f} re f")
    stroke(cmds, (0.75, 0.75, 0.75), 0.8)
    cmds.append(f"{lx:.2f} {ypdf(height, ly + lh):.2f} {lw:.2f} {lh:.2f} re S")
    fill(cmds, ORANGE_DARK)
    cmds.append(f"{lx + 16:.2f} {ypdf(height, ly + 20):.2f} 18 8 re f")
    text(cmds, lx + 42, ypdf(height, ly + 27), "ESL2.5 all SNPs", 11, BLACK)
    fill(cmds, ORANGE_LIGHT)
    cmds.append(f"{lx + 176:.2f} {ypdf(height, ly + 20):.2f} 8 8 re f")
    text(cmds, lx + 192, ypdf(height, ly + 27), "protein-changing", 11, BLACK)
    fill(cmds, BLUE_DARK)
    cmds.append(f"{lx + 16:.2f} {ypdf(height, ly + 50):.2f} 18 8 re f")
    text(cmds, lx + 42, ypdf(height, ly + 57), "Non-2.5 all SNPs", 11, BLACK)
    fill(cmds, BLUE_LIGHT)
    cmds.append(f"{lx + 176:.2f} {ypdf(height, ly + 50):.2f} 8 8 re f")
    text(cmds, lx + 192, ypdf(height, ly + 57), "protein-changing", 11, BLACK)

    return cmds


if __name__ == "__main__":
    main()
