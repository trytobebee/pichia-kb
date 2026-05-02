"""Probe whether figures in each paper PDF are vector (SVG-like paths) or raster (embedded bitmap).

Heuristic per page:
- Counts embedded raster images and their pixel dimensions
- Counts vector drawing primitives (lines, beziers, fills)
- Verdict for each page that *has any figure-like content*:
    raster   → page has a big embedded image and few/no drawings
    vector   → page has many vector primitives and no big embedded image
    mixed    → both present in significant amounts
    text-only/unknown otherwise

Outputs a per-paper summary and a per-page verdict for figure-bearing pages only.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import fitz

PAPERS_DIR = Path(__file__).parent.parent / "data" / "projects" / "pichia-collagen" / "papers"

# Thresholds (tuned for letter/A4 academic PDFs)
BIG_IMAGE_PX = 200 * 200          # an "image" smaller than this is likely a logo/decoration
MANY_DRAWINGS = 30                 # a page with this many path primitives is "drawing-heavy"
FIGURE_PAGE_DRAWINGS_MIN = 8       # below this we don't consider the page to have a figure


def classify_page(page: fitz.Page) -> tuple[str, dict]:
    images = page.get_images(full=True)
    big_images: list[tuple[int, int]] = []  # (w, h) in pixels of the source XObject
    for img in images:
        # img tuple: (xref, smask, w, h, bpc, colorspace, ...)
        try:
            w, h = img[2], img[3]
            if w * h >= BIG_IMAGE_PX:
                big_images.append((w, h))
        except Exception:
            continue

    try:
        drawings = page.get_drawings()
    except Exception:
        drawings = []
    n_draw = len(drawings)

    has_big_img = bool(big_images)
    has_many_draw = n_draw >= MANY_DRAWINGS

    # Verdict
    if has_big_img and has_many_draw:
        verdict = "mixed"
    elif has_big_img:
        verdict = "raster"
    elif has_many_draw:
        verdict = "vector"
    elif n_draw >= FIGURE_PAGE_DRAWINGS_MIN:
        verdict = "vector?"  # weak vector signal
    elif images:
        verdict = "raster?"  # only small images, prob not a chart
    else:
        verdict = "text"

    return verdict, {
        "n_big_images": len(big_images),
        "biggest_image_px": max((w * h for w, h in big_images), default=0),
        "n_drawings": n_draw,
    }


def survey(pdf_path: Path) -> None:
    doc = fitz.open(str(pdf_path))
    n = len(doc)
    verdicts: list[str] = []
    fig_pages: list[tuple[int, str, dict]] = []
    for i in range(n):
        v, info = classify_page(doc[i])
        verdicts.append(v)
        if v in ("raster", "vector", "mixed", "vector?"):
            fig_pages.append((i + 1, v, info))
    doc.close()

    name = pdf_path.stem
    print(f"\n=== {name}  ({n} pages) ===")
    counter = Counter(verdicts)
    summary = "  ".join(f"{k}={counter[k]}" for k in sorted(counter))
    print(f"  全体: {summary}")
    print(f"  含图页面: {len(fig_pages)}  (verdicts: " +
          ", ".join(f"{k}={sum(1 for _,vv,_ in fig_pages if vv==k)}"
                    for k in ("vector", "vector?", "raster", "mixed")) +
          ")")
    # Show first few interesting pages
    interesting = [p for p in fig_pages if p[1] in ("vector", "mixed", "raster")][:6]
    for page_no, v, info in interesting:
        print(f"    p{page_no:>3}  {v:<6}  drawings={info['n_drawings']:>4}  "
              f"big_imgs={info['n_big_images']}  biggest_px={info['biggest_image_px']:,}")


def main() -> None:
    pdfs = sorted(PAPERS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs in {PAPERS_DIR}")
        sys.exit(1)
    for pdf in pdfs:
        try:
            survey(pdf)
        except Exception as e:
            print(f"\n=== {pdf.stem} ===\n  ERROR: {e}")


if __name__ == "__main__":
    main()
