"""raw/*.pdf -> data/me/pages/<doc>_p<n>.png at the scan's native resolution."""
import glob, os, re, sys
import pymupdf

src, out = (sys.argv[1:] + ["raw", "data/me/pages"])[:2] if len(sys.argv) > 1 else ("raw", "data/me/pages")
os.makedirs(out, exist_ok=True)
for f in sorted(glob.glob(os.path.join(src, "*.pdf"))):
    doc = re.sub(r"\W+", "_", os.path.splitext(os.path.basename(f))[0]).strip("_").lower()
    d = pymupdf.open(f)
    for i, page in enumerate(d, 1):
        xref = page.get_images()[0][0]
        pix = pymupdf.Pixmap(d, xref)
        if pix.n - pix.alpha >= 4:
            pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
        path = os.path.join(out, f"{doc}_p{i}.png")
        pix.save(path)
        print(path, pix.width, pix.height)
