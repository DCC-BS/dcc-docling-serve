# Patches of upstream packages in the image

Applied by `plugins/Dockerfile.docling-serve` with `patch -p1` in site-packages, after a check that
the installed versions are the ones the patches were made for. A docling-serve release with other
versions fails the build at that step: refresh the patches against the new files (or drop the parts
upstream has merged) before bumping `DOCLING_SERVE_TAG`.

| Patch | Against | What it does |
|---|---|---|
| `docling-ibm-models-tableformer.patch` | docling-ibm-models 4.0.4 | TableFormer v1 tag decoder with a KV cache and a CUDA graph (upstream [docling-ibm-models#210](https://github.com/docling-project/docling-ibm-models/pull/210)), 8 decode steps per graph replay with blocking waits, vectorised cell matching, normalisation by lookup table |
| `docling-table-batch.patch` | docling-slim 2.142.1 | all tables of a page batch in one TableFormer call (each page resized once), cell lookup without deep copies |

All give the same tables as upstream (709/709 tables and all 96 test documents byte-identical).
Each change can be turned off at runtime, which runs the upstream code path:

| Env var | Default | Change |
|---|---|---|
| `DCC_TABLES_DECODER` | `1` | 8 steps per CUDA graph replay, blocking CUDA events, cache-length buckets (the KV-cache decoder itself stays) |
| `DCC_TABLES_MATCH` | `1` | cell matching and post-processing with numpy candidate search, no deep copies |
| `DCC_TABLES_PREP` | `1` | image normalisation with `cv2.LUT` instead of float64 arithmetic |
| `DCC_TABLES_BATCH` | `1` | tables of a page batch in one call (`DCC_TABLES_MAXB`, default 4, tables per decoder batch) |
| `DCC_TABLES_CELLS` | `1` | page word cells of a table without deep copies |
| `DCC_TABLES_RESIZE` | `0` | resize only the table's row band (same pixels, no measurable gain) |

To regenerate a patch: unpack the original files from the upstream image into `a/`, the changed
ones into `b/`, and run `diff -ruN a/<package> b/<package>` from their parent directory.
