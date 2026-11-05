# 50-page parity

The fixture is 50 PubTables-v2 single pages with their dataset words file. The pins are the raw
ONNX outputs of the v3 checkpoint (`arrays/ort_cpu__<id>.npz`: logits, boxes, query embeddings,
encoder indices), one file per page.

`run.py` replays each page through the shipped ONNX files with the dataset words and compares
every array. On 26 October 2026, on `ae3-image:0.2.0`, the result was:

```
RESULT pages=50 exact=50 mismatched=0
```

Every array matched with `max_abs = 0.0`, so the shipped weights, the preprocessing and the
session options reproduce the pinned v3 outputs exactly. The images are the PubTables-v2 page
images and the words are the dataset annotation; neither ships in this zip for size reasons.
