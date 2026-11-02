# AE3-TL — Open-Source Table Extraction Toolkit
# Copyright 2026 Team APEX:E3
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# This software is free and open source: you may use, modify and distribute
# it, including in commercial products, under the terms of the License.
# AE3-TL is created and maintained by Team APEX:E3.

"""The page model on ONNX Runtime: same inputs, same outputs, numpy post-processing.

predict(page_image, words, page_size) mirrors app/service/engine.py. The forward pass is the only
difference between the two: selection of the 300 queries, the object threshold, box scaling, the
cell builder and the v3 linking rule are the same numpy and Python code for every backend.
"""

import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from ae3_tl.postprocess import tables as T
from ae3_tl.postprocess.card import ModelConfig
from ae3_tl.backends.onnx_backend import OnnxNetwork, OnnxRelationHead, OnnxSettings, load_onnx_settings
from ae3_tl.pdf.preprocess import page_tensors
from ae3_tl.postprocess.prediction import PagePrediction

Image.MAX_IMAGE_PIXELS = None


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.float32(1.0) / (np.float32(1.0) + np.exp(-x.astype(np.float32)))


class EngineOrt:
    """Loads the ONNX files once and runs pages through ONNX Runtime."""

    def __init__(self, settings, onnx: OnnxSettings | None = None):
        self.settings = settings
        self.config = ModelConfig(settings.model_config_file)
        self.onnx = onnx or load_onnx_settings(settings.weights_dir)
        self.network = OnnxNetwork(self.onnx)
        self.relation = OnnxRelationHead(self.onnx)
        self.device = self.onnx.provider
        self.top_queries = int(self.config.thresholds["top_queries"])
        self.threshold = float(self.config.thresholds["object"])
        self.floor = float(self.config.thresholds["score_floor"])
        self.edge_threshold = float(self.config.thresholds["edge"])
        self.pad = float(self.config.thresholds["pad"])

    # ---------------------------------------------------------------- inference

    def predict(self, page_image: Image.Image, words: Sequence[dict],
                page_size: Sequence[int] | None = None) -> PagePrediction:
        return self.predict_many([(page_image, words, page_size)])[0]

    def predict_many(self, pages) -> list[PagePrediction]:
        out: list[PagePrediction] = []
        size = max(1, self.settings.batch_size)
        for start in range(0, len(pages), size):
            out.extend(self._predict_batch(pages[start:start + size]))
        return out

    def _predict_batch(self, pages) -> list[PagePrediction]:
        images, sizes, scales = [], [], []
        for image, _words, page_size in pages:
            width, height = page_size if page_size is not None else (image.width, image.height)
            sizes.append((int(width), int(height)))
            scales.append([float(width), float(height), float(width), float(height)])
            images.append(image)

        batch = page_tensors(images)
        started = time.perf_counter()
        logits, boxes, embeddings, encoder_indices = self.network.run(batch)

        prob = _sigmoid(logits)
        classes = prob.shape[-1]
        flat = prob.reshape(len(pages), -1)
        order = np.argsort(-flat, axis=1, kind="stable")[:, :self.top_queries]
        scores = np.take_along_axis(flat, order, axis=1)
        labels = order % classes
        query_ids = order // classes

        centred = boxes.astype(np.float32)
        xyxy = np.stack([centred[..., 0] - centred[..., 2] / 2.0,
                         centred[..., 1] - centred[..., 3] / 2.0,
                         centred[..., 0] + centred[..., 2] / 2.0,
                         centred[..., 1] + centred[..., 3] / 2.0], axis=-1)
        xyxy = xyxy * np.asarray(scales, dtype=np.float32)[:, None, :]
        selected = np.take_along_axis(xyxy, query_ids[..., None], axis=1)
        inference_ms = (time.perf_counter() - started) * 1000 / len(pages)

        detections = [self._page_detections(labels[k], selected[k], scores[k], query_ids[k])
                      for k in range(len(pages))]
        edges = [self._edge_probabilities(embeddings[k], detections[k]) for k in range(len(pages))]

        results = []
        for k, (_image, words, _size) in enumerate(pages):
            post_started = time.perf_counter()
            objects = detections[k]
            tables = T.page_tables(objects, words, edges[k]["caption"], edges[k]["footer"],
                                   self.edge_threshold, self.pad)
            results.append(PagePrediction(
                page_size=sizes[k],
                tables=tables,
                captions=edges[k]["caption_objects"],
                footers=edges[k]["footer_objects"],
                objects=len(objects),
                inference_ms=inference_ms,
                postprocess_ms=(time.perf_counter() - post_started) * 1000,
            ))
        return results

    def _page_detections(self, labels, boxes, scores, query_ids) -> list[dict[str, Any]]:
        out = []
        for cls, box, score, qid in zip(labels, boxes, scores, query_ids):
            if score < self.floor or score < self.threshold:
                continue
            out.append({"cls": int(cls), "bbox": [float(v) for v in box], "score": float(score),
                        "qid": int(qid)})
        return out

    def _edge_probabilities(self, embedding: np.ndarray, objects: Sequence[dict]
                            ) -> dict[str, Any]:
        """Relation-head scores between a caption or footer box and each table.

        Scores are kept in both directions, box -> table first and table -> box second: the v3
        linking rule ranks the mean of the two.
        """
        tables = [o for o in objects if o["cls"] in T.TABLE_CLASSES]
        out: dict[str, Any] = {}
        for kind, classes in (("caption", T.CAPTION_CLASSES), ("footer", T.FOOTER_CLASSES)):
            boxes = [o for o in objects if o["cls"] in classes]
            probabilities = [[[0.0, 0.0] for _ in tables] for _ in boxes]
            pairs, slots = [], []
            for i, box in enumerate(boxes):
                for j, table in enumerate(tables):
                    if table["qid"] == box["qid"]:
                        continue
                    pairs.append((box["qid"], table["qid"]))
                    slots.append((i, j, 0))
                    pairs.append((table["qid"], box["qid"]))
                    slots.append((i, j, 1))
            if pairs:
                edge = _sigmoid(self.relation.run(embedding, np.asarray(pairs, dtype=np.int64)))
                for (i, j, direction), value in zip(slots, edge.astype(np.float64).tolist()):
                    probabilities[i][j][direction] = value
            out[kind] = probabilities
            out[kind + "_objects"] = [
                {"bbox": [float(v) for v in box["bbox"]], "cls": box["cls"],
                 "score": float(box["score"]), "probabilities": probabilities[i]}
                for i, box in enumerate(boxes)
            ]
        return out
