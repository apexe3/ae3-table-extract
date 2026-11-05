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

"""Vectorised cell builder: the same cells as objects_to_cells(), without the per-object Python
geometry.

Boxes travel as numpy arrays, each token joins the row or column it overlaps most in one
vectorised pass, and spanning cells are derived from row/column index ranges. Adapted from the
builder used in the 25 October 2026 A10 throughput work. The 0.5 thresholds and the 10 px pad
below are the ones objects_to_cells() is called with in tables.py.
"""
from collections import defaultdict

import numpy as np

PAD = 10
CLASS_NAMES = ['table', 'table row', 'table column', 'table column header',
               'table projected row header', 'table spanning cell']
# page head class -> tlib crop class (head 1 is column, head 2 is row)
T2C = {0: 0, 1: 2, 2: 1, 3: 3, 4: 4, 5: 5, 8: 0, 9: 2, 10: 1, 11: 3, 12: 4, 13: 5}
FAMILY = {0: (1, 2, 3, 4, 5), 8: (9, 10, 11, 12, 13)}


def _boxes(objs):
    if len(objs) == 0:
        return np.zeros((0, 4))
    return np.array([o['bbox'] for o in objs], dtype=np.float64).reshape(-1, 4)


def _area(b):
    return np.maximum(0.0, b[..., 2] - b[..., 0]) * np.maximum(0.0, b[..., 3] - b[..., 1])


def _inter(a, b):
    """Pairwise intersection areas: a (n,4) x b (m,4) -> (n,m)."""
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    return np.maximum(0.0, x1 - x0) * np.maximum(0.0, y1 - y0)


def _frac(a, b):
    """Intersection over the area of each box in a, against every box in b."""
    ar = _area(a)
    with np.errstate(divide='ignore', invalid='ignore'):
        f = _inter(a, b) / ar[:, None]
    if f.shape[0] and f.shape[1]:
        f[ar == 0] = 0.0
    return f


def _union_box(boxes):
    if len(boxes) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(np.min(boxes[:, 0])), float(np.min(boxes[:, 1])),
            float(np.max(boxes[:, 2])), float(np.max(boxes[:, 3]))]


def _intersect_box(a, b):
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1])
    x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return [0.0, 0.0, 0.0, 0.0]
    return [x0, y0, x1, y1]




def _slot_assign(containers, packages, threshold=0.5, forced=False):
    """slot_into_containers(unique_assignment=True): every package joins the container
    holding the largest fraction of it, if that fraction reaches the threshold.
    Returns (package indices per container, best match score per package)."""
    nc, npk = len(containers), len(packages)
    out = [[] for _ in range(nc)]
    if nc == 0 or npk == 0:
        return out, []
    frac = _frac(packages, containers)          # (npk, nc)
    best = frac.argmax(axis=1)
    best_score = frac[np.arange(npk), best]
    take = np.ones(npk, bool) if forced else best_score >= threshold
    for j in np.nonzero(take)[0]:
        out[int(best[j])].append(int(j))
    return out, best_score.tolist()


def _nms_by_containment(containers, packages, thr=0.5):
    """Keep the highest scoring container among those sharing packages."""
    cont = sorted(containers, key=lambda o: -o['score'])
    n = len(cont)
    if n == 0:
        return []
    by_cont, _ = _slot_assign(_boxes(cont), _boxes(packages), thr)
    if not packages:
        return []
    assign = np.zeros((n, len(packages)), dtype=np.int32)
    for i, lst in enumerate(by_cont):
        if lst:
            assign[i, lst] = 1
    share = (assign @ assign.T) > 0
    suppressed = [False] * n
    for o2 in range(1, n):
        if not by_cont[o2]:
            suppressed[o2] = True
        for o1 in range(o2):
            if not suppressed[o1] and share[o1, o2]:
                suppressed[o2] = True
                break
    return [o for i, o in enumerate(cont) if not suppressed[i]]


def _nms(objects, match_criteria='object2_overlap', thr=0.05, keep_higher=True):
    if len(objects) == 0:
        return []
    objs = sorted(objects, key=(lambda o: -o['score']) if keep_higher else (lambda o: o['score']))
    n = len(objs)
    b = _boxes(objs)
    ar = _area(b)
    inter = _inter(b, b)
    suppressed = [False] * n
    for o2 in range(1, n):
        a2 = ar[o2]
        if a2 <= 0:
            continue
        cand = [o1 for o1 in range(o2) if not suppressed[o1]]
        if not cand:
            continue
        cand = np.array(cand)
        num = inter[cand, o2]
        if match_criteria == 'object1_overlap':
            den = ar[cand]
        elif match_criteria == 'iou':
            den = ar[cand] + a2 - num
        else:
            den = np.full(len(cand), a2)
        ok = den > 0
        if ok.any() and np.any(num[ok] / den[ok] >= thr):
            suppressed[o2] = True
    return [o for i, o in enumerate(objs) if not suppressed[i]]


def _remove_without_content(spans, objects):
    """Drop rows/columns whose box holds no text."""
    if len(objects) == 0:
        return objects
    if len(spans) == 0:
        return []
    frac = _frac(_boxes(spans), _boxes(objects))
    keep = []
    for i, obj in enumerate(objects):
        idx = np.nonzero(frac[:, i] >= 0.5)[0]
        text = extract_text_from_spans([spans[j] for j in idx], remove_integer_superscripts=True)
        if len(text.strip()) > 0:
            keep.append(obj)
    return keep


def _refine_rows(rows, tokens):
    if len(tokens) > 0:
        rows = _nms_by_containment(rows, tokens, 0.5)
        rows = _remove_without_content(tokens, rows)
    else:
        rows = _nms(rows, 'object2_overlap', 0.5, True)
    if len(rows) > 1:
        rows = sorted(rows, key=lambda k: k['bbox'][1] + k['bbox'][3])
    return rows


def _refine_columns(columns, tokens):
    if len(tokens) > 0:
        columns = _nms_by_containment(columns, tokens, 0.5)
        columns = _remove_without_content(tokens, columns)
    else:
        columns = _nms(columns, 'object2_overlap', 0.25, True)
    if len(columns) > 1:
        columns = sorted(columns, key=lambda k: k['bbox'][0] + k['bbox'][2])
    return columns


def _align_headers(headers, rows):
    """Header = convex hull of the leading run of rows the header box covers >=50% of."""
    for row in rows:
        row['header'] = False
    if len(headers) == 0 or len(rows) == 0:
        return []
    ra = _boxes(rows)
    ha = _boxes(headers)
    row_h = ra[:, 3] - ra[:, 1]
    overlap = (np.minimum(ra[None, :, 3], ha[:, None, 3])
               - np.maximum(ra[None, :, 1], ha[:, None, 1]))
    with np.errstate(divide='ignore', invalid='ignore'):
        hit = (overlap / row_h[None, :]) >= 0.5
    header_row_nums = []
    for hi in range(len(headers)):
        header_row_nums.extend(int(i) for i in np.nonzero(hit[hi])[0])
    if len(header_row_nums) == 0:
        return []
    if header_row_nums[0] > 0:
        header_row_nums = list(range(header_row_nums[0] + 1)) + header_row_nums
    hdr_rows = []
    last_row_num = -1
    for row_num in header_row_nums:
        if row_num == last_row_num + 1:
            rows[row_num]['header'] = True
            hdr_rows.append(row_num)
            last_row_num = row_num
        else:
            break
    return [{'bbox': _union_box(_boxes([rows[i] for i in hdr_rows]))}]


def _align_supercells(supercells, rows, columns):
    """Snap each supercell to the rows/columns it covers >=50% of; keep its index range."""
    if len(supercells) == 0 or len(rows) == 0 or len(columns) == 0:
        return []
    ra = _boxes(rows)
    ca = _boxes(columns)
    row_h = ra[:, 3] - ra[:, 1]
    col_w = ca[:, 2] - ca[:, 0]
    row_is_hdr = np.array([bool(r.get('header')) for r in rows])
    aligned = []
    for sc in supercells:
        sb = np.asarray(sc['bbox'], dtype=np.float64)
        sc['header'] = False
        sc_h = sb[3] - sb[1]
        sc_w = sb[2] - sb[0]
        is_span = 'span' in sc
        ov = np.minimum(ra[:, 3], sb[3]) - np.maximum(ra[:, 1], sb[1])
        with np.errstate(divide='ignore', invalid='ignore'):
            if is_span:
                fr = np.maximum(ov / row_h, ov / sc_h)
            else:
                fr = ov / row_h
        hit_r = np.nonzero(fr >= 0.5)[0]
        hdr_rows = set()
        data_rows = set()
        for rn in hit_r:
            if row_is_hdr[rn]:
                hdr_rows.add(int(rn))
            else:
                data_rows.add(int(rn))
        if len(data_rows) > 0 and len(hdr_rows) > 0:
            if len(data_rows) > len(hdr_rows):
                hdr_rows = set()
            else:
                data_rows = set()
        if len(hdr_rows) > 0:
            sc['header'] = True
        elif is_span:
            continue
        intersecting_rows = data_rows.union(hdr_rows)
        if len(intersecting_rows) == 0:
            continue
        row_box = _union_box(ra[sorted(intersecting_rows)])
        ovc = np.minimum(ca[:, 2], sb[2]) - np.maximum(ca[:, 0], sb[0])
        with np.errstate(divide='ignore', invalid='ignore'):
            if is_span:
                frc = np.maximum(ovc / col_w, ovc / sc_w)
                if sc['header']:
                    frc = frc * 2
            else:
                frc = ovc / col_w
        hit_c = np.nonzero(frc >= 0.5)[0]
        if len(hit_c) == 0:
            continue
        col_box = _union_box(ca[hit_c])
        sc['bbox'] = _intersect_box(row_box, col_box)
        if len(intersecting_rows) > 0 and len(hit_c) > 0 and (len(intersecting_rows) > 1 or len(hit_c) > 1):
            sc['row_numbers'] = list(intersecting_rows)
            sc['column_numbers'] = [int(i) for i in hit_c]
            aligned.append(sc)
            if is_span and sc['header'] and len(sc['column_numbers']) > 1:
                for row_num in range(0, min(sc['row_numbers'])):
                    new_supercell = {'row_numbers': [row_num],
                                     'column_numbers': sc['column_numbers'],
                                     'score': sc['score'], 'propagated': True}
                    new_cols = [columns[i] for i in sc['column_numbers']]
                    new_rows = [rows[i] for i in sc['row_numbers']]
                    new_supercell['bbox'] = [min(c['bbox'][0] for c in new_cols),
                                             min(r['bbox'][1] for r in new_rows),
                                             max(c['bbox'][2] for c in new_cols),
                                             max(r['bbox'][3] for r in new_rows)]
                    aligned.append(new_supercell)
    return aligned



def is_int(x):
    try:
        int(x)
        return True
    except (TypeError, ValueError):
        return False


def extract_text_from_spans(spans, join_with_space=True, remove_integer_superscripts=True):
    """
    Convert a collection of page tokens/words/spans into a single text string.
    """

    if join_with_space:
        join_char = " "
    else:
        join_char = ""
    spans_copy = spans[:]
    
    if remove_integer_superscripts:
        for span in spans:
            if not 'flags' in span:
                continue
            flags = span['flags']
            if flags & 2**0: # superscript flag
                if is_int(span['text']):
                    spans_copy.remove(span)
                else:
                    span['superscript'] = True

    if len(spans_copy) == 0:
        return ""
    
    spans_copy.sort(key=lambda span: span['span_num'])
    spans_copy.sort(key=lambda span: span['line_num'])
    spans_copy.sort(key=lambda span: span['block_num'])
    
    # Force the span at the end of every line within a block to have exactly one space
    # unless the line ends with a space or ends with a non-space followed by a hyphen
    line_texts = []
    line_span_texts = [spans_copy[0]['text']]
    for span1, span2 in zip(spans_copy[:-1], spans_copy[1:]):
        if not span1['block_num'] == span2['block_num'] or not span1['line_num'] == span2['line_num']:
            line_text = join_char.join(line_span_texts).strip()
            if (len(line_text) > 0
                    and not line_text[-1] == ' '
                    and not (len(line_text) > 1 and line_text[-1] == "-" and not line_text[-2] == ' ')):
                if not join_with_space:
                    line_text += ' '
            line_texts.append(line_text)
            line_span_texts = [span2['text']]
        else:
            line_span_texts.append(span2['text'])
    line_text = join_char.join(line_span_texts)
    line_texts.append(line_text)
            
    return join_char.join(line_texts).strip()




def sort_objects_by_score(objects, reverse=True):
    return sorted(objects, key=(lambda k: -k['score']) if reverse else (lambda k: k['score']))


def nms_supercells(supercells):
    """
    A NMS scheme for supercells that first attempts to shrink supercells to
    resolve overlap.
    If two supercells overlap the same (sub)cell, shrink the lower confidence
    supercell to resolve the overlap. If shrunk supercell is empty, remove it.
    """

    supercells = sort_objects_by_score(supercells)
    num_supercells = len(supercells)
    suppression = [False for supercell in supercells]

    for supercell2_num in range(1, num_supercells):
        supercell2 = supercells[supercell2_num]
        for supercell1_num in range(supercell2_num):
            supercell1 = supercells[supercell1_num]
            remove_supercell_overlap(supercell1, supercell2)
        if ((len(supercell2['row_numbers']) < 2 and len(supercell2['column_numbers']) < 2)
                or len(supercell2['row_numbers']) == 0 or len(supercell2['column_numbers']) == 0):
            suppression[supercell2_num] = True

    return [obj for idx, obj in enumerate(supercells) if not suppression[idx]]




def header_supercell_tree(supercells):
    """
    Make sure no supercell in the header is below more than one supercell in any row above it.
    The cells in the header form a tree, but a supercell with more than one supercell in a row
    above it means that some cell has more than one parent, which is not allowed. Eliminate
    any supercell that would cause this to be violated.
    """
    header_supercells = [supercell for supercell in supercells if 'header' in supercell and supercell['header']]
    header_supercells = sort_objects_by_score(header_supercells)
    
    for header_supercell in header_supercells[:]:
        ancestors_by_row = defaultdict(int)
        min_row = min(header_supercell['row_numbers'])
        for header_supercell2 in header_supercells:
            max_row2 = max(header_supercell2['row_numbers'])
            if max_row2 < min_row:
                if (set(header_supercell['column_numbers']).issubset(
                    set(header_supercell2['column_numbers']))):
                    for row2 in header_supercell2['row_numbers']:
                        ancestors_by_row[row2] += 1
        for row in range(0, min_row):
            if not ancestors_by_row[row] == 1:
                supercells.remove(header_supercell)
                break
                
                


def remove_supercell_overlap(supercell1, supercell2):
    """
    This function resolves overlap between supercells (supercells must be
    disjoint) by iteratively shrinking supercells by the fewest grid cells
    necessary to resolve the overlap.
    Example:
    If two supercells overlap at grid cell (R, C), and supercell #1 is less
    confident than supercell #2, we eliminate either row R from supercell #1
    or column C from supercell #1 by comparing the number of columns in row R
    versus the number of rows in column C. If the number of columns in row R
    is less than the number of rows in column C, we eliminate row R from
    supercell #1. This resolves the overlap by removing fewer grid cells from
    supercell #1 than if we eliminated column C from it.
    """
    common_rows = set(supercell1['row_numbers']).intersection(set(supercell2['row_numbers']))
    common_columns = set(supercell1['column_numbers']).intersection(set(supercell2['column_numbers']))

    # While the supercells have overlapping grid cells, continue shrinking the less-confident
    # supercell one row or one column at a time
    while len(common_rows) > 0 and len(common_columns) > 0:
        # Try to shrink the supercell as little as possible to remove the overlap;
        # if the supercell has fewer rows than columns, remove an overlapping column,
        # because this removes fewer grid cells from the supercell;
        # otherwise remove an overlapping row
        if len(supercell2['row_numbers']) < len(supercell2['column_numbers']):
            min_column = min(supercell2['column_numbers'])
            max_column = max(supercell2['column_numbers'])
            if max_column in common_columns:
                common_columns.remove(max_column)
                supercell2['column_numbers'].remove(max_column)
            elif min_column in common_columns:
                common_columns.remove(min_column)
                supercell2['column_numbers'].remove(min_column)
            else:
                supercell2['column_numbers'] = []
                common_columns = set()
        else:
            min_row = min(supercell2['row_numbers'])
            max_row = max(supercell2['row_numbers'])
            if max_row in common_rows:
                common_rows.remove(max_row)
                supercell2['row_numbers'].remove(max_row)
            elif min_row in common_rows:
                common_rows.remove(min_row)
                supercell2['row_numbers'].remove(min_row)
            else:
                supercell2['row_numbers'] = []
                common_rows = set()


def table_structures(tokens, objects):
    """objects_to_table_structures: rows/columns/headers/supercells, refined and aligned."""
    columns = [o for o in objects if CLASS_NAMES[o['label']] == 'table column']
    rows = [o for o in objects if CLASS_NAMES[o['label']] == 'table row']
    headers = [o for o in objects if CLASS_NAMES[o['label']] == 'table column header']
    supercells = [o for o in objects if CLASS_NAMES[o['label']] == 'table spanning cell']
    for obj in supercells:
        obj['subheader'] = False
    subheaders = [o for o in objects if CLASS_NAMES[o['label']] == 'table projected row header']
    for obj in subheaders:
        obj['subheader'] = True
    supercells = supercells + subheaders

    if len(rows) and len(headers):
        hdr = _frac(_boxes(rows), _boxes(headers)).max(axis=1) >= 0.5
    else:
        hdr = np.zeros(len(rows), dtype=bool)
    for i, row in enumerate(rows):
        row['header'] = bool(hdr[i])

    rows = _refine_rows(rows, tokens)
    columns = _refine_columns(columns, tokens)

    # table bbox shrunk to the total height of the rows and width of the columns
    row_box = _union_box(_boxes(rows))
    col_box = _union_box(_boxes(columns))
    bbox = [col_box[0], row_box[1], col_box[2], row_box[3]]
    for column in columns:
        column['bbox'][1] = bbox[1]
        column['bbox'][3] = bbox[3]
    for row in rows:
        row['bbox'][0] = bbox[0]
        row['bbox'][2] = bbox[2]

    structs = {'rows': rows, 'columns': columns, 'headers': headers,
               'supercells': supercells, 'bbox': bbox}
    if len(rows) > 0 and len(columns) > 1:
        structs = _refine_table_structures(bbox, structs)
    return structs


def _refine_table_structures(table_bbox, structs):
    rows = structs['rows']
    columns = structs['columns']
    headers = [o for o in structs['headers'] if o['score'] >= 0.5]
    headers = _nms(headers)
    headers = _align_headers(headers, rows)

    supercells = [e for e in structs['supercells'] if not e['subheader']]
    subheaders = [e for e in structs['supercells'] if e['subheader']]
    supercells = [o for o in supercells if o['score'] >= 0.5]
    subheaders = [o for o in subheaders if o['score'] >= 0.5]
    supercells = supercells + subheaders
    supercells = _align_supercells(supercells, rows, columns)
    supercells = nms_supercells(supercells)
    header_supercell_tree(supercells)

    structs['columns'] = columns
    structs['rows'] = rows
    structs['supercells'] = supercells
    structs['headers'] = headers
    return structs


def _dilate_and_fit(cells, rows, columns, span_by_cell, table_spans):
    """Final box pass: dilate cells from their index ranges, then pull row and column
    boundaries onto the text they contain (same two passes as the reference builder)."""
    R, C = len(rows), len(columns)
    ra = _boxes(rows)
    ca = _boxes(columns)
    for cell in cells:
        rn = cell['row_nums']
        cn = cell['column_nums']
        if len(rn) == 1 and len(cn) == 1:
            continue
        row_box = _union_box(ra[rn]) if rn else [0.0, 0.0, 0.0, 0.0]
        col_box = _union_box(ca[cn]) if cn else [0.0, 0.0, 0.0, 0.0]
        cell['bbox'] = _intersect_box(row_box, col_box)

    if len(cells) == 0:
        return cells

    counts = np.array([len(s) for s in span_by_cell])
    if counts.sum() == 0:
        return cells
    cell_idx = np.repeat(np.arange(len(cells)), counts)
    span_idx = np.concatenate([np.asarray(s, dtype=np.int64) for s in span_by_cell if len(s)])
    sb = _boxes(table_spans)[span_idx]
    min_row = np.array([min(c['row_nums']) for c in cells])[cell_idx]
    max_row = np.array([max(c['row_nums']) for c in cells])[cell_idx]
    min_col = np.array([min(c['column_nums']) for c in cells])[cell_idx]
    max_col = np.array([max(c['column_nums']) for c in cells])[cell_idx]

    def _extreme(size, idx, values, ufunc, fill):
        out = np.full(size, fill)
        ufunc.at(out, idx, values)
        return out

    min_x_by_col = _extreme(C, min_col, sb[:, 0], np.minimum, np.inf)
    min_y_by_row = _extreme(R, min_row, sb[:, 1], np.minimum, np.inf)
    max_x_by_col = _extreme(C, max_col, sb[:, 2], np.maximum, -np.inf)
    max_y_by_row = _extreme(R, max_row, sb[:, 3], np.maximum, -np.inf)

    if np.isfinite(min_x_by_col[0]):
        ra[:, 0] = min_x_by_col[0]
    if np.isfinite(max_x_by_col[C - 1]):
        ra[:, 2] = max_x_by_col[C - 1]
    ra[:, 1] = np.where(np.isfinite(min_y_by_row), min_y_by_row, ra[:, 1])
    ra[:, 3] = np.where(np.isfinite(max_y_by_row), max_y_by_row, ra[:, 3])

    ca[:, 1] = np.where(np.isfinite(min_y_by_row[0]), min_y_by_row[0], ca[:, 1])
    ca[:, 0] = np.where(np.isfinite(min_x_by_col), min_x_by_col, ca[:, 0])
    ca[:, 2] = np.where(np.isfinite(max_x_by_col), max_x_by_col, ca[:, 2])
    ca[:, 3] = np.where(np.isfinite(max_y_by_row[R - 1]), max_y_by_row[R - 1], ca[:, 3])

    rx0, ry0, rx1, ry1 = ra[:, 0].tolist(), ra[:, 1].tolist(), ra[:, 2].tolist(), ra[:, 3].tolist()
    cx0, cy0, cx1, cy1 = ca[:, 0].tolist(), ca[:, 1].tolist(), ca[:, 2].tolist(), ca[:, 3].tolist()
    for cell in cells:
        rn = cell['row_nums']
        cn = cell['column_nums']
        r = [min(rx0[i] for i in rn), min(ry0[i] for i in rn),
             max(rx1[i] for i in rn), max(ry1[i] for i in rn)]
        c = [min(cx0[i] for i in cn), min(cy0[i] for i in cn),
             max(cx1[i] for i in cn), max(cy1[i] for i in cn)]
        box = _intersect_box(r, c)
        if _area(np.asarray(box, dtype=np.float64)) > 0:
            cell['bbox'] = box
    return cells


def _cell_grid(rows, columns):
    """Every row x column intersection: boxes (R,C,4) and areas (R,C)."""
    ra = _boxes(rows)
    ca = _boxes(columns)
    x0 = np.maximum(ra[:, None, 0], ca[None, :, 0])
    y0 = np.maximum(ra[:, None, 1], ca[None, :, 1])
    x1 = np.minimum(ra[:, None, 2], ca[None, :, 2])
    y1 = np.minimum(ra[:, None, 3], ca[None, :, 3])
    empty = (x1 <= x0) | (y1 <= y0)
    grid = np.stack([np.where(empty, 0.0, x0), np.where(empty, 0.0, y0),
                     np.where(empty, 0.0, x1), np.where(empty, 0.0, y1)], axis=-1)
    return grid, np.where(empty, 0.0, (grid[..., 2] - grid[..., 0]) * (grid[..., 3] - grid[..., 1]))


def cells_from_structures(structs, table_spans):
    """table_structure_to_cells: complete cells, supercell cells, text, final boxes."""
    rows = structs['rows']
    columns = structs['columns']
    supercells = structs['supercells']
    R, C = len(rows), len(columns)
    grid, grid_area = _cell_grid(rows, columns)

    # which grid cells are covered (>50%) by a supercell
    if len(supercells):
        inter = _inter(_boxes(supercells), grid.reshape(-1, 4))
        with np.errstate(divide='ignore', invalid='ignore'):
            covered = (inter / grid_area.reshape(-1)[None, :]) > 0.5
        is_subcell = covered.any(axis=0).reshape(R, C)
    else:
        is_subcell = np.zeros((R, C), dtype=bool)

    row_hdr = [bool(r.get('header')) for r in rows]
    cells = []
    sub_rows = []
    sub_cols = []
    for c in range(C):
        for r in np.nonzero(~is_subcell[:, c])[0]:
            cells.append({'bbox': grid[r, c].tolist(), 'column_nums': [int(c)], 'row_nums': [int(r)],
                          'header': row_hdr[r], 'subcell': False, 'subheader': False})
        for r in np.nonzero(is_subcell[:, c])[0]:
            sub_rows.append(int(r))
            sub_cols.append(int(c))

    # supercell cells: the subcells each supercell covers, unioned
    if len(supercells) and sub_rows:
        sub_b = grid[sub_rows, sub_cols]
        sub_area = _area(sub_b)
        inter = _inter(_boxes(supercells), sub_b)
        with np.errstate(divide='ignore', invalid='ignore'):
            frac = inter / sub_area[None, :]
        for k, supercell in enumerate(supercells):
            hit = np.nonzero(frac[k] > 0.5)[0]
            if len(hit) == 0:
                continue
            cell_rows = set()
            cell_columns = set()
            for h in hit:
                cell_rows.add(sub_rows[h])
                cell_columns.add(sub_cols[h])
            cells.append({'bbox': _union_box(sub_b[hit]), 'column_nums': list(cell_columns),
                          'row_nums': list(cell_rows),
                          'header': bool(all(row_hdr[i] for i in cell_rows)),
                          'subheader': supercell['subheader']})

    # confidence: how well the tokens slot into the cells (kept for parity)
    _, best_scores = _slot_assign(_boxes(cells), _boxes(table_spans), 0.5)
    if best_scores:
        confidence_score = (sum(best_scores) / len(best_scores) + min(best_scores)) / 2
    else:
        confidence_score = 0

    # one span -> cell assignment for the whole table
    span_by_cell, _ = _slot_assign(_boxes(cells), _boxes(table_spans), 0.001)
    span_boxes = _boxes(table_spans)
    for cell, span_nums in zip(cells, span_by_cell):
        cell_spans = [table_spans[i] for i in span_nums]
        cell['cell_text'] = extract_text_from_spans(cell_spans, remove_integer_superscripts=False)
        cell['spans'] = cell_spans

    _dilate_and_fit(cells, rows, columns, span_by_cell, table_spans)
    return cells, confidence_score


def cells_for_table(tab_objs, words, tbox, rot, pad=PAD):
    """Structure objects and page words -> (cells, content box, confidence), in the table's frame.

    tab_objs are the page detections of the table's structure family, with cls, bbox and score;
    tbox is the detected table box. rot marks a table rotated 90 degrees clockwise. The cells are
    the shape cell_records() expects; the content box is the box the detected rows and columns
    span.
    """
    tb = [tbox[0] - pad, tbox[1] - pad, tbox[2] + pad, tbox[3] + pad]
    cx0, cy0 = tb[0], tb[1]
    ch = tb[3] - tb[1]
    wb = _boxes(words)
    if len(wb):
        sel = np.nonzero(_frac(wb, np.asarray([tb], dtype=np.float64))[:, 0] >= 0.5)[0]
    else:
        sel = np.zeros(0, dtype=np.int64)
    boxes = wb[sel] - np.asarray([cx0, cy0, cx0, cy0]) if len(sel) else np.zeros((0, 4))
    if rot and len(boxes):
        b = boxes
        boxes = np.stack([ch - b[:, 3] - 1.0, b[:, 0], ch - b[:, 1] - 1.0, b[:, 2]], axis=1)

    toks = []
    for i, j in enumerate(sel):
        w = words[j]
        toks.append({'bbox': boxes[i].tolist(), 'text': w['text'],
                     'block_num': w.get('block_num', 0), 'line_num': w.get('line_num', 0),
                     'span_num': w.get('span_num', 0), 'flags': w.get('flags', 0)})

    objs = []
    for o in tab_objs:
        b = o['bbox']
        bx = [b[0] - cx0, b[1] - cy0, b[2] - cx0, b[3] - cy0]
        if rot:
            bx = [ch - bx[3] - 1.0, bx[0], ch - bx[1] - 1.0, bx[2]]
        objs.append({'bbox': bx, 'label': T2C[o['cls']], 'score': o['score'], 'page_num': 0})

    structs = table_structures(toks, objs)
    if len(structs['columns']) < 1 or len(structs['rows']) < 1:
        return [], structs['bbox'], 0
    cells, confidence = cells_from_structures(structs, toks)
    return cells, structs['bbox'], confidence


def pred_cells_for_table(tab_objs, words, tbox, rot):
    """One table's cells only, the entry point the A10 timing harness used."""
    return cells_for_table(tab_objs, words, tbox, rot)[0]
