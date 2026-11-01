'''Compare the two object detection backends on real images.

Run it over the stills in recordings/ (or any folder of images) before switching a
cat flap from tflite-support to ai-edge-litert, to see whether the detections - the
labels, scores and boxes the cat flap decides on - are the same.

Needs both libraries, which install together in a Python 3.9 environment:
    pip install tflite-support==0.4.3 ai-edge-litert "numpy<2" opencv-python-headless
    PYTHONPATH=src/catflap:src/modules python src/tools/compare_detectors.py recordings/ --model test/cats.tflite
'''
import argparse
import csv
import glob
import os
import sys

import cv2 as cv


def iou(a, b):
    ax2, ay2, bx2, by2 = a.origin_x + a.width, a.origin_y + a.height, b.origin_x + b.width, b.origin_y + b.height
    w = min(ax2, bx2) - max(a.origin_x, b.origin_x)
    h = min(ay2, by2) - max(a.origin_y, b.origin_y)
    inter = max(0, w) * max(0, h)
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0 else 0.0


def find_images(paths):
    found = []
    for p in paths:
        if os.path.isdir(p):
            for ext in ('jpg', 'jpeg', 'png'):
                found += glob.glob(os.path.join(p, '**', f'*.{ext}'), recursive=True)
        else:
            found += glob.glob(p)
    return sorted(found)


def best_per_label(detections):
    '''The system works with (label, score) pairs, so compare the best score per label'''
    best = {}
    for d in detections:
        if d.label not in best or d.score > best[d.label].score:
            best[d.label] = d
    return best


def compare_image(image, reference, candidate):
    ref, cand = list(reference.detect(image)), list(candidate.detect(image))
    row = {'ref_top': '', 'ref_score': 0.0, 'cand_top': '', 'cand_score': 0.0,
           'same_top': True, 'score_diff': 0.0, 'iou': 1.0, 'same_labels': True}
    if ref:
        row['ref_top'], row['ref_score'] = ref[0].label, ref[0].score
    if cand:
        row['cand_top'], row['cand_score'] = cand[0].label, cand[0].score
    row['same_top'] = row['ref_top'] == row['cand_top']
    row['score_diff'] = row['cand_score'] - row['ref_score']
    row['iou'] = iou(ref[0].box, cand[0].box) if ref and cand else (1.0 if not ref and not cand else 0.0)
    r, c = best_per_label(ref), best_per_label(cand)
    row['same_labels'] = set(r) == set(c)
    return row


def compare_images(images, reference, candidate):
    return [compare_image(image, reference, candidate) for image in images]


def compare(paths, reference, candidate):
    rows = []
    for path in paths:
        image = cv.imread(path)
        if image is None:
            continue
        row = compare_image(image, reference, candidate)
        row['file'] = path
        rows.append(row)
    return rows


def summarise(rows, worst=10):
    n = len(rows)
    if n == 0:
        return "No images found"
    same_top = sum(r['same_top'] for r in rows)
    same_labels = sum(r['same_labels'] for r in rows)
    diffs = sorted(abs(r['score_diff']) for r in rows if r['ref_top'] and r['same_top'])
    ious = [r['iou'] for r in rows if r['ref_top'] and r['cand_top']]
    lines = [f"{n} images",
             f"top label the same:        {same_top}/{n} ({100 * same_top / n:.1f}%)",
             f"same set of labels found:  {same_labels}/{n} ({100 * same_labels / n:.1f}%)"]
    if diffs:
        lines.append(f"score difference, same top label: median {diffs[len(diffs) // 2]:.3f}, "
                     f"95th percentile {diffs[int(len(diffs) * 0.95)]:.3f}, max {diffs[-1]:.3f}")
    if ious:
        lines.append(f"top box overlap (IoU): mean {sum(ious) / len(ious):.2f}, min {min(ious):.2f}")
    bad = [r for r in rows if not r['same_labels'] or not r['same_top']]
    bad.sort(key=lambda r: -abs(r['score_diff']))
    if bad:
        lines.append(f"\nImages where the answer differs ({len(bad)}), largest score change first:")
        for r in bad[:worst]:
            lines.append(f"  {r['file']}: tflite-support {r['ref_top'] or '-'} {r['ref_score']:.2f}"
                         f"  ->  litert {r['cand_top'] or '-'} {r['cand_score']:.2f}")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('images', nargs='+', help='Image files, globs or folders')
    parser.add_argument('--model', required=True)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--csv', help='Write every image to this file')
    parser.add_argument('--worst', type=int, default=10)
    args = parser.parse_args(argv)

    try:
        from tflite_detect import TFLiteDetect
        from litert_detect import LiteRTDetect
    except ImportError as e:
        sys.exit(f"Both tflite-support and ai-edge-litert must be installed: {e}")
    paths = find_images(args.images)
    rows = compare(paths, TFLiteDetect(args.model, False, args.threads), LiteRTDetect(args.model, False, args.threads))
    print(summarise(rows, args.worst))
    if args.csv and rows:
        with open(args.csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == '__main__':
    main()
