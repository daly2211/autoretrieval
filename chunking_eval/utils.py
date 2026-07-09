import re
from fuzzywuzzy import fuzz
from fuzzywuzzy import process


def find_query_despite_whitespace(document, query):
    normalized_query = re.sub(r'\s+', ' ', query).strip()
    pattern = r'\s*'.join(re.escape(word) for word in normalized_query.split())
    regex = re.compile(pattern, re.IGNORECASE)
    match = regex.search(document)
    if match:
        return document[match.start(): match.end()], match.start(), match.end()
    return None


def rigorous_document_search(document: str, target: str):
    if target.endswith('.'):
        target = target[:-1]

    if target in document:
        start_index = document.find(target)
        end_index = start_index + len(target)
        return target, start_index, end_index

    raw_search = find_query_despite_whitespace(document, target)
    if raw_search is not None:
        return raw_search

    sentences = re.split(r'[.!?]\s*|\n', document)
    best_match = process.extractOne(target, sentences, scorer=fuzz.token_sort_ratio)
    if best_match[1] < 98:
        raise ValueError(
            f"No match found in document for target (best fuzzy score: {best_match[1]}). "
            f"Target: {target[:200]!r}")

    reference = best_match[0]
    start_index = document.find(reference)
    end_index = start_index + len(reference)
    return reference, start_index, end_index


# Range helpers

from typing import List, Optional, Tuple

RangeTuple = Tuple[int, int]


def sum_of_ranges(ranges: List[RangeTuple]) -> int:
    return sum(end - start for start, end in ranges)


def union_ranges(ranges: List[RangeTuple]) -> List[RangeTuple]:
    if not ranges:
        return []
    sorted_ranges = sorted(ranges, key=lambda x: x[0])
    merged = [sorted_ranges[0]]
    for current_start, current_end in sorted_ranges[1:]:
        last_start, last_end = merged[-1]
        if current_start <= last_end:
            merged[-1] = (last_start, max(last_end, current_end))
        else:
            merged.append((current_start, current_end))
    return merged


def intersect_two_ranges(range1: RangeTuple, range2: RangeTuple) -> Optional[RangeTuple]:
    start1, end1 = range1
    start2, end2 = range2
    intersect_start = max(start1, start2)
    intersect_end = min(end1, end2)
    if intersect_start <= intersect_end:
        return (intersect_start, intersect_end)
    return None


def difference(ranges: List[RangeTuple], target: RangeTuple) -> List[RangeTuple]:
    result: List[RangeTuple] = []
    target_start, target_end = target
    for start, end in ranges:
        if end < target_start or start > target_end:
            result.append((start, end))
        elif start < target_start and end > target_end:
            result.append((start, target_start))
            result.append((target_end, end))
        elif start < target_start:
            result.append((start, target_start))
        elif end > target_end:
            result.append((target_end, end))
    return result


def _safe_name(path: str) -> str:
    name = path.replace(".", "_").replace("/", "_").replace("\\", "_").strip("_")[:60]
    if not name or name[0] not in "abcdefghijklmnopqrstuvwxyz0123456789":
        name = "c_" + name
    return name


def chunks_to_ranges(corpus_text: str, chunks: List[str]) -> List[RangeTuple]:
    ranges: List[RangeTuple] = []
    for chunk in chunks:
        _, start, end = rigorous_document_search(corpus_text, chunk)
        ranges.append((start, end))
    return ranges