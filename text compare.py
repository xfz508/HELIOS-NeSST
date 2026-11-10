from itertools import islice, zip_longest


def find_range_mismatches(file1_path, range1, file2_path, range2, ignore_whitespace=False):
    """
    Compares two line ranges and returns a detailed list of mismatches.

    Ranges are tuples of (start_line, end_line) using 1-based indexing.
    """
    start1, end1 = range1
    start2, end2 = range2

    slice1 = (start1 - 1, end1)
    slice2 = (start2 - 1, end2)

    mismatches = []

    with open(file1_path, 'r') as f1, open(file2_path, 'r') as f2:
        lines1 = islice(f1, *slice1)
        lines2 = islice(f2, *slice2)

        for idx, (l1, l2) in enumerate(zip_longest(lines1, lines2), start=0):
            actual_line1 = start1 + idx
            actual_line2 = start2 + idx

            # Optional whitespace stripping
            val1 = l1.rstrip('\r\n') if l1 is not None else None
            val2 = l2.rstrip('\r\n') if l2 is not None else None

            if ignore_whitespace:
                val1 = val1.strip() if val1 is not None else None
                val2 = val2.strip() if val2 is not None else None

            # 1. Content mismatch
            if val1 != val2:
                if l1 is None:
                    status = "File 1 range ended early"
                elif l2 is None:
                    status = "File 2 range ended early"
                else:
                    status = "Content difference"

                mismatches.append({
                    "relative_index": idx + 1,
                    "status": status,
                    "file1": {"line_num": actual_line1, "content": val1},
                    "file2": {"line_num": actual_line2, "content": val2}
                })

    return mismatches


# --- Example Usage ---
diffs = find_range_mismatches(
    file1_path='.venv/Lib/site-packages/NeSST/data/ENDF/n-002_He_003.endf', range1=(237, 282),
    file2_path='.venv/Lib/site-packages/NeSST/data/ENDF/2-He-003.C31', range2=(142, 238),
    ignore_whitespace=True
)

if not diffs:
    print("Ranges match perfectly!")
else:
    print(f"Found {len(diffs)} mismatch(es):\n")
    for diff in diffs:
        print(f"[Offset +{diff['relative_index']}] {diff['status']}:")
        if diff['file1']['content'] is not None:
            print(f"  File 1 (Line {diff['file1']['line_num']}): {diff['file1']['content']}")
        if diff['file2']['content'] is not None:
            print(f"  File 2 (Line {diff['file2']['line_num']}): {diff['file2']['content']}")
        print("-" * 40)