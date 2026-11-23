# Fault localization from issue text

Evaluated on 129 public tasks; 128 have a reference fix that edits an existing file, 1 only create new files and are excluded below.
Median candidate pool: 506 non-test Python files per task.
Regenerate with `gemma-swe localize`.

## File level: is a file the reference fix edits in the top k?

| Method | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Recall@5 | Recall@10 | MRR |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Identifier grep (manual `git grep` strategy) | 31.2% | 45.3% | 50.8% | 57.0% | 44.6% | 51.5% | 0.4 |
| BM25, one document per file | 28.1% | 44.5% | 55.5% | 71.9% | 46.7% | 64.0% | 0.411 |
| Chunk BM25 (ablation: no directory prior) | 43.0% | 57.0% | 64.8% | 75.8% | 56.2% | 68.4% | 0.523 |
| `repo-locate` skill as shipped (chunk BM25 + directory prior) | 48.4% | 64.1% | 75.0% | 83.6% | 65.1% | 77.0% | 0.592 |

Hit@k: at least one gold file in the top k. Recall@k: share of gold files in the top k.

## Function level

For the 128 tasks whose fix modifies existing lines: does one of the three line ranges the skill prints for a top-k file overlap a line the reference fix changes?

| Top-k files | Hit rate |
| :--- | ---: |
| 1 | 44.5% |
| 3 | 59.4% |
| 5 | 70.3% |

## By repository (file level)

| Repository | Tasks | Method | Hit@1 | Hit@5 |
| :--- | ---: | :--- | ---: | ---: |
| Textualize/rich | 48 | ident_grep | 12.5% | 35.4% |
| Textualize/rich | 48 | bm25_file | 16.7% | 50.0% |
| Textualize/rich | 48 | locate_no_prior | 31.2% | 54.2% |
| Textualize/rich | 48 | locate | 33.3% | 58.3% |
| encode/httpx | 1 | ident_grep | 100.0% | 100.0% |
| encode/httpx | 1 | bm25_file | 100.0% | 100.0% |
| encode/httpx | 1 | locate_no_prior | 100.0% | 100.0% |
| encode/httpx | 1 | locate | 100.0% | 100.0% |
| fastapi/fastapi | 66 | ident_grep | 39.4% | 54.5% |
| fastapi/fastapi | 66 | bm25_file | 28.8% | 51.5% |
| fastapi/fastapi | 66 | locate_no_prior | 40.9% | 65.2% |
| fastapi/fastapi | 66 | locate | 50.0% | 81.8% |
| psf/requests | 13 | ident_grep | 53.8% | 84.6% |
| psf/requests | 13 | bm25_file | 61.5% | 92.3% |
| psf/requests | 13 | locate_no_prior | 92.3% | 100.0% |
| psf/requests | 13 | locate | 92.3% | 100.0% |
