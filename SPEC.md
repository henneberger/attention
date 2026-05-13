# Attention Extractor Spec

## What
Local attention tool. Show what model looks at.

Intended input: source context + markdown query docs.
Output: `heatmap.tar.gz` containing `heatmap.json`. View with `web/explorer.html`.

Not search. Not proof. Model attention, made visible.

## Inputs
Required:
- `--model`: GGUF file.
- `--context` or `--context-tree`: source context.
- `--query` or `--query-tree`: markdown docs.

Use `--per-file` with `--query-tree`. The legacy single-window path reads `--query` as one file and is only kept for small single-query inputs.

Knobs:
- `--context-glob`, `--query-glob`: file extension filter.
- `--prompt` / `--query-prompt`: override the default task instruction inserted in a `<TASK>` block before `<CONTEXT>`. Passing an empty string disables the task block.
- `--strip-context`, `--strip-query`: cleanup mode.
- `--layers`: explicit layer list, e.g. `14-20,22`.
- `--layer-fraction-start`, `--layer-fraction-end`: layer band by fraction. Default 0.60-0.88.
- `--no-sink-normalization`: disable sink mask.
- `--per-file`: per-file scan. This is the intended mode for markdown query items.
- `--ubatch`: micro-batch size.
- `--ctx-size` / `--ctx`: token context size. In `--per-file` mode, default is automatic and capped at 16384 to avoid oversized Metal KV/cache allocations. In legacy mode, default is the full composed sequence length plus 8.
- `--gpu-layers`: number of model layers to offload. Default 99.
- `--prune-top-k`: JSON score pruning. For each query row, keeps the top K context tokens plus every token on the surrounding +/-5 context lines around each retained token. It also keeps the top K context lines by total attention mass so distributed line-level hits are not pruned away.
- `--prune`: alias for `--prune-top-k`.
- `--no-prune`: disable JSON score pruning.
- `--output`: output path. Default `web/heatmap.tar.gz`.
- `--json`: unsupported; use `--output FILE.tar.gz`.
- `--cache-dir`: token cache. Default `cpp/cache`.
- `--reasoning-steps`: parsed and accepted, but currently reserved and has no effect.

Defaults:
- `--context-glob .ts`
- `--query-glob .mdx`
- `--ubatch 256`
- `--gpu-layers 99`
- `--prune-top-k 80`
- `--prompt "Find implementation evidence in the code for each specification item. Prefer exact implementing functions, branches, data structures, and output fields. Focus on semantic matches, not shared words."`

For C++ context and Markdown query docs, pass `--context-glob .cpp --query-glob .md`.

Strip modes:
- `none` / empty: no change.
- `whitespace`: trim outer whitespace, remove trailing spaces before newlines, and collapse 3+ newlines to 2.
- `python`: remove Python triple-quoted strings and `#` line comments, then apply `whitespace`. This is heuristic cleanup, not a Python parser.

## Units
Context is always model-token level.

With `--per-file`, query rows are markdown items:
- Bullet lines, numbered lines, blockquote lines, fenced code blocks, and text paragraphs each become one query row.
- `--prompt` text is included above `<CONTEXT>` as a `<TASK>` block, but it is outside the query body and does not become a query row.
- Heading hierarchy is retained in the prompt for extra context.
- Parent list items are retained in the prompt for nested items.
- Only the active item's tokens are captured and aggregated into that row.
- Query docs are concatenated into displayed `query_text` with a synthetic `## === path ===` header before each doc. The synthetic header is display structure; it is not itself a query row.

Query chunk labels use `path:item_000000`. Context chunk labels use `path:tok_000000`.
If tokenization produces no token chunks for non-empty text, the fallback label is `path:file`.

Without `--per-file`, the legacy single-window scan still exists for small inputs and uses token-level query rows rather than markdown item rows. Legacy query token labels use `query:tok_000000`.

## Prompt Layout
The decoded sequence is causal, so context must appear before query tokens. With the default prompt, each scan is shaped like:

```xml
<TASK>
Find implementation evidence in the code for each specification item. Prefer exact implementing functions, branches, data structures, and output fields. Focus on semantic matches, not shared words.
</TASK>

<CONTEXT>
...
</CONTEXT>
<QUERY>
...
</QUERY>
```

The task block is decoded before context and query, but `context_text` and `query_text` in JSON contain only displayed corpora, not wrapper tags or task text. The effective task prompt is recorded in `metadata.input_prompt`.

## Flow
With `--per-file`:
1. Build markdown query items once.
2. Tokenize each query-item prompt. The prompt contains heading/list hierarchy plus the active item.
3. Split each context file into token-budgeted windows when needed.
4. For each context window, decode `<TASK>` if present, `<CONTEXT>`, the window contents, `</CONTEXT>`, and `<QUERY>` to fill KV cache.
5. For each markdown query item, remove the old query segment from KV, decode that item prompt, and capture `kq_soft_max-<layer>` for selected layers.
6. Map active item tokens to one query row and context tokens to context columns.
7. Average selected layers and heads.
8. Sink-mask context columns with outlier high column-median unless disabled.
9. In per-file mode, keep absolute context-attention mass per window; do not row-normalize across only that window.
10. Append window columns to global JSON state. Write partial output after each processed context file, then write final output.

Query token `i` reads attention from row `i - 1`. Causal next-token shift.

## Score Contract
Per-file markdown mode:
```text
scores[query_item][context_token]
```

`scores.length == query_chunks.length`.
`scores[i].length == context_chunks.length`.

Legacy single-window mode:
```text
scores[query_token][context_token]
```

High = looked more. Zero = no captured or mapped attention.

Per-file scores are not globally row-normalized after each window; this preserves absolute mass so sparse windows do not become 100%. Legacy single-window scores are row-normalized, then adjusted by an attention prior across rows.

## JSON Contract
`heatmap.json`:
- `context_text`, `query_text`: displayed corpora.
- `context_chunks`: context token chunks.
- `query_chunks`: markdown item chunks in per-file mode; token chunks in legacy single-window mode.
- `scores`: matrix.
- `head_scores`, `layer_scores`: currently empty arrays.
- `metadata`: backend, model path, mode flags, selected layers, and mode-specific fields.

Per-file metadata includes `per_file`, `query_items_count`, `query_chunk_mode: "markdown_items"`, `files_total`, `files_done`, `score_prune_top_k`, `input_prompt`, and `layers`.
Legacy metadata includes `score_prune_top_k`, `sink_normalization`, `flash_attn`, `input_prompt`, and `layers`.

Context chunk:
```json
{ "label": "path:tok_000000", "start": 0, "end": 10, "text": "..." }
```

Query chunk:
```json
{ "label": "path:item_000000", "start": 0, "end": 20, "text": "- requirement" }
```

Legacy query token chunk:
```json
{ "label": "query:tok_000000", "start": 0, "end": 10, "text": "..." }
```

Offsets are char offsets in displayed text.

Displayed `query_text` in per-file mode includes synthetic markdown doc headers. Displayed `context_text` includes synthetic `// === path ===` headers, and split context windows are labeled with `window N/M` in those headers. Wrapper tags and the task prompt are not included in displayed text.

## Scan Mode
Per-file is the intended mode for markdown query items: each context file is split into token-budgeted windows and scanned against all markdown query items. Lower memory. A partial `heatmap.tar.gz` is written after each context file with at least one processed window.

Large context windows can be skipped if they still do not fit beside the largest query-item prompt. Empty context files are skipped.

On Metal, out-of-memory failures usually mean the context size or offload is too high for the selected model and query set. Retry with a smaller context, smaller micro-batch, or partial CPU offload, for example `--ctx-size 8192 --ubatch 128 --gpu-layers 40`.

## Token Cache
Per-content SHA-256 key. Binary file: magic `TOKN`, version, n, tokens, char_start, char_end. Speed only for the same model/tokenizer. The cache key does not include model identity; clear or separate `--cache-dir` when changing models.

## Viewer
`web/explorer.html` loads `web/heatmap.tar.gz` by default. Drop or pick another `.tar.gz`.

Shows query docs first, then source tree. On load it selects the first query doc. Query docs render as full markdown with line numbers and highlighted query items. Source files render token heat with line numbers. Clicking a highlighted query item opens an inline results pane with ranked source spans by summed attention mass. Clicking a highlighted source token shows contributing query chunks.

## Code Map
All in `main.cpp`:
- Markdown query items: `build_markdown_query_set`, `MarkdownQueryItem`, `MarkdownQuerySet`.
- Context token chunks: `token_chunks_from_offsets`.
- Token cache: `tokenize_with_cache`, `load_token_cache`, `save_token_cache`.
- Capture: `eval_callback`.
- Aggregation: `aggregate_captured`.
- Per-file scan: `run_per_file_scan`.
