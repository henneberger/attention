#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <omp.h>

#include <CommonCrypto/CommonDigest.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <zlib.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

// -----------------------------------------------------------------------------
// Types and constants

struct Chunk {
    std::string label;
    int start;
    int end;
    std::string text;
};

struct CapturedAttention {
    int layer;
    int n_heads;
    int n_query;
    int n_context;
    std::vector<float> weights; // shape [n_heads, n_query, n_context], row-major
};

struct CaptureContext {
    std::vector<int> context_token_positions; // positions in full sequence
    std::vector<int> query_token_positions;   // causal-shifted, full-sequence positions
    int batch_query_pos_start = 0;            // start position (in full sequence) of current batch
    int batch_query_pos_end = 0;              // exclusive
    std::set<int> selected_layers;
    // accumulation: per-layer captured attention (filled across batches)
    std::unordered_map<int, CapturedAttention> per_layer;
};

static const char * kContextOpen = "<CONTEXT>\n";
static const char * kMiddle = "\n</CONTEXT>\n<QUERY>\n";
static const char * kQueryClose = "\n</QUERY>";

struct TokenizedDoc {
    std::vector<llama_token> tokens;
    std::vector<int> char_start;
    std::vector<int> char_end;
};
static TokenizedDoc tokenize_with_offsets(const llama_vocab * vocab, const std::string & text);

// -----------------------------------------------------------------------------
// Logging

static void log_line(const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    std::fprintf(stderr, "[attn-extract] ");
    std::vfprintf(stderr, fmt, ap);
    std::fprintf(stderr, "\n");
    va_end(ap);
}

// -----------------------------------------------------------------------------
// File I/O

static std::string read_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "failed to open %s\n", path.c_str());
        std::exit(1);
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

static void write_file(const std::string & path, const std::string & content) {
    std::ofstream out(path, std::ios::binary);
    out << content;
}

static bool path_ends_with(const std::string & s, const std::string & suffix) {
    if (suffix.size() > s.size()) return false;
    return std::equal(suffix.rbegin(), suffix.rend(), s.rbegin());
}

static void append_tar_octal(char * dst, size_t width, uint64_t value) {
    std::snprintf(dst, width, "%0*llo", (int) width - 1, (unsigned long long) value);
}

static void write_tar_gz_json(const std::string & path, const std::string & json) {
    gzFile gz = gzopen(path.c_str(), "wb9");
    if (!gz) {
        log_line("failed to open gzip output %s", path.c_str());
        std::exit(1);
    }

    char header[512];
    std::memset(header, 0, sizeof(header));
    const char * name = "heatmap.json";
    std::memcpy(header, name, std::strlen(name));
    append_tar_octal(header + 100, 8, 0644);
    append_tar_octal(header + 108, 8, 0);
    append_tar_octal(header + 116, 8, 0);
    append_tar_octal(header + 124, 12, (uint64_t) json.size());
    append_tar_octal(header + 136, 12, 0);
    std::memset(header + 148, ' ', 8);
    header[156] = '0';
    std::memcpy(header + 257, "ustar", 5);
    std::memcpy(header + 263, "00", 2);

    unsigned int checksum = 0;
    for (unsigned char c : header) checksum += c;
    std::snprintf(header + 148, 8, "%06o", checksum);
    header[154] = '\0';
    header[155] = ' ';

    if (gzwrite(gz, header, sizeof(header)) != (int) sizeof(header)) {
        log_line("failed to write tar header to %s", path.c_str());
        gzclose(gz);
        std::exit(1);
    }
    if (!json.empty() && gzwrite(gz, json.data(), (unsigned int) json.size()) != (int) json.size()) {
        log_line("failed to write json payload to %s", path.c_str());
        gzclose(gz);
        std::exit(1);
    }

    size_t padding = (512 - (json.size() % 512)) % 512;
    if (padding > 0) {
        char zeros[512] = {0};
        if (gzwrite(gz, zeros, (unsigned int) padding) != (int) padding) {
            log_line("failed to write tar padding to %s", path.c_str());
            gzclose(gz);
            std::exit(1);
        }
    }
    char end_blocks[1024] = {0};
    if (gzwrite(gz, end_blocks, sizeof(end_blocks)) != (int) sizeof(end_blocks)) {
        log_line("failed to write tar footer to %s", path.c_str());
        gzclose(gz);
        std::exit(1);
    }
    if (gzclose(gz) != Z_OK) {
        log_line("failed to close gzip output %s", path.c_str());
        std::exit(1);
    }
}

static void write_heatmap_output(const std::string & path, const std::string & json) {
    if (path_ends_with(path, ".tar.gz") || path_ends_with(path, ".tgz")) {
        write_tar_gz_json(path, json);
        return;
    }
    log_line("output path must end in .tar.gz or .tgz: %s", path.c_str());
    std::exit(1);
}

// -----------------------------------------------------------------------------
// Text preprocessing

static std::string collapse_whitespace(const std::string & text) {
    std::string s = std::regex_replace(text, std::regex("[ \\t]+\\n"), "\n");
    s = std::regex_replace(s, std::regex("\\n{3,}"), "\n\n");
    // trim
    size_t start = s.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) return "\n";
    size_t end = s.find_last_not_of(" \t\n\r");
    return s.substr(start, end - start + 1) + "\n";
}

static std::string strip_python(const std::string & text) {
    // Triple-quoted strings (greedy across newlines).
    std::string s = std::regex_replace(text, std::regex("\"\"\"[\\s\\S]*?\"\"\""), "");
    s = std::regex_replace(s, std::regex("'''[\\s\\S]*?'''"), "");
    // Line comments. Naive — strips `#` inside string literals too, acceptable for attribution.
    s = std::regex_replace(s, std::regex("(?:^|[\\t ])#.*$", std::regex::multiline), "");
    return collapse_whitespace(s);
}

static std::string strip_text(const std::string & text, const std::string & mode) {
    if (mode == "none" || mode.empty()) return text;
    if (mode == "whitespace") return collapse_whitespace(text);
    if (mode == "python") return strip_python(text);
    log_line("unknown strip mode: %s", mode.c_str());
    std::exit(1);
}

// -----------------------------------------------------------------------------
// SHA-256, directory walking, token cache

static std::string sha256_hex(const std::string & data) {
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(data.data(), (CC_LONG) data.size(), digest);
    char buf[CC_SHA256_DIGEST_LENGTH * 2 + 1];
    for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; i++) {
        snprintf(buf + i * 2, 3, "%02x", digest[i]);
    }
    return std::string(buf, CC_SHA256_DIGEST_LENGTH * 2);
}

static bool ends_with(const std::string & s, const std::string & suffix) {
    if (suffix.size() > s.size()) return false;
    return std::equal(suffix.rbegin(), suffix.rend(), s.rbegin());
}

static void walk_directory(
    const std::string & root,
    const std::string & extension,
    std::vector<std::string> & out
) {
    DIR * d = opendir(root.c_str());
    if (!d) return;
    struct dirent * entry;
    std::vector<std::string> subdirs;
    while ((entry = readdir(d))) {
        std::string name = entry->d_name;
        if (name == "." || name == "..") continue;
        std::string path = root + "/" + name;
        struct stat st;
        if (stat(path.c_str(), &st) != 0) continue;
        if (S_ISDIR(st.st_mode)) {
            // Skip common large/uninteresting dirs.
            if (name == "node_modules" || name == ".git" || name == "dist" || name == "build") continue;
            subdirs.push_back(path);
        } else if (S_ISREG(st.st_mode) && ends_with(name, extension)) {
            out.push_back(path);
        }
    }
    closedir(d);
    std::sort(subdirs.begin(), subdirs.end());
    for (auto & sd : subdirs) walk_directory(sd, extension, out);
}

static std::string ensure_dir(const std::string & path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) {
        mkdir(path.c_str(), 0755);
    }
    return path;
}

// On-disk format for cached file tokens:
//   magic "TOKN" (4) | version (u32) | n_tokens (u32) |
//   tokens (i32 * n_tokens) | char_start (i32 * n_tokens) | char_end (i32 * n_tokens)
static constexpr uint32_t kTokenCacheMagic = 0x544f4b4eU;
static constexpr uint32_t kTokenCacheVersion = 1;

struct CachedTokens {
    std::vector<llama_token> tokens;
    std::vector<int> char_start;
    std::vector<int> char_end;
};

static std::vector<Chunk> token_chunks_from_offsets(
    const std::string & text,
    const CachedTokens & ct,
    const std::string & prefix
) {
    std::vector<Chunk> chunks;
    chunks.reserve(ct.tokens.size());
    for (size_t i = 0; i < ct.tokens.size(); i++) {
        int start = ct.char_start[i];
        int end = ct.char_end[i];
        if (start < 0) start = 0;
        if (end < start) end = start;
        if (start > (int) text.size()) start = (int) text.size();
        if (end > (int) text.size()) end = (int) text.size();
        if (end <= start) continue;
        std::string token_text = text.substr(start, end - start);
        if (token_text.empty()) continue;
        char label[64];
        std::snprintf(label, sizeof(label), ":tok_%06zu", chunks.size());
        chunks.push_back({prefix + label, start, end, token_text});
    }
    return chunks;
}

struct MarkdownQueryItem {
    std::string rel_path;
    std::string label;
    std::string prompt_text;
    std::string text;
    int active_start;  // char offsets in prompt_text
    int active_end;
    int display_start; // char offsets in displayed query_text
    int display_end;
    CachedTokens ct;   // prompt_text + kQueryClose
};

struct MarkdownQuerySet {
    std::string text;
    std::vector<Chunk> chunks;
    std::vector<MarkdownQueryItem> items;
};

static int markdown_indent_width(const std::string & line) {
    int n = 0;
    for (char c : line) {
        if (c == ' ') n++;
        else if (c == '\t') n += 4;
        else break;
    }
    return n;
}

static bool markdown_fence_line(const std::string & line) {
    size_t i = 0;
    while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) i++;
    return line.compare(i, 3, "```") == 0 || line.compare(i, 3, "~~~") == 0;
}

static MarkdownQuerySet build_markdown_query_set(
    const std::vector<std::string> & paths,
    const std::string & root,
    const std::string & strip_mode
) {
    MarkdownQuerySet out;
    std::regex heading_re(R"(^\s{0,3}(#{1,6})\s+(.+)$)");
    std::regex list_item_re(R"(^(\s*)(?:[-*+]\s+|\d+[.)]\s+).+\S.*$)");
    std::regex block_item_re(R"(^\s{0,3}>\s*\S.*$)");

    struct Ancestor {
        int indent;
        std::string line;
    };

    for (auto & p : paths) {
        std::string rel = root.empty()
            ? p
            : p.substr(root.size() + (root.back() == '/' ? 0 : 1));
        std::string content = strip_text(read_file(p), strip_mode);
        if (content.empty() || content == "\n") continue;

        std::string header = "\n## === " + rel + " ===\n";
        out.text += header;
        int doc_offset = (int) out.text.size();
        out.text += content;

        std::vector<std::string> headings;
        std::vector<Ancestor> list_stack;
        auto is_blank = [](const std::string & s) {
            return s.find_first_not_of(" \t\r") == std::string::npos;
        };
        auto emit_item = [&](const std::string & text, size_t item_start, size_t item_end) {
            std::vector<std::string> prompt_lines;
            prompt_lines.insert(prompt_lines.end(), headings.begin(), headings.end());
            for (auto & a : list_stack) prompt_lines.push_back(a.line);
            int active_start = 0;
            for (auto & pl : prompt_lines) active_start += (int) pl.size() + 1;

            std::string prompt;
            for (auto & pl : prompt_lines) {
                prompt += pl;
                prompt += '\n';
            }
            prompt += text;
            prompt += '\n';

            char suffix[32];
            std::snprintf(suffix, sizeof(suffix), ":item_%06zu", out.items.size());
            std::string label = rel + suffix;
            int display_start = doc_offset + (int) item_start;
            int display_end = doc_offset + (int) item_end;

            MarkdownQueryItem item;
            item.rel_path = rel;
            item.label = label;
            item.prompt_text = prompt;
            item.text = text;
            item.active_start = active_start;
            item.active_end = active_start + (int) text.size();
            item.display_start = display_start;
            item.display_end = display_end;
            out.items.push_back(std::move(item));
            out.chunks.push_back({label, display_start, display_end, text});
        };

        size_t cursor = 0;
        while (cursor < content.size()) {
            size_t line_start = cursor;
            size_t line_end = content.find('\n', cursor);
            if (line_end == std::string::npos) line_end = content.size();
            else line_end += 1;
            size_t content_end = line_end;
            while (content_end > line_start && (content[content_end - 1] == '\n' || content[content_end - 1] == '\r')) {
                content_end--;
            }

            std::string line = content.substr(line_start, content_end - line_start);
            if (markdown_fence_line(line)) {
                size_t block_start = line_start;
                size_t block_content_end = content_end;
                size_t next = line_end;
                while (next < content.size()) {
                    size_t inner_start = next;
                    size_t inner_end = content.find('\n', next);
                    if (inner_end == std::string::npos) inner_end = content.size();
                    else inner_end += 1;
                    size_t inner_content_end = inner_end;
                    while (inner_content_end > inner_start &&
                           (content[inner_content_end - 1] == '\n' || content[inner_content_end - 1] == '\r')) {
                        inner_content_end--;
                    }
                    std::string inner_line = content.substr(inner_start, inner_content_end - inner_start);
                    block_content_end = inner_content_end;
                    next = inner_end;
                    if (markdown_fence_line(inner_line)) break;
                }
                emit_item(content.substr(block_start, block_content_end - block_start),
                          block_start, block_content_end);
                cursor = next;
                continue;
            }
            if (is_blank(line)) {
                list_stack.clear();
                cursor = line_end;
                continue;
            }

            std::smatch m;
            if (std::regex_match(line, m, heading_re)) {
                int level = (int) m[1].str().size();
                if ((int) headings.size() >= level) headings.resize(level - 1);
                headings.push_back(line);
                list_stack.clear();
                cursor = line_end;
                continue;
            }

            bool is_list_item = std::regex_match(line, m, list_item_re);
            bool is_block_item = std::regex_match(line, block_item_re);
            if (is_list_item || is_block_item) {
                int indent = is_list_item ? markdown_indent_width(m[1].str()) : 0;
                while (!list_stack.empty() && list_stack.back().indent >= indent) {
                    list_stack.pop_back();
                }

                emit_item(line, line_start, content_end);
                list_stack.push_back({indent, line});
                cursor = line_end;
                continue;
            }

            size_t paragraph_start = line_start;
            size_t paragraph_content_end = content_end;
            size_t next = line_end;
            while (next < content.size()) {
                size_t para_line_start = next;
                size_t para_line_end = content.find('\n', next);
                if (para_line_end == std::string::npos) para_line_end = content.size();
                else para_line_end += 1;
                size_t para_content_end = para_line_end;
                while (para_content_end > para_line_start &&
                       (content[para_content_end - 1] == '\n' || content[para_content_end - 1] == '\r')) {
                    para_content_end--;
                }
                std::string para_line = content.substr(para_line_start, para_content_end - para_line_start);
                std::smatch para_match;
                if (is_blank(para_line) ||
                    markdown_fence_line(para_line) ||
                    std::regex_match(para_line, para_match, heading_re) ||
                    std::regex_match(para_line, para_match, list_item_re) ||
                    std::regex_match(para_line, para_match, block_item_re)) {
                    break;
                }
                paragraph_content_end = para_content_end;
                next = para_line_end;
            }

            emit_item(content.substr(paragraph_start, paragraph_content_end - paragraph_start),
                      paragraph_start, paragraph_content_end);
            list_stack.clear();
            cursor = next;
        }
    }

    return out;
}

static bool load_token_cache(const std::string & path, CachedTokens & out) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    uint32_t magic = 0, version = 0, n = 0;
    if (std::fread(&magic, 4, 1, f) != 1 || magic != kTokenCacheMagic) { std::fclose(f); return false; }
    if (std::fread(&version, 4, 1, f) != 1 || version != kTokenCacheVersion) { std::fclose(f); return false; }
    if (std::fread(&n, 4, 1, f) != 1) { std::fclose(f); return false; }
    out.tokens.assign(n, 0);
    out.char_start.assign(n, 0);
    out.char_end.assign(n, 0);
    if (std::fread(out.tokens.data(), sizeof(llama_token), n, f) != n) { std::fclose(f); return false; }
    if (std::fread(out.char_start.data(), sizeof(int), n, f) != n) { std::fclose(f); return false; }
    if (std::fread(out.char_end.data(), sizeof(int), n, f) != n) { std::fclose(f); return false; }
    std::fclose(f);
    return true;
}

static void save_token_cache(const std::string & path, const CachedTokens & ct) {
    FILE * f = std::fopen(path.c_str(), "wb");
    if (!f) { log_line("warning: cannot write cache %s", path.c_str()); return; }
    uint32_t magic = kTokenCacheMagic, version = kTokenCacheVersion;
    uint32_t n = (uint32_t) ct.tokens.size();
    std::fwrite(&magic, 4, 1, f);
    std::fwrite(&version, 4, 1, f);
    std::fwrite(&n, 4, 1, f);
    std::fwrite(ct.tokens.data(), sizeof(llama_token), n, f);
    std::fwrite(ct.char_start.data(), sizeof(int), n, f);
    std::fwrite(ct.char_end.data(), sizeof(int), n, f);
    std::fclose(f);
}

static CachedTokens tokenize_with_cache(
    const llama_vocab * vocab,
    const std::string & text,
    const std::string & cache_path
) {
    CachedTokens ct;
    if (!cache_path.empty() && load_token_cache(cache_path, ct)) {
        return ct;
    }
    auto doc = tokenize_with_offsets(vocab, text);
    ct.tokens = doc.tokens;
    ct.char_start = doc.char_start;
    ct.char_end = doc.char_end;
    if (!cache_path.empty()) save_token_cache(cache_path, ct);
    return ct;
}

// -----------------------------------------------------------------------------
// Multi-file corpus composer
//
// Concatenates many source files into one "context corpus" inside the same
// <CONTEXT>...</CONTEXT><QUERY>...</QUERY> wrapper. Each file's tokens come
// from the per-file cache (so adding/removing files only re-tokenizes deltas).
// Chunks are computed per-file and shifted into corpus coordinates.

struct ContextFile {
    std::string path;
    std::string rel_path;   // relative to source tree root, used for chunk labels
    std::string content;
    int corpus_offset;      // char position of this file's content in corpus
    std::vector<Chunk> chunks; // file-local, will be shifted to corpus coords on aggregation
};

struct ComposedInput {
    std::vector<llama_token> tokens;
    std::vector<int> char_start; // per token, absolute in composed text
    std::vector<int> char_end;
    int context_body_start;
    int context_body_end;
    int query_body_start;
    int query_body_end;
    std::vector<Chunk> context_chunks_corpus; // chunks in corpus coordinates
};

static std::string make_file_header(const std::string & rel_path) {
    return "\n// === " + rel_path + " ===\n";
}

// -----------------------------------------------------------------------------
// Tokenization with char offsets

static TokenizedDoc tokenize_with_offsets(
    const llama_vocab * vocab,
    const std::string & text
) {
    TokenizedDoc out;

    // First, tokenize without special tokens. add_special=false, parse_special=false.
    std::vector<llama_token> tokens(text.size() + 16);
    int n = llama_tokenize(
        vocab,
        text.data(), (int32_t) text.size(),
        tokens.data(), (int32_t) tokens.size(),
        /*add_special=*/false,
        /*parse_special=*/false
    );
    if (n < 0) {
        tokens.resize(-n);
        n = llama_tokenize(
            vocab,
            text.data(), (int32_t) text.size(),
            tokens.data(), (int32_t) tokens.size(),
            false, false
        );
    }
    tokens.resize(n);
    out.tokens = tokens;
    out.char_start.reserve(n);
    out.char_end.reserve(n);

    size_t cursor = 0;
    for (auto token : tokens) {
        char buf[256];
        int piece_n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, false);
        if (piece_n < 0) {
            // try larger buffer
            std::vector<char> big(1024);
            piece_n = llama_token_to_piece(vocab, token, big.data(), (int32_t) big.size(), 0, false);
            if (piece_n < 0) {
                out.char_start.push_back((int) cursor);
                out.char_end.push_back((int) cursor);
                continue;
            }
            std::string piece(big.data(), piece_n);
            size_t pos = text.find(piece, cursor);
            if (pos == std::string::npos) pos = cursor;
            out.char_start.push_back((int) pos);
            out.char_end.push_back((int) (pos + piece.size()));
            cursor = pos + piece.size();
            continue;
        }
        std::string piece(buf, piece_n);
        size_t pos = text.find(piece, cursor);
        if (pos == std::string::npos) {
            // Mismatch (e.g., special-token piece). Best-effort: keep cursor.
            out.char_start.push_back((int) cursor);
            out.char_end.push_back((int) cursor);
            continue;
        }
        out.char_start.push_back((int) pos);
        out.char_end.push_back((int) (pos + piece.size()));
        cursor = pos + piece.size();
    }

    return out;
}

// -----------------------------------------------------------------------------
// Chunk → token-index map

static std::vector<int> token_indices_for_range(
    const TokenizedDoc & doc,
    int region_start,
    int region_end
) {
    std::vector<int> out;
    for (int i = 0; i < (int) doc.tokens.size(); i++) {
        int ts = doc.char_start[i];
        int te = doc.char_end[i];
        if (ts == te) continue;
        if (te > region_start && ts < region_end) out.push_back(i);
    }
    return out;
}

static std::vector<int> token_to_chunk_map(
    const TokenizedDoc & doc,
    const std::vector<Chunk> & chunks
) {
    // For each token, the index of the chunk that contains it, or -1.
    std::vector<int> mapping(doc.tokens.size(), -1);

    // Build a sorted index by chunk.start.
    std::vector<int> order(chunks.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return chunks[a].start < chunks[b].start;
    });

    size_t cursor = 0; // index into order
    for (size_t i = 0; i < doc.tokens.size(); i++) {
        int ts = doc.char_start[i];
        int te = doc.char_end[i];
        if (ts == te) continue;
        while (cursor < order.size() && chunks[order[cursor]].end <= ts) cursor++;
        if (cursor >= order.size()) break;
        const auto & c = chunks[order[cursor]];
        if (c.start < te && c.end > ts) {
            mapping[i] = order[cursor];
        }
    }
    return mapping;
}

// -----------------------------------------------------------------------------
// Attention capture callback

struct CallbackUserData {
    CaptureContext * cap;
};

static bool eval_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * ud = static_cast<CallbackUserData *>(user_data);
    auto & cap = *ud->cap;

    const char * name = t->name;
    if (!name) return false;

    // Filter `kq_soft_max-<il>`.
    static const std::regex re("^kq_soft_max-(\\d+)$");
    std::cmatch m;
    if (!std::regex_match(name, m, re)) return false;

    int layer = std::atoi(m[1].first);
    if (cap.selected_layers.count(layer) == 0) return false;

    if (ask) return true;

    // Tensor shape: ne0 = n_kv (context size in cache), ne1 = n_tokens (this batch),
    // ne2 = n_head, ne3 = 1. Memory: ne0 fastest-varying.
    int64_t n_kv = t->ne[0];
    int64_t n_tokens = t->ne[1];
    int64_t n_heads = t->ne[2];

    size_t bytes = ggml_nbytes(t);
    std::vector<float> host(bytes / sizeof(float));
    ggml_backend_tensor_get(t, host.data(), 0, bytes);

    if (t->type != GGML_TYPE_F32) {
        // We pass F32 prec on softmax; assume F32. If not, would need a conversion.
        log_line("warning: tensor %s is not f32 (type=%d)", name, t->type);
    }

    // For this batch, the global token positions of the rows are
    // [batch_query_pos_start .. batch_query_pos_end).
    // We need rows corresponding to causal-shifted query positions in
    // cap.query_token_positions whose value is in [batch_query_pos_start, batch_query_pos_end).
    // Columns: keep only the context positions in cap.context_token_positions
    // (these are <= cache extent so always present in n_kv).

    auto & per_layer = cap.per_layer[layer];
    if (per_layer.layer == 0 && per_layer.weights.empty()) {
        // Initialize.
        per_layer.layer = layer;
        per_layer.n_heads = (int) n_heads;
        per_layer.n_query = (int) cap.query_token_positions.size();
        per_layer.n_context = (int) cap.context_token_positions.size();
        per_layer.weights.assign((size_t) n_heads * per_layer.n_query * per_layer.n_context, 0.0f);
    }

    // Map context token positions → column indices in this tensor.
    // Cols correspond to positions [0, n_kv). cap.context_token_positions are
    // all < cap.batch_query_pos_start (since context precedes query).
    // For each context pos, col = position.
    std::vector<int> ctx_cols;
    ctx_cols.reserve(cap.context_token_positions.size());
    for (int p : cap.context_token_positions) {
        if (p >= 0 && p < n_kv) ctx_cols.push_back(p);
        else ctx_cols.push_back(-1); // shouldn't happen for context
    }

    // For each row of this batch (q_local in [0, n_tokens)), its global position
    // is cap.batch_query_pos_start + q_local. If that global position is in our
    // query_token_positions, store.
    std::unordered_map<int, int> qpos_to_local;
    qpos_to_local.reserve(cap.query_token_positions.size());
    for (int i = 0; i < (int) cap.query_token_positions.size(); i++) {
        qpos_to_local[cap.query_token_positions[i]] = i;
    }

    for (int64_t q_local = 0; q_local < n_tokens; q_local++) {
        int global_pos = cap.batch_query_pos_start + (int) q_local;
        auto it = qpos_to_local.find(global_pos);
        if (it == qpos_to_local.end()) continue;
        int row_out = it->second;
        for (int h = 0; h < (int) n_heads; h++) {
            // offset in host: h * n_tokens * n_kv + q_local * n_kv + ...
            size_t base = (size_t) h * n_tokens * n_kv + (size_t) q_local * n_kv;
            float * row_data = host.data() + base;
            for (int c = 0; c < (int) ctx_cols.size(); c++) {
                int col = ctx_cols[c];
                if (col < 0) continue;
                size_t out_idx = (size_t) h * per_layer.n_query * per_layer.n_context
                                + (size_t) row_out * per_layer.n_context + c;
                per_layer.weights[out_idx] = row_data[col];
            }
        }
    }

    return true;
}

// -----------------------------------------------------------------------------
// Aggregation

static void normalize_row(std::vector<float> & row) {
    double sum = 0.0;
    for (float v : row) sum += v;
    if (sum <= 0) {
        std::fill(row.begin(), row.end(), 0.0f);
        return;
    }
    for (auto & v : row) v = (float)(v / sum);
}

static std::vector<std::vector<float>> apply_sink_norm(
    const std::vector<std::vector<float>> & matrix
) {
    // Mask context-key columns with unusually high column-median attention.
    // Preserve absolute mass; windowed scans should not inflate the remaining
    // visible context columns into a full probability distribution.
    if (matrix.empty()) return matrix;
    size_t rows = matrix.size();
    size_t cols = matrix[0].size();
    if (cols == 0) return matrix;

    std::vector<float> col_median(cols, 0.0f);
    for (size_t c = 0; c < cols; c++) {
        std::vector<float> column(rows);
        for (size_t r = 0; r < rows; r++) column[r] = matrix[r][c];
        std::sort(column.begin(), column.end());
        col_median[c] = column[rows / 2];
    }
    double min_positive = 1e-12;
    for (float v : col_median) if (v > 0 && v < min_positive) min_positive = v;
    std::vector<float> logs(cols);
    for (size_t c = 0; c < cols; c++) logs[c] = std::log(col_median[c] + (float)(min_positive / 2.0 + 1e-12));
    double mean = 0;
    for (float v : logs) mean += v;
    mean /= cols;
    double var = 0;
    for (float v : logs) { double d = v - mean; var += d * d; }
    var /= cols;
    double threshold = mean + 3.0 * std::sqrt(var);
    std::vector<bool> sink(cols, false);
    bool any = false;
    for (size_t c = 0; c < cols; c++) if (logs[c] > threshold) { sink[c] = true; any = true; }
    if (!any) return matrix;
    std::vector<std::vector<float>> out(rows, std::vector<float>(cols, 0.0f));
    for (size_t r = 0; r < rows; r++) {
        for (size_t c = 0; c < cols; c++) out[r][c] = sink[c] ? 0.0f : matrix[r][c];
    }
    return out;
}

static std::vector<std::vector<float>> prior_normalize(
    const std::vector<std::vector<float>> & scores,
    double power = 0.75,
    int min_rows = 3
) {
    if ((int) scores.size() < min_rows) return scores;
    size_t cols = 0;
    for (auto & row : scores) cols = std::max(cols, row.size());
    if (cols == 0) return scores;
    std::vector<double> priors(cols, 0.0);
    std::vector<int> counts(cols, 0);
    for (auto & row : scores) {
        for (size_t c = 0; c < row.size(); c++) {
            float v = row[c];
            if (std::isfinite(v) && v > 0) {
                priors[c] += v;
                counts[c] += 1;
            }
        }
    }
    for (size_t c = 0; c < cols; c++) priors[c] = counts[c] > 0 ? priors[c] / counts[c] : 0.0;

    std::vector<std::vector<float>> out(scores.size(), std::vector<float>(cols, 0.0f));
    for (size_t r = 0; r < scores.size(); r++) {
        std::vector<float> adjusted(cols, 0.0f);
        for (size_t c = 0; c < scores[r].size(); c++) {
            float v = scores[r][c];
            if (std::isfinite(v) && v > 0 && c < cols && priors[c] > 0) {
                adjusted[c] = (float)(v / std::pow(priors[c], power));
            }
        }
        normalize_row(adjusted);
        out[r] = adjusted;
    }
    return out;
}

static std::vector<std::vector<float>> aggregate_captured(
    const std::vector<CapturedAttention> & per_layer_captures,
    const std::vector<int> & query_local_to_chunk,
    const std::vector<int> & context_local_to_chunk,
    int num_query_chunks,
    int num_context_chunks,
    bool sink_normalization,
    bool global_normalize = true
) {
    // Average all selected layers + heads first.
    if (per_layer_captures.empty()) {
        return std::vector<std::vector<float>>(num_query_chunks, std::vector<float>(num_context_chunks, 0.0f));
    }
    int n_q = per_layer_captures[0].n_query;
    int n_c = per_layer_captures[0].n_context;
    int n_heads = per_layer_captures[0].n_heads;
    std::vector<float> avg((size_t) n_q * n_c, 0.0f);
    double denom = (double) per_layer_captures.size() * n_heads;
    // Parallelize over query rows — each thread owns its slice of `avg`.
    #pragma omp parallel for schedule(static)
    for (int q = 0; q < n_q; q++) {
        size_t out_base = (size_t) q * n_c;
        for (auto & cap : per_layer_captures) {
            for (int h = 0; h < n_heads; h++) {
                size_t in_base = (size_t) h * n_q * n_c + (size_t) q * n_c;
                for (int c = 0; c < n_c; c++) {
                    avg[out_base + c] += cap.weights[in_base + c];
                }
            }
        }
        float inv = (float)(1.0 / denom);
        for (int c = 0; c < n_c; c++) avg[out_base + c] *= inv;
    }

    // Per query chunk: select its rows, sink-norm, mean down to per-token,
    // then scatter total attention mass to context chunks. Do not renormalize
    // over only the visible context columns here; per-window scans need the
    // absolute context-attention mass so sparse windows do not become 100%.
    std::vector<std::vector<float>> result(num_query_chunks, std::vector<float>(num_context_chunks, 0.0f));

    #pragma omp parallel for schedule(dynamic)
    for (int q_chunk = 0; q_chunk < num_query_chunks; q_chunk++) {
        // collect local query row indices
        std::vector<int> rows;
        for (int i = 0; i < (int) query_local_to_chunk.size(); i++) {
            if (query_local_to_chunk[i] == q_chunk) rows.push_back(i);
        }
        if (rows.empty()) continue;

        std::vector<std::vector<float>> sub(rows.size(), std::vector<float>(n_c, 0.0f));
        for (size_t i = 0; i < rows.size(); i++) {
            size_t base = (size_t) rows[i] * n_c;
            for (int c = 0; c < n_c; c++) sub[i][c] = avg[base + c];
        }

        if (sink_normalization) sub = apply_sink_norm(sub);

        // mean over rows -> per-context-token vector
        std::vector<float> ctx_scores(n_c, 0.0f);
        for (auto & r : sub) {
            for (int c = 0; c < n_c; c++) ctx_scores[c] += r[c];
        }
        if (!sub.empty()) {
            float inv_rows = (float)(1.0 / (double) sub.size());
            for (auto & v : ctx_scores) v *= inv_rows;
        }
        // Scatter to chunks: per chunk, sum per-token scores assigned to it.
        std::vector<float> chunk_sum(num_context_chunks, 0.0f);
        std::vector<int> chunk_count(num_context_chunks, 0);
        for (int c = 0; c < n_c; c++) {
            int ch = context_local_to_chunk[c];
            if (ch < 0) continue;
            chunk_sum[ch] += ctx_scores[c];
            chunk_count[ch] += 1;
        }
        std::vector<float> row(num_context_chunks, 0.0f);
        double sum = 0;
        for (int c = 0; c < num_context_chunks; c++) {
            row[c] = chunk_count[c] > 0 ? chunk_sum[c] : 0.0f;
            sum += row[c];
        }
        if (global_normalize && sum > 0) for (auto & v : row) v = (float)(v / sum);
        result[q_chunk] = row;
    }

    if (global_normalize) return prior_normalize(result);
    return result;
}

// -----------------------------------------------------------------------------
// JSON writer (minimal)

struct JsonOut {
    std::ostringstream s;

    void escape(const std::string & str) {
        s << '"';
        for (char c : str) {
            switch (c) {
                case '"':  s << "\\\""; break;
                case '\\': s << "\\\\"; break;
                case '\n': s << "\\n"; break;
                case '\r': s << "\\r"; break;
                case '\t': s << "\\t"; break;
                default:
                    if ((unsigned char) c < 0x20) {
                        char buf[8];
                        snprintf(buf, sizeof(buf), "\\u%04x", c);
                        s << buf;
                    } else s << c;
            }
        }
        s << '"';
    }

    void num(double v) {
        if (std::isnan(v) || std::isinf(v)) s << "0";
        else {
            char buf[32];
            snprintf(buf, sizeof(buf), "%.6g", v);
            s << buf;
        }
    }

    void chunks(const std::vector<Chunk> & cs) {
        s << '[';
        for (size_t i = 0; i < cs.size(); i++) {
            if (i) s << ',';
            s << '{';
            s << "\"label\":"; escape(cs[i].label);
            s << ",\"start\":" << cs[i].start;
            s << ",\"end\":" << cs[i].end;
            s << ",\"text\":"; escape(cs[i].text);
            s << '}';
        }
        s << ']';
    }

    void matrix(
        const std::vector<std::vector<float>> & m,
        const std::vector<Chunk> & column_chunks,
        const std::string & column_text,
        int prune_top_k = 0,
        int retain_line_radius = 5
    ) {
        std::vector<int> line_starts;
        line_starts.push_back(0);
        for (size_t i = 0; i < column_text.size(); i++) {
            if (column_text[i] == '\n') line_starts.push_back((int) i + 1);
        }

        auto line_for_offset = [&](int offset) -> int {
            if (line_starts.empty()) return 0;
            int clamped = std::max(0, std::min(offset, (int) column_text.size()));
            auto it = std::upper_bound(line_starts.begin(), line_starts.end(), clamped);
            if (it == line_starts.begin()) return 0;
            return (int) (it - line_starts.begin() - 1);
        };

        std::vector<int> column_lines(column_chunks.size(), 0);
        int max_line = 0;
        for (size_t i = 0; i < column_chunks.size(); i++) {
            column_lines[i] = line_for_offset(column_chunks[i].start);
            max_line = std::max(max_line, column_lines[i]);
        }
        std::vector<std::vector<size_t>> columns_by_line((size_t) max_line + 1);
        for (size_t i = 0; i < column_lines.size(); i++) {
            columns_by_line[(size_t) column_lines[i]].push_back(i);
        }

        s << '[';
        for (size_t i = 0; i < m.size(); i++) {
            if (i) s << ',';
            s << '[';
            std::set<size_t> keep;
            if (prune_top_k > 0 && prune_top_k < (int) m[i].size()) {
                std::vector<std::pair<float, size_t>> ranked;
                ranked.reserve(m[i].size());
                for (size_t j = 0; j < m[i].size(); j++) {
                    float v = m[i][j];
                    if (std::isfinite(v) && v > 0) ranked.push_back({v, j});
                }
                int k = std::min(prune_top_k, (int) ranked.size());
                std::partial_sort(
                    ranked.begin(),
                    ranked.begin() + k,
                    ranked.end(),
                    [](const auto & a, const auto & b) { return a.first > b.first; }
                );
                for (int r = 0; r < k; r++) {
                    size_t hot_col = ranked[r].second;
                    keep.insert(hot_col);
                    if (hot_col >= column_lines.size()) continue;
                    int hot_line = column_lines[hot_col];
                    int first_line = std::max(0, hot_line - retain_line_radius);
                    int last_line = std::min(max_line, hot_line + retain_line_radius);
                    for (int line = first_line; line <= last_line; line++) {
                        for (size_t col : columns_by_line[(size_t) line]) {
                            keep.insert(col);
                        }
                    }
                }
            }
            for (size_t j = 0; j < m[i].size(); j++) {
                if (j) s << ',';
                if (prune_top_k > 0 && prune_top_k < (int) m[i].size() && keep.count(j) == 0) {
                    s << '0';
                } else {
                    num(m[i][j]);
                }
            }
            s << ']';
        }
        s << ']';
    }
};

// -----------------------------------------------------------------------------
// Main

struct Args {
    std::string model;
    std::string context_path;       // single-file context
    std::string context_tree;       // directory containing many context files
    std::string context_glob = ".ts"; // file extension filter when scanning tree
    std::string cache_dir = "cpp/cache";
    std::string query_path;
    std::string query_tree;         // directory of multi-doc queries
    std::string query_glob = ".mdx"; // extension filter for query tree
    std::string out_path = "web/heatmap.tar.gz";
    std::string strip_context = "none";
    std::string strip_query = "none";
    std::string layers; // e.g. "14-20"; empty = default fractions
    double layer_fraction_start = 0.60;
    double layer_fraction_end = 0.88;
    bool sink_normalization = true;
    bool per_file = false;          // iterate context files individually (tiny windows)
    int n_ubatch = 256;
    int prune_top_k = 80;           // keep top-K context hits plus nearby lines per query row in JSON
};

static std::vector<int> parse_layers(const std::string & raw) {
    std::vector<int> out;
    std::stringstream ss(raw);
    std::string part;
    while (std::getline(ss, part, ',')) {
        auto dash = part.find('-');
        if (dash != std::string::npos) {
            int a = std::atoi(part.substr(0, dash).c_str());
            int b = std::atoi(part.substr(dash + 1).c_str());
            if (b < a) std::swap(a, b);
            for (int i = a; i <= b; i++) out.push_back(i);
        } else if (!part.empty()) {
            out.push_back(std::atoi(part.c_str()));
        }
    }
    return out;
}

static Args parse_args(int argc, char ** argv) {
    Args a;
    for (int i = 1; i < argc; i++) {
        std::string k = argv[i];
        auto need = [&](const char * what) {
            if (++i >= argc) { log_line("missing value for %s", what); std::exit(1); }
            return std::string(argv[i]);
        };
        if (k == "--model") a.model = need("--model");
        else if (k == "--context") a.context_path = need("--context");
        else if (k == "--context-tree") a.context_tree = need("--context-tree");
        else if (k == "--context-glob") a.context_glob = need("--context-glob");
        else if (k == "--cache-dir") a.cache_dir = need("--cache-dir");
        else if (k == "--query") a.query_path = need("--query");
        else if (k == "--query-tree") a.query_tree = need("--query-tree");
        else if (k == "--query-glob") a.query_glob = need("--query-glob");
        else if (k == "--per-file") a.per_file = true;
        else if (k == "--output") a.out_path = need("--output");
        else if (k == "--json") {
            log_line("plain JSON output is not supported; use --output FILE.tar.gz");
            std::exit(1);
        }
        else if (k == "--strip-context") a.strip_context = need("--strip-context");
        else if (k == "--strip-query") a.strip_query = need("--strip-query");
        else if (k == "--layers") a.layers = need("--layers");
        else if (k == "--layer-fraction-start") a.layer_fraction_start = std::atof(need("--layer-fraction-start").c_str());
        else if (k == "--layer-fraction-end") a.layer_fraction_end = std::atof(need("--layer-fraction-end").c_str());
        else if (k == "--no-sink-normalization") a.sink_normalization = false;
        else if (k == "--ubatch") a.n_ubatch = std::atoi(need("--ubatch").c_str());
        else if (k == "--prune-top-k" || k == "--prune") a.prune_top_k = std::atoi(need(k.c_str()).c_str());
        else if (k == "--no-prune") a.prune_top_k = 0;
        else { log_line("unknown arg: %s", k.c_str()); std::exit(1); }
    }
    if (a.model.empty() || (a.query_path.empty() && a.query_tree.empty()) || (a.context_path.empty() && a.context_tree.empty())) {
        log_line("usage: --model FILE (--query FILE | --query-tree DIR) (--context FILE | --context-tree DIR) [--per-file] [--output web/heatmap.tar.gz] [--prune-top-k N] [opts]");
        std::exit(1);
    }
    if (!path_ends_with(a.out_path, ".tar.gz") && !path_ends_with(a.out_path, ".tgz")) {
        log_line("output path must end in .tar.gz or .tgz: %s", a.out_path.c_str());
        std::exit(1);
    }
    return a;
}

// -----------------------------------------------------------------------------
// Multi-doc scan: iterate context files one at a time against a single combined
// query corpus (concatenation of every doc under --query-tree). Each context
// file is its own tiny window — pass 1 stays small. KV cache is cleared between
// files. The output JSON contains every doc and every file in one document, so
// the explorer can group them in its existing tree views.

static int run_per_file_scan(const Args & args) {
    auto t_start = ggml_time_us();

    // 1. Walk markdown query docs.
    std::vector<std::string> qry_paths;
    if (!args.query_tree.empty()) {
        walk_directory(args.query_tree, args.query_glob, qry_paths);
    } else {
        qry_paths.push_back(args.query_path);
    }
    if (qry_paths.empty()) { log_line("no query docs found"); return 1; }
    std::sort(qry_paths.begin(), qry_paths.end());

    // 2. Walk context files.
    std::vector<std::string> ctx_paths;
    if (!args.context_tree.empty()) {
        walk_directory(args.context_tree, args.context_glob, ctx_paths);
    } else {
        ctx_paths.push_back(args.context_path);
    }
    if (ctx_paths.empty()) { log_line("no context files found"); return 1; }
    std::sort(ctx_paths.begin(), ctx_paths.end());
    log_line("context: %zu files matching '%s'", ctx_paths.size(), args.context_glob.c_str());

    // 3. Load model once, init context.
    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(args.model.c_str(), mparams);
    if (!model) { log_line("failed to load model %s", args.model.c_str()); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    int n_layers = llama_model_n_layer(model);
    log_line("loaded model: layers=%d (%.2fs)", n_layers, (ggml_time_us() - t_start) / 1e6);

    std::vector<int> selected_layers;
    if (!args.layers.empty()) {
        selected_layers = parse_layers(args.layers);
    } else {
        int s = std::max(0, std::min(n_layers - 1, (int)(n_layers * args.layer_fraction_start)));
        int e = std::max(s + 1, std::min(n_layers, (int)(n_layers * args.layer_fraction_end)));
        for (int i = s; i < e; i++) selected_layers.push_back(i);
    }

    if (!args.cache_dir.empty()) ensure_dir(args.cache_dir);

    // 4. Build markdown query items. Each pass sees heading/list hierarchy,
    // while only the active item line maps to the output row.
    MarkdownQuerySet query_set = build_markdown_query_set(qry_paths, args.query_tree, args.strip_query);
    if (query_set.items.empty()) { log_line("no markdown query items found"); return 1; }

    int total_query_tokens = 0;
    int max_doc_tokens = 0;
    for (auto & item : query_set.items) {
        std::string prompt_with_close = item.prompt_text + kQueryClose;
        std::string sha = sha256_hex(prompt_with_close);
        std::string cache_path = args.cache_dir.empty() ? "" : args.cache_dir + "/qryitem_" + sha + ".tok";
        item.ct = tokenize_with_cache(vocab, prompt_with_close, cache_path);
        int n = (int) item.ct.tokens.size();
        total_query_tokens += n;
        max_doc_tokens = std::max(max_doc_tokens, n);
    }
    log_line("query: %zu markdown items / %zu chars / %d prompt tokens",
        query_set.items.size(), query_set.text.size(), total_query_tokens);

    // 5. Init llama context. n_ctx must be enough for biggest (pass-1 + biggest_doc).
    int n_ctx_budget = 32768; // full Qwen2.5 context — large enough for big files + biggest doc

    CaptureContext cap;
    for (int l : selected_layers) cap.selected_layers.insert(l);
    CallbackUserData ud{ &cap };

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = (uint32_t) n_ctx_budget;
    cparams.n_batch = (uint32_t) n_ctx_budget;
    cparams.n_ubatch = (uint32_t) args.n_ubatch;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
    cparams.cb_eval = eval_callback;
    cparams.cb_eval_user_data = &ud;
    cparams.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { log_line("failed to init context"); return 1; }
    llama_memory_t mem = llama_get_memory(ctx);

    // 6. Iterate per file.
    std::string global_ctx_text;
    std::vector<Chunk> global_ctx_chunks;
    std::vector<std::vector<float>> global_scores(query_set.items.size(), std::vector<float>());

    auto t_scan_start = ggml_time_us();
    int written_files = 0;

    auto write_json_now = [&]() {
        JsonOut j;
        j.s << "{";
        j.s << "\"context_text\":"; j.escape(global_ctx_text);
        j.s << ",\"query_text\":"; j.escape(query_set.text);
        j.s << ",\"context_chunks\":"; j.chunks(global_ctx_chunks);
        j.s << ",\"query_chunks\":"; j.chunks(query_set.chunks);
        j.s << ",\"scores\":"; j.matrix(global_scores, global_ctx_chunks, global_ctx_text, args.prune_top_k);
        j.s << ",\"head_scores\":[]";
        j.s << ",\"layer_scores\":[]";
        j.s << ",\"metadata\":{";
        j.s << "\"backend\":\"llama.cpp\"";
        j.s << ",\"model\":"; j.escape(args.model);
        j.s << ",\"per_file\":true";
        j.s << ",\"query_items_count\":" << query_set.items.size();
        j.s << ",\"query_chunk_mode\":\"markdown_items\"";
        j.s << ",\"files_total\":" << ctx_paths.size();
        j.s << ",\"files_done\":" << written_files;
        j.s << ",\"score_prune_top_k\":" << args.prune_top_k;
        j.s << ",\"layers\":[";
        for (size_t i = 0; i < selected_layers.size(); i++) {
            if (i) j.s << ',';
            j.s << selected_layers[i];
        }
        j.s << "]";
        j.s << "}";
        j.s << "}";
        write_heatmap_output(args.out_path, j.s.str());
    };

    auto decode_token_run = [&](const std::vector<llama_token> & toks,
                                int abs_start_pos,
                                bool capture_active) {
        for (size_t pos = 0; pos < toks.size(); pos += args.n_ubatch) {
            int batch_n = (int) std::min((size_t) args.n_ubatch, toks.size() - pos);
            llama_batch batch = llama_batch_init(batch_n, 0, 1);
            batch.n_tokens = batch_n;
            for (int i = 0; i < batch_n; i++) {
                batch.token[i] = toks[pos + i];
                batch.pos[i] = abs_start_pos + (int) pos + i;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = 0;
            }
            cap.batch_query_pos_start = capture_active ? (abs_start_pos + (int) pos) : -1;
            cap.batch_query_pos_end = capture_active ? (abs_start_pos + (int) pos + batch_n) : -1;
            int rc = llama_decode(ctx, batch);
            llama_batch_free(batch);
            if (rc != 0) { log_line("decode failed pos=%d rc=%d", abs_start_pos + (int) pos, rc); std::exit(1); }
        }
    };

    int per_window_token_budget = (int) cparams.n_ctx - max_doc_tokens - 1024; // safety
    if (per_window_token_budget < 1024) per_window_token_budget = 1024;

    for (size_t fi = 0; fi < ctx_paths.size(); fi++) {
        auto t_file_start = ggml_time_us();
        const std::string & path = ctx_paths[fi];
        std::string rel = args.context_tree.empty()
            ? path
            : path.substr(args.context_tree.size() + (args.context_tree.back() == '/' ? 0 : 1));

        std::string content = strip_text(read_file(path), args.strip_context);
        if (content.empty() || content == "\n") {
            log_line("[%zu/%zu] skip empty %s", fi + 1, ctx_paths.size(), rel.c_str());
            continue;
        }
        // ---- Tokenize file content once for window planning + caching.
        std::string content_sha = sha256_hex(content);
        std::string content_cache = args.cache_dir.empty() ? "" : args.cache_dir + "/" + content_sha + ".tok";
        CachedTokens full_ct = tokenize_with_cache(vocab, content, content_cache);

        auto file_chunks_local = token_chunks_from_offsets(content, full_ct, rel);
        if (file_chunks_local.empty()) {
            file_chunks_local.push_back({rel + ":file", 0, (int) content.size(), content});
        }

        // Count tokens per chunk so we can pack windows by token budget.
        std::vector<int> per_chunk_tokens(file_chunks_local.size(), 0);
        {
            int cidx = 0;
            for (size_t ti = 0; ti < full_ct.tokens.size(); ti++) {
                int ts = full_ct.char_start[ti];
                while (cidx < (int) file_chunks_local.size() && file_chunks_local[cidx].end <= ts) cidx++;
                if (cidx < (int) file_chunks_local.size() &&
                    file_chunks_local[cidx].start <= ts && ts < file_chunks_local[cidx].end) {
                    per_chunk_tokens[cidx]++;
                }
            }
        }

        // Pack chunks into windows of ≤ per_window_token_budget tokens each.
        // Each chunk goes into the current window unless it would push the total
        // past the budget; in that case we open a new window. Oversize single
        // chunks land alone and may still exceed budget — those get skipped at
        // decode-time.
        std::vector<std::pair<int, int>> windows;
        {
            int wstart = 0, wtok = 0;
            for (int i = 0; i < (int) file_chunks_local.size(); i++) {
                if (wstart < i && wtok + per_chunk_tokens[i] > per_window_token_budget) {
                    windows.push_back({wstart, i});
                    wstart = i;
                    wtok = 0;
                }
                wtok += per_chunk_tokens[i];
            }
            if (wstart < (int) file_chunks_local.size()) {
                windows.push_back({wstart, (int) file_chunks_local.size()});
            }
        }
        if (windows.empty()) windows.push_back({0, (int) file_chunks_local.size()});

        // ---- Per-window pass 1 + per-doc pass 2.
        std::string prefix_text = kContextOpen;
        std::string middle_text = kMiddle;
        std::vector<int32_t> skip(selected_layers.begin(), selected_layers.end());
        size_t total_chunks_emitted = 0;
        bool any_window_processed = false;

        for (size_t wi = 0; wi < windows.size(); wi++) {
            int wfirst = windows[wi].first;
            int wlast  = windows[wi].second;
            int win_start_char = file_chunks_local[wfirst].start;
            int win_end_char   = file_chunks_local[wlast - 1].end;
            std::string win_content = content.substr(win_start_char, win_end_char - win_start_char);

            std::vector<Chunk> win_chunks_local;
            win_chunks_local.reserve(wlast - wfirst);
            for (int i = wfirst; i < wlast; i++) {
                Chunk c = file_chunks_local[i];
                c.start -= win_start_char;
                c.end   -= win_start_char;
                win_chunks_local.push_back(c);
            }

            std::string window_tag;
            if (windows.size() > 1) {
                char buf[64];
                std::snprintf(buf, sizeof(buf), " (window %zu/%zu)", wi + 1, windows.size());
                window_tag = buf;
            }
            std::string file_header = "\n// === " + rel + window_tag + " ===\n";

            // ---- Build pass-1 token stream: prefix + file_header + win_content + middle.
            TokenizedDoc pre;
            int pre_cursor = 0;
            auto append_pre = [&](const std::vector<llama_token> & toks,
                                  const std::vector<int> & cs,
                                  const std::vector<int> & ce,
                                  int shift) {
                for (size_t i = 0; i < toks.size(); i++) {
                    pre.tokens.push_back(toks[i]);
                    pre.char_start.push_back(cs[i] + shift);
                    pre.char_end.push_back(ce[i] + shift);
                }
            };
            auto append_pre_inline = [&](const std::string & text) {
                int shift = pre_cursor;
                auto td = tokenize_with_offsets(vocab, text);
                append_pre(td.tokens, td.char_start, td.char_end, shift);
                pre_cursor += (int) text.size();
            };
            auto append_pre_cached = [&](const std::string & text, const std::string & cache_path) {
                int shift = pre_cursor;
                auto ct_inner = tokenize_with_cache(vocab, text, cache_path);
                append_pre(ct_inner.tokens, ct_inner.char_start, ct_inner.char_end, shift);
                pre_cursor += (int) text.size();
            };

            append_pre_inline(prefix_text);
            append_pre_inline(file_header);
            int ctx_body_char_start = pre_cursor;
            std::string win_sha = sha256_hex(win_content);
            std::string win_cache = args.cache_dir.empty() ? "" : args.cache_dir + "/" + win_sha + ".tok";
            append_pre_cached(win_content, win_cache);
            int ctx_body_char_end = pre_cursor;
            append_pre_inline(middle_text);
            int pass1_char_end = pre_cursor;

            std::vector<Chunk> win_chunks_abs;
            win_chunks_abs.reserve(win_chunks_local.size());
            for (auto & c : win_chunks_local) {
                Chunk abs_chunk = c;
                abs_chunk.start = ctx_body_char_start + c.start;
                abs_chunk.end   = ctx_body_char_start + c.end;
                win_chunks_abs.push_back(abs_chunk);
            }

            std::vector<int> ctx_tok_indices = token_indices_for_range(pre, ctx_body_char_start, ctx_body_char_end);
            if (ctx_tok_indices.empty()) {
                log_line("[%zu/%zu] window %zu/%zu skip (no ctx tokens) %s",
                    fi + 1, ctx_paths.size(), wi + 1, windows.size(), rel.c_str());
                continue;
            }

            int pass1_token_count = (int) pre.tokens.size();
            if (pass1_token_count + max_doc_tokens + 8 > (int) cparams.n_ctx) {
                log_line("[%zu/%zu] window %zu/%zu skip (too large: %d ctx + %d doc > %u) %s",
                    fi + 1, ctx_paths.size(), wi + 1, windows.size(),
                    pass1_token_count, max_doc_tokens, cparams.n_ctx, rel.c_str());
                continue;
            }

            // Pass 1.
            llama_memory_clear(mem, false);
            llama_set_flash_attn_skip(ctx, nullptr, 0);
            decode_token_run(pre.tokens, /*abs_start_pos=*/0, /*capture=*/false);

            // Per-query-item pass 2.
            std::vector<std::vector<float>> file_scores(query_set.items.size(),
                                                         std::vector<float>(win_chunks_local.size(), 0.0f));

            for (size_t qi = 0; qi < query_set.items.size(); qi++) {
                auto & item = query_set.items[qi];
                if (item.ct.tokens.empty()) continue;

                int doc_char_start = pass1_char_end;
                int doc_token_pos_start = pass1_token_count;

                TokenizedDoc synth = pre;
                for (size_t i = 0; i < item.ct.tokens.size(); i++) {
                    synth.tokens.push_back(item.ct.tokens[i]);
                    synth.char_start.push_back(doc_char_start + item.ct.char_start[i]);
                    synth.char_end.push_back(doc_char_start + item.ct.char_end[i]);
                }

                std::vector<int> ctx_token_to_chunk = token_to_chunk_map(synth, win_chunks_abs);

                std::vector<int> qry_tok_indices_raw = token_indices_for_range(
                    synth,
                    doc_char_start + item.active_start,
                    doc_char_start + item.active_end
                );
                std::vector<int> qry_tok_indices;
                qry_tok_indices.reserve(qry_tok_indices_raw.size());
                for (int t : qry_tok_indices_raw) if (t - 1 >= 0) qry_tok_indices.push_back(t - 1);
                std::sort(qry_tok_indices.begin(), qry_tok_indices.end());
                qry_tok_indices.erase(std::unique(qry_tok_indices.begin(), qry_tok_indices.end()), qry_tok_indices.end());
                if (qry_tok_indices.empty()) continue;

                cap.per_layer.clear();
                cap.context_token_positions = ctx_tok_indices;
                cap.query_token_positions = qry_tok_indices;

                llama_memory_seq_rm(mem, 0, pass1_token_count, -1);
                llama_set_flash_attn_skip(ctx, skip.data(), (int32_t) skip.size());
                decode_token_run(item.ct.tokens, /*abs_start_pos=*/doc_token_pos_start, /*capture=*/true);

                std::vector<CapturedAttention> captures_ordered;
                for (int l : selected_layers) {
                    auto it = cap.per_layer.find(l);
                    if (it != cap.per_layer.end()) captures_ordered.push_back(it->second);
                }
                if (captures_ordered.empty()) continue;

                std::vector<int> query_local_to_chunk(qry_tok_indices.size(), 0);
                std::vector<int> context_local_to_chunk(ctx_tok_indices.size(), -1);
                for (size_t i = 0; i < ctx_tok_indices.size(); i++) {
                    int t = ctx_tok_indices[i];
                    if (t >= 0 && t < (int) ctx_token_to_chunk.size()) {
                        context_local_to_chunk[i] = ctx_token_to_chunk[t];
                    }
                }

                auto doc_scores = aggregate_captured(
                    captures_ordered,
                    query_local_to_chunk,
                    context_local_to_chunk,
                    1,
                    (int) win_chunks_local.size(),
                    args.sink_normalization,
                    /*global_normalize=*/false
                );

                for (size_t c = 0; c < win_chunks_local.size(); c++) {
                    file_scores[qi][c] = doc_scores.empty() ? 0.0f : doc_scores[0][c];
                }
            }

            // Append this window's chunks + per-token score columns to global state.
            int win_body_offset_in_global = (int) global_ctx_text.size() + (int) file_header.size();
            global_ctx_text += file_header;
            global_ctx_text += win_content;
            for (auto & c : win_chunks_local) {
                Chunk shifted = c;
                shifted.start = win_body_offset_in_global + c.start;
                shifted.end   = win_body_offset_in_global + c.end;
                global_ctx_chunks.push_back(shifted);
            }
            for (size_t q = 0; q < query_set.items.size(); q++) {
                for (size_t c = 0; c < win_chunks_local.size(); c++) {
                    global_scores[q].push_back(file_scores[q][c]);
                }
            }
            total_chunks_emitted += win_chunks_local.size();
            any_window_processed = true;
        }

        if (!any_window_processed) continue;

        written_files++;
        double elapsed = (ggml_time_us() - t_scan_start) / 1e6;
        double file_time = (ggml_time_us() - t_file_start) / 1e6;
        double per_file_avg = elapsed / written_files;
        double eta = per_file_avg * (ctx_paths.size() - (fi + 1));
        log_line("[%zu/%zu] %s (%zu chunks, %zu window%s, %.1fs, avg %.1fs, ETA %.0fs)",
            fi + 1, ctx_paths.size(), rel.c_str(),
            total_chunks_emitted, windows.size(), windows.size() == 1 ? "" : "s",
            file_time, per_file_avg, eta);

        write_json_now();
    }
    write_json_now();

    log_line("scan complete: %d/%zu files in %.1fs",
        written_files, ctx_paths.size(), (ggml_time_us() - t_scan_start) / 1e6);
    log_line("wrote %s", args.out_path.c_str());

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

int main(int argc, char ** argv) {
    Args args = parse_args(argc, argv);

    if (args.per_file) return run_per_file_scan(args);

    auto t_start = ggml_time_us();
    int64_t t_model_load_done = 0;
    int64_t t_tokenize_done = 0;
    int64_t t_context_init_done = 0;
    int64_t t_pass1_done = 0;
    int64_t t_pass2_done = 0;
    int64_t t_aggregate_done = 0;

    // 1. Read query.
    std::string qry_text_raw = read_file(args.query_path);
    std::string qry_text = strip_text(qry_text_raw, args.strip_query);

    // 2. Collect context files (either one file or a directory tree).
    std::vector<ContextFile> context_files;
    if (!args.context_tree.empty()) {
        std::vector<std::string> paths;
        walk_directory(args.context_tree, args.context_glob, paths);
        log_line("scanned %s: %zu files matching '%s'", args.context_tree.c_str(), paths.size(), args.context_glob.c_str());
        context_files.reserve(paths.size());
        for (auto & p : paths) {
            ContextFile cf;
            cf.path = p;
            cf.rel_path = p.substr(args.context_tree.size() + (args.context_tree.back() == '/' ? 0 : 1));
            cf.content = strip_text(read_file(p), args.strip_context);
            context_files.push_back(std::move(cf));
        }
    } else {
        ContextFile cf;
        cf.path = args.context_path;
        cf.rel_path = args.context_path;
        cf.content = strip_text(read_file(args.context_path), args.strip_context);
        context_files.push_back(std::move(cf));
    }
    size_t total_ctx_chars = 0;
    for (auto & cf : context_files) {
        total_ctx_chars += cf.content.size();
    }
    std::vector<Chunk> qry_chunks_local;
    log_line("context: %zu files / %zu chars", context_files.size(), total_ctx_chars);
    log_line("query: %zu chars", qry_text.size());

    // 3. Load model + create context.
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(args.model.c_str(), mparams);
    if (!model) { log_line("failed to load model %s", args.model.c_str()); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    int n_layers = llama_model_n_layer(model);
    t_model_load_done = ggml_time_us();
    log_line("loaded model: layers=%d (model_load=%.2fs)", n_layers, (t_model_load_done - t_start) / 1e6);

    // Determine selected layers.
    std::vector<int> selected_layers;
    if (!args.layers.empty()) {
        selected_layers = parse_layers(args.layers);
    } else {
        int s = std::max(0, std::min(n_layers - 1, (int)(n_layers * args.layer_fraction_start)));
        int e = std::max(s + 1, std::min(n_layers, (int)(n_layers * args.layer_fraction_end)));
        for (int i = s; i < e; i++) selected_layers.push_back(i);
    }
    log_line("selected layers: count=%zu", selected_layers.size());

    // 5. Compose the token stream. Wrapper open + (file_header + file_body)* +
    //    middle + query body + suffix. File bodies come from per-file token cache.
    TokenizedDoc doc;
    int corpus_cursor = 0;

    if (!args.cache_dir.empty()) ensure_dir(args.cache_dir);

    for (auto & cf : context_files) {
        std::string sha = sha256_hex(cf.content);
        std::string cache_path = args.cache_dir.empty() ? "" : args.cache_dir + "/" + sha + ".tok";
        CachedTokens ct = tokenize_with_cache(vocab, cf.content, cache_path);
        cf.chunks = token_chunks_from_offsets(cf.content, ct, cf.rel_path);
        if (cf.chunks.empty() && !cf.content.empty()) {
            cf.chunks.push_back({cf.rel_path + ":file", 0, (int) cf.content.size(), cf.content});
        }
    }
    {
        std::string sha = sha256_hex(qry_text);
        std::string cache_path = args.cache_dir.empty() ? "" : args.cache_dir + "/qrydoc_" + sha + ".tok";
        CachedTokens ct = tokenize_with_cache(vocab, qry_text, cache_path);
        qry_chunks_local = token_chunks_from_offsets(qry_text, ct, "query");
        if (qry_chunks_local.empty() && !qry_text.empty()) {
            qry_chunks_local.push_back({"query:file", 0, (int) qry_text.size(), qry_text});
        }
    }
    size_t total_ctx_chunks = 0;
    for (auto & cf : context_files) total_ctx_chunks += cf.chunks.size();
    log_line("effective chunks: context=%zu query=%zu", total_ctx_chunks, qry_chunks_local.size());

    auto append_chunk = [&](const std::vector<llama_token> & toks,
                            const std::vector<int> & cs,
                            const std::vector<int> & ce,
                            int char_shift) {
        for (size_t i = 0; i < toks.size(); i++) {
            doc.tokens.push_back(toks[i]);
            doc.char_start.push_back(cs[i] + char_shift);
            doc.char_end.push_back(ce[i] + char_shift);
        }
    };
    auto append_text_inline = [&](const std::string & text) {
        int shift = corpus_cursor;
        auto td = tokenize_with_offsets(vocab, text);
        append_chunk(td.tokens, td.char_start, td.char_end, shift);
        corpus_cursor += (int) text.size();
    };
    auto append_text_cached = [&](const std::string & text, const std::string & cache_path) {
        int shift = corpus_cursor;
        auto ct = tokenize_with_cache(vocab, text, cache_path);
        append_chunk(ct.tokens, ct.char_start, ct.char_end, shift);
        corpus_cursor += (int) text.size();
    };

    append_text_inline(kContextOpen);
    int ctx_body_start_wrapped = corpus_cursor;
    int cache_hits = 0;
    std::string ctx_body_text;
    std::vector<Chunk> ctx_chunks_abs;
    std::vector<Chunk> ctx_chunks_local;
    for (auto & cf : context_files) {
        std::string header = make_file_header(cf.rel_path);
        int header_offset_body = (int) ctx_body_text.size();
        append_text_inline(header);
        ctx_body_text += header;

        int file_body_offset_wrapped = corpus_cursor;
        int file_body_offset_body = (int) ctx_body_text.size();
        std::string cache_path;
        if (!args.cache_dir.empty()) {
            std::string sha = sha256_hex(cf.content);
            cache_path = args.cache_dir + "/" + sha + ".tok";
            struct stat st;
            if (stat(cache_path.c_str(), &st) == 0) cache_hits++;
        }
        append_text_cached(cf.content, cache_path);
        ctx_body_text += cf.content;

        for (auto & c : cf.chunks) {
            Chunk abs_chunk = c;
            abs_chunk.start = file_body_offset_wrapped + c.start;
            abs_chunk.end   = file_body_offset_wrapped + c.end;
            ctx_chunks_abs.push_back(abs_chunk);

            Chunk for_json = c;
            for_json.start = file_body_offset_body + c.start;
            for_json.end   = file_body_offset_body + c.end;
            ctx_chunks_local.push_back(for_json);
        }
        (void) header_offset_body;
    }
    int ctx_body_end_wrapped = corpus_cursor;

    append_text_inline(kMiddle);
    int qry_body_start_wrapped = corpus_cursor;
    append_text_inline(qry_text);
    int qry_body_end_wrapped = corpus_cursor;
    append_text_inline(kQueryClose);

    t_tokenize_done = ggml_time_us();
    log_line("tokenized: %zu tokens (tokenize=%.2fs, cache_hits=%d/%zu)",
        doc.tokens.size(), (t_tokenize_done - t_model_load_done) / 1e6,
        cache_hits, context_files.size());

    int ctx_base = ctx_body_start_wrapped;
    int ctx_end = ctx_body_end_wrapped;
    int qry_base = qry_body_start_wrapped;
    int qry_end = qry_body_end_wrapped;

    // qry_chunks_abs (wrapped coordinates) for token-index mapping.
    std::vector<Chunk> qry_chunks_abs;
    qry_chunks_abs.reserve(qry_chunks_local.size());
    for (auto & c : qry_chunks_local) {
        Chunk abs_chunk = c;
        abs_chunk.start = qry_base + c.start;
        abs_chunk.end   = qry_base + c.end;
        qry_chunks_abs.push_back(abs_chunk);
    }

    // 6. Compute context_token_positions (key columns) and query_token_positions
    //    (causal-shifted: token_position - 1 for each query body token).
    std::vector<int> ctx_tok_indices = token_indices_for_range(doc, ctx_base, ctx_end);
    std::vector<int> qry_tok_indices_raw = token_indices_for_range(doc, qry_base, qry_end);
    std::vector<int> qry_tok_indices;
    qry_tok_indices.reserve(qry_tok_indices_raw.size());
    for (int t : qry_tok_indices_raw) if (t - 1 >= 0) qry_tok_indices.push_back(t - 1);
    std::sort(qry_tok_indices.begin(), qry_tok_indices.end());
    qry_tok_indices.erase(std::unique(qry_tok_indices.begin(), qry_tok_indices.end()), qry_tok_indices.end());
    std::sort(ctx_tok_indices.begin(), ctx_tok_indices.end());
    ctx_tok_indices.erase(std::unique(ctx_tok_indices.begin(), ctx_tok_indices.end()), ctx_tok_indices.end());

    if (qry_tok_indices.empty()) { log_line("no query tokens"); return 1; }
    if (ctx_tok_indices.empty()) { log_line("no context tokens"); return 1; }
    int split = qry_tok_indices.front();
    log_line("split=%d context_tokens=%zu query_tokens=%zu",
        split, ctx_tok_indices.size(), qry_tok_indices.size());

    // 7. Token-to-chunk maps.
    std::vector<int> ctx_token_to_chunk = token_to_chunk_map(doc, ctx_chunks_abs);
    std::vector<int> qry_token_to_chunk = token_to_chunk_map(doc, qry_chunks_abs);

    std::vector<int> query_local_to_chunk(qry_tok_indices.size(), -1);
    for (size_t i = 0; i < qry_tok_indices.size(); i++) {
        int original_token = qry_tok_indices[i] + 1; // undo causal shift
        if (original_token >= 0 && original_token < (int) qry_token_to_chunk.size()) {
            query_local_to_chunk[i] = qry_token_to_chunk[original_token];
        }
    }
    std::vector<int> context_local_to_chunk(ctx_tok_indices.size(), -1);
    for (size_t i = 0; i < ctx_tok_indices.size(); i++) {
        int t = ctx_tok_indices[i];
        if (t >= 0 && t < (int) ctx_token_to_chunk.size()) {
            context_local_to_chunk[i] = ctx_token_to_chunk[t];
        }
    }

    // 8. Create llama_context with the eval callback.
    CaptureContext cap;
    cap.context_token_positions = ctx_tok_indices;
    cap.query_token_positions = qry_tok_indices;
    for (int l : selected_layers) cap.selected_layers.insert(l);
    CallbackUserData ud{ &cap };

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = (uint32_t) doc.tokens.size() + 8;
    cparams.n_batch = (uint32_t) doc.tokens.size() + 8;
    cparams.n_ubatch = (uint32_t) std::min<uint32_t>(args.n_ubatch, (uint32_t) doc.tokens.size());
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
    cparams.cb_eval = eval_callback;
    cparams.cb_eval_user_data = &ud;
    cparams.no_perf = true;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { log_line("failed to init context"); return 1; }
    t_context_init_done = ggml_time_us();
    log_line("context init (ctx_init=%.2fs)", (t_context_init_done - t_tokenize_done) / 1e6);

    // 9. Pass 1: feed context tokens [0, split) in ubatch-sized pieces.
    log_line("pass 1: %d context tokens", split);
    auto decode_range = [&](int from, int to_excl, bool capture_active) {
        for (int pos = from; pos < to_excl; pos += args.n_ubatch) {
            int batch_n = std::min(args.n_ubatch, to_excl - pos);
            llama_batch batch = llama_batch_init(batch_n, 0, 1);
            batch.n_tokens = batch_n;
            for (int i = 0; i < batch_n; i++) {
                batch.token[i] = doc.tokens[pos + i];
                batch.pos[i] = pos + i;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = 0;
            }
            if (capture_active) {
                cap.batch_query_pos_start = pos;
                cap.batch_query_pos_end = pos + batch_n;
            } else {
                // sentinel: out of range, callback ignores
                cap.batch_query_pos_start = -1;
                cap.batch_query_pos_end = -1;
            }
            int rc = llama_decode(ctx, batch);
            if (rc != 0) {
                log_line("llama_decode failed at pos=%d rc=%d", pos, rc);
                std::exit(1);
            }
            llama_batch_free(batch);
        }
    };

    // Pass 1 uses flash attention on every layer (fast). No captures needed.
    llama_set_flash_attn_skip(ctx, nullptr, 0);
    decode_range(0, split, /*capture_active=*/false);
    t_pass1_done = ggml_time_us();
    log_line("pass 1 done (pass1=%.2fs)", (t_pass1_done - t_context_init_done) / 1e6);

    // Pass 2: enable the eager attention path for the selected layers so
    // kq_soft_max-<il> shows up in the graph and the callback can grab it.
    // All other layers stay on flash attention.
    std::vector<int32_t> skip_layers(selected_layers.begin(), selected_layers.end());
    llama_set_flash_attn_skip(ctx, skip_layers.data(), (int32_t) skip_layers.size());

    int qry_total = (int) doc.tokens.size() - split;
    log_line("pass 2: %d query tokens (flash-attn skipped on %zu layers)", qry_total, skip_layers.size());
    decode_range(split, (int) doc.tokens.size(), /*capture_active=*/true);
    t_pass2_done = ggml_time_us();
    log_line("pass 2 done; captured layers=%zu (pass2=%.2fs)", cap.per_layer.size(), (t_pass2_done - t_pass1_done) / 1e6);

    // 11. Aggregate.
    std::vector<CapturedAttention> captures_ordered;
    for (int l : selected_layers) {
        auto it = cap.per_layer.find(l);
        if (it != cap.per_layer.end()) captures_ordered.push_back(it->second);
    }
    if (captures_ordered.empty()) {
        log_line("no captures - did flash attention get disabled? aborting.");
        return 1;
    }
    auto scores = aggregate_captured(
        captures_ordered,
        query_local_to_chunk,
        context_local_to_chunk,
        (int) qry_chunks_local.size(),
        (int) ctx_chunks_local.size(),
        args.sink_normalization
    );
    t_aggregate_done = ggml_time_us();
    log_line("aggregation done (aggregate=%.2fs)", (t_aggregate_done - t_pass2_done) / 1e6);

    // 12. Write JSON.
    JsonOut j;
    j.s << "{";
    j.s << "\"context_text\":"; j.escape(ctx_body_text);
    j.s << ",\"query_text\":"; j.escape(qry_text);
    j.s << ",\"context_chunks\":"; j.chunks(ctx_chunks_local);
    j.s << ",\"query_chunks\":"; j.chunks(qry_chunks_local);
    j.s << ",\"scores\":"; j.matrix(scores, ctx_chunks_local, ctx_body_text, args.prune_top_k);
    j.s << ",\"head_scores\":[]";
    j.s << ",\"layer_scores\":[]";
    j.s << ",\"metadata\":{";
    j.s << "\"backend\":\"llama.cpp\"";
    j.s << ",\"model\":"; j.escape(args.model);
    j.s << ",\"layers\":[";
    for (size_t i = 0; i < selected_layers.size(); i++) {
        if (i) j.s << ',';
        j.s << selected_layers[i];
    }
    j.s << "]";
    j.s << ",\"sink_normalization\":" << (args.sink_normalization ? "true" : "false");
    j.s << ",\"score_prune_top_k\":" << args.prune_top_k;
    j.s << ",\"flash_attn\":\"disabled\"";
    j.s << "}";
    j.s << "}";
    write_heatmap_output(args.out_path, j.s.str());

    auto t_end = ggml_time_us();
    double total = (t_end - t_start) / 1e6;
    double model_load = (t_model_load_done - t_start) / 1e6;
    double after_load = (t_end - t_model_load_done) / 1e6;
    double pass1 = (t_pass1_done - t_context_init_done) / 1e6;
    double pass2 = (t_pass2_done - t_pass1_done) / 1e6;
    double ctx_init = (t_context_init_done - t_tokenize_done) / 1e6;
    double tokenize = (t_tokenize_done - t_model_load_done) / 1e6;
    double agg = (t_aggregate_done - t_pass2_done) / 1e6;
    double json_io = (t_end - t_aggregate_done) / 1e6;
    log_line("wrote %s", args.out_path.c_str());
    log_line("--- timing ---");
    log_line("  total:           %.2fs", total);
    log_line("  model_load:      %.2fs", model_load);
    log_line("  after-load:      %.2fs", after_load);
    log_line("    tokenize:      %.2fs", tokenize);
    log_line("    ctx_init:      %.2fs (Metal pipeline warmup)", ctx_init);
    log_line("    pass1 fwd:     %.2fs (%d ctx tokens)", pass1, split);
    log_line("    pass2 fwd:     %.2fs (%d qry tokens) [captures]", pass2, qry_total);
    log_line("    aggregate:     %.2fs", agg);
    log_line("    json+free:     %.2fs", json_io);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
