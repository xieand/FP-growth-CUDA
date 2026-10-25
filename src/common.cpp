#include "common.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

namespace fpg {

Dataset load_transactions(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open input file: " + path);
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

    Dataset d;
    std::unordered_map<std::string_view, int32_t> ids;
    ids.reserve(1 << 12);
    std::vector<int32_t> tx;

    auto is_space = [](char c) { return c == ' ' || c == '\t' || c == '\r'; };
    size_t i = 0, n = text.size();
    while (i < n) {
        tx.clear();
        while (i < n && text[i] != '\n') {
            while (i < n && is_space(text[i])) ++i;
            size_t start = i;
            while (i < n && text[i] != '\n' && !is_space(text[i])) ++i;
            if (i == start) continue;
            std::string_view tok(text.data() + start, i - start);
            auto it = ids.find(tok);
            int32_t id;
            if (it == ids.end()) {
                id = static_cast<int32_t>(d.labels.size());
                d.labels.emplace_back(tok);
                ids.emplace(tok, id);
            } else {
                id = it->second;
            }
            tx.push_back(id);
        }
        ++i;  // skip '\n'
        if (tx.empty()) continue;
        std::sort(tx.begin(), tx.end());
        tx.erase(std::unique(tx.begin(), tx.end()), tx.end());
        d.items.insert(d.items.end(), tx.begin(), tx.end());
        d.offsets.push_back(static_cast<uint32_t>(d.items.size()));
    }
    return d;
}

static void usage(const char* prog, bool gpu) {
    std::fprintf(stderr,
                 "usage: %s <transactions.txt> <minsup> [options]\n"
                 "  minsup         relative (0.3 or 30%%) or absolute count (e.g. 3)\n"
                 "  -o <file>      write frequent itemsets (\"a b c #SUP: n\" per line)\n"
                 "  --json         print run statistics as one JSON line\n"
                 "  --quiet        suppress the human-readable summary\n",
                 prog);
    if (gpu)
        std::fprintf(stderr,
                     "  --max-entries N  max conditional-pattern-base entries per GPU batch (default 8388608)\n"
                     "  --max-cands N    max candidate counters per GPU batch (default 8388608)\n");
    std::exit(2);
}

Options parse_args(int argc, char** argv, bool gpu) {
    Options o;
    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) usage(argv[0], gpu);
            return argv[++i];
        };
        if (a == "-o") o.output = next();
        else if (a == "--json") o.json = true;
        else if (a == "--quiet") o.quiet = true;
        else if (gpu && a == "--max-entries") o.max_entries = std::stoull(next());
        else if (gpu && a == "--max-cands") o.max_cands = std::stoull(next());
        else if (a == "-h" || a == "--help") usage(argv[0], gpu);
        else if (!a.empty() && a[0] == '-' && a.size() > 1 && !std::isdigit(static_cast<unsigned char>(a[1]))) usage(argv[0], gpu);
        else pos.push_back(a);
    }
    if (pos.size() != 2) usage(argv[0], gpu);
    o.input = pos[0];
    o.min_support_arg = pos[1];
    return o;
}

int resolve_min_count(const std::string& arg, size_t num_transactions) {
    std::string s = arg;
    bool percent = !s.empty() && s.back() == '%';
    if (percent) s.pop_back();
    double v = std::stod(s);
    if (percent) v /= 100.0;
    if (v <= 0) throw std::runtime_error("minimum support must be positive");
    if (percent || v < 1.0) {
        double c = std::ceil(v * static_cast<double>(num_transactions) - 1e-9);
        return std::max(1, static_cast<int>(c));
    }
    return static_cast<int>(std::ceil(v));
}

ItemRanking rank_items(const std::vector<int32_t>& counts, int min_count) {
    ItemRanking r;
    r.rank_of.assign(counts.size(), -1);
    for (size_t i = 0; i < counts.size(); ++i)
        if (counts[i] >= min_count) r.item_of.push_back(static_cast<int32_t>(i));
    std::sort(r.item_of.begin(), r.item_of.end(), [&](int32_t a, int32_t b) {
        return counts[a] != counts[b] ? counts[a] > counts[b] : a < b;
    });
    r.support.resize(r.item_of.size());
    for (size_t k = 0; k < r.item_of.size(); ++k) {
        r.rank_of[r.item_of[k]] = static_cast<int32_t>(k);
        r.support[k] = counts[r.item_of[k]];
    }
    return r;
}

ItemsetWriter::ItemsetWriter(const std::string& path, const Dataset& data, const ItemRanking& ranking)
    : data_(data), ranking_(ranking) {
    if (path.empty()) return;
    file_ = std::fopen(path.c_str(), "wb");
    if (!file_) throw std::runtime_error("cannot open output file: " + path);
    buf_.reserve(1 << 21);
}

ItemsetWriter::~ItemsetWriter() { close(); }

void ItemsetWriter::write(const int32_t* ranks, int k, int support) {
    if (!file_) return;
    ids_.assign(ranks, ranks + k);
    for (auto& x : ids_) x = ranking_.item_of[x];
    std::sort(ids_.begin(), ids_.end());
    for (int i = 0; i < k; ++i) {
        if (i) buf_.push_back(' ');
        buf_ += data_.labels[ids_[i]];
    }
    buf_ += " #SUP: ";
    buf_ += std::to_string(support);
    buf_.push_back('\n');
    if (buf_.size() >= (1 << 20)) flush();
}

void ItemsetWriter::flush() {
    if (file_ && !buf_.empty()) std::fwrite(buf_.data(), 1, buf_.size(), file_);
    buf_.clear();
}

void ItemsetWriter::close() {
    if (!file_) return;
    flush();
    std::fclose(file_);
    file_ = nullptr;
}

size_t peak_host_memory_bytes() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc;
    if (K32GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) return pmc.PeakWorkingSetSize;
    return 0;
#else
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
#if defined(__APPLE__)
    return static_cast<size_t>(ru.ru_maxrss);
#else
    return static_cast<size_t>(ru.ru_maxrss) * 1024;
#endif
#endif
}

static std::string json_escape(const std::string& s) {
    std::string r;
    for (char c : s) {
        if (c == '"' || c == '\\') r.push_back('\\');
        r.push_back(c);
    }
    return r;
}

void print_stats(const RunStats& s, bool json) {
    double mb = 1024.0 * 1024.0;
    if (json) {
        std::printf("{\"algorithm\":\"%s\",\"device\":\"%s\",\"input\":\"%s\",\"transactions\":%zu,"
                    "\"distinct_items\":%zu,\"total_items\":%zu,\"min_count\":%d,\"frequent_items\":%d,"
                    "\"tree_nodes\":%zu,\"itemsets\":%llu,\"by_length\":[",
                    s.algorithm.c_str(), json_escape(s.device).c_str(), json_escape(s.input).c_str(),
                    s.transactions, s.distinct_items, s.total_items, s.min_count, s.frequent_items, s.tree_nodes,
                    static_cast<unsigned long long>(s.itemsets));
        for (size_t k = 1; k < s.by_length.size(); ++k)
            std::printf("%s%llu", k > 1 ? "," : "", static_cast<unsigned long long>(s.by_length[k]));
        std::printf("],\"read_ms\":%.3f,\"build_ms\":%.3f,\"mine_ms\":%.3f,\"write_ms\":%.3f,"
                    "\"algo_ms\":%.3f,\"peak_host_mb\":%.2f,\"peak_device_mb\":%.2f,\"batches\":%llu}\n",
                    s.read_ms, s.build_ms, s.mine_ms, s.write_ms, s.build_ms + s.mine_ms, s.peak_host_bytes / mb,
                    s.peak_device_bytes / mb, static_cast<unsigned long long>(s.batches));
        return;
    }
    std::printf("== FP-growth (%s) on %s\n", s.algorithm.c_str(), s.device.c_str());
    std::printf("input            : %s\n", s.input.c_str());
    std::printf("transactions     : %zu  (distinct items %zu, avg length %.2f)\n", s.transactions,
                s.distinct_items, s.transactions ? double(s.total_items) / s.transactions : 0.0);
    std::printf("min support      : %d transactions (%.4f%%)\n", s.min_count,
                s.transactions ? 100.0 * s.min_count / s.transactions : 0.0);
    std::printf("frequent items   : %d\n", s.frequent_items);
    std::printf("FP-tree nodes    : %zu\n", s.tree_nodes);
    std::printf("frequent itemsets: %llu\n", static_cast<unsigned long long>(s.itemsets));
    for (size_t k = 1; k < s.by_length.size(); ++k)
        if (s.by_length[k])
            std::printf("    size %-3zu      : %llu\n", k, static_cast<unsigned long long>(s.by_length[k]));
    std::printf("time read        : %10.3f ms\n", s.read_ms);
    std::printf("time build tree  : %10.3f ms\n", s.build_ms);
    std::printf("time mine        : %10.3f ms\n", s.mine_ms);
    std::printf("time write       : %10.3f ms\n", s.write_ms);
    std::printf("time algorithm   : %10.3f ms  (build + mine)\n", s.build_ms + s.mine_ms);
    std::printf("peak host memory : %10.2f MB\n", s.peak_host_bytes / mb);
    if (s.peak_device_bytes || s.batches)
        std::printf("peak device mem  : %10.2f MB  (%llu mining batches)\n", s.peak_device_bytes / mb,
                    static_cast<unsigned long long>(s.batches));
}

}  // namespace fpg
