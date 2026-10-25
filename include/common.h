// Shared infrastructure for the CPU and CUDA FP-growth implementations:
// dataset loading, command-line parsing, item ranking, itemset output and
// run statistics (time / memory) reporting.
#pragma once

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace fpg {

// Transactions in CSR form. Item ids are dense (0 .. labels.size()-1), assigned
// in order of first appearance, and unique within a transaction.
struct Dataset {
    std::vector<std::string> labels;
    std::vector<uint32_t> offsets{0};
    std::vector<int32_t> items;

    size_t num_transactions() const { return offsets.size() - 1; }
    size_t num_items() const { return labels.size(); }
};

// One transaction per line, items separated by whitespace. Any token is
// accepted as an item name. Blank lines are ignored.
Dataset load_transactions(const std::string& path);

struct Options {
    std::string input;
    std::string output;        // empty => itemsets are counted but not written
    std::string min_support_arg;
    bool json = false;
    bool quiet = false;
    // GPU batching limits (ignored by the CPU implementation).
    size_t max_entries = size_t(1) << 23;
    size_t max_cands = size_t(1) << 23;
};

Options parse_args(int argc, char** argv, bool gpu);

// "0.25" or "25%" => relative support, "3" => absolute transaction count.
int resolve_min_count(const std::string& arg, size_t num_transactions);

// Frequent items ordered by descending support (ties broken by item id).
// rank 0 is the most frequent item.
struct ItemRanking {
    std::vector<int32_t> rank_of;   // item id -> rank, or -1 if infrequent
    std::vector<int32_t> item_of;   // rank -> item id
    std::vector<int32_t> support;   // rank -> support count
    int num_frequent() const { return static_cast<int>(item_of.size()); }
};

ItemRanking rank_items(const std::vector<int32_t>& counts, int min_count);

// Writes itemsets as "label label ... #SUP: n" (SPMF style), items sorted by
// item id so that output is deterministic.
class ItemsetWriter {
public:
    ItemsetWriter(const std::string& path, const Dataset& data, const ItemRanking& ranking);
    ~ItemsetWriter();
    bool enabled() const { return file_ != nullptr; }
    void write(const int32_t* ranks, int k, int support);
    void close();

private:
    void flush();
    FILE* file_ = nullptr;
    const Dataset& data_;
    const ItemRanking& ranking_;
    std::vector<int32_t> ids_;
    std::string buf_;
};

class Timer {
public:
    Timer() : start_(std::chrono::steady_clock::now()) {}
    double ms() const {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_).count();
    }
    void reset() { start_ = std::chrono::steady_clock::now(); }

private:
    std::chrono::steady_clock::time_point start_;
};

size_t peak_host_memory_bytes();

struct RunStats {
    std::string algorithm;
    std::string device;
    std::string input;
    size_t transactions = 0;
    size_t distinct_items = 0;
    size_t total_items = 0;
    int min_count = 0;
    int frequent_items = 0;
    size_t tree_nodes = 0;
    uint64_t itemsets = 0;
    std::vector<uint64_t> by_length;   // index k => number of itemsets of size k
    double read_ms = 0, build_ms = 0, mine_ms = 0, write_ms = 0;
    size_t peak_host_bytes = 0;
    size_t peak_device_bytes = 0;
    uint64_t batches = 0;              // GPU only

    void count_itemset(int k) {
        ++itemsets;
        if (by_length.size() <= static_cast<size_t>(k)) by_length.resize(k + 1, 0);
        ++by_length[k];
    }
};

void print_stats(const RunStats& s, bool json);

}  // namespace fpg
