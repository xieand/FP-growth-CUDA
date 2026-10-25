// Sequential FP-growth (Han, Pei & Yin, SIGMOD 2000).
//
//   1. Count item supports and rank frequent items by descending support.
//   2. Insert every transaction (frequent items only, in rank order) into an
//      FP-tree; nodes holding the same item are chained by node-links.
//   3. For each item i of a tree, from least to most frequent, output
//      suffix + {i}, build the conditional FP-tree of i from the prefix paths
//      of its nodes and mine it recursively. A tree that is a single path is
//      enumerated directly (every subset of the path is frequent).
#include <algorithm>
#include <cstdio>
#include <exception>
#include <vector>

#include "common.h"

namespace {

struct FPTree {
    std::vector<int32_t> item, count, parent, first_child, next_sibling, next_same;
    std::vector<int32_t> head;      // rank -> first node of that item (-1 if none)
    std::vector<int32_t> support;   // rank -> sum of node counts
    bool single_path = true;

    explicit FPTree(int n_items) : head(n_items, -1), support(n_items, 0) { add_node(-1, -1); }

    int32_t size() const { return static_cast<int32_t>(item.size()); }
    int n_items() const { return static_cast<int>(head.size()); }

    int32_t add_node(int32_t it, int32_t par) {
        int32_t id = size();
        item.push_back(it);
        count.push_back(0);
        parent.push_back(par);
        first_child.push_back(-1);
        next_sibling.push_back(-1);
        next_same.push_back(-1);
        if (par >= 0) {
            if (first_child[par] != -1) single_path = false;
            next_sibling[id] = first_child[par];
            first_child[par] = id;
            next_same[id] = head[it];
            head[it] = id;
        }
        return id;
    }

    // path: item ranks in ascending order (root side first).
    void insert(const int32_t* path, int len, int32_t cnt) {
        int32_t cur = 0;
        for (int k = 0; k < len; ++k) {
            int32_t it = path[k];
            int32_t c = first_child[cur];
            while (c != -1 && item[c] != it) c = next_sibling[c];
            if (c == -1) c = add_node(it, cur);
            count[c] += cnt;
            support[it] += cnt;
            cur = c;
        }
    }
};

class Miner {
public:
    Miner(int min_count, fpg::ItemsetWriter& out, fpg::RunStats& stats)
        : min_count_(min_count), out_(out), stats_(stats) {}

    void mine(const FPTree& t) {
        if (t.single_path) {
            mine_single_path(t);
            return;
        }
        std::vector<int32_t> cond_count, path;
        for (int i = t.n_items() - 1; i >= 0; --i) {
            if (t.support[i] < min_count_) continue;
            suffix_.push_back(i);
            emit(t.support[i]);

            if (i > 0) {
                // Conditional pattern base of i: prefix paths of all i-nodes.
                cond_count.assign(i, 0);
                for (int32_t n = t.head[i]; n != -1; n = t.next_same[n]) {
                    int32_t c = t.count[n];
                    for (int32_t p = t.parent[n]; p != 0; p = t.parent[p]) cond_count[t.item[p]] += c;
                }
                bool any = false;
                for (int k = 0; k < i && !any; ++k) any = cond_count[k] >= min_count_;
                if (any) {
                    FPTree cond(i);
                    for (int32_t n = t.head[i]; n != -1; n = t.next_same[n]) {
                        path.clear();
                        for (int32_t p = t.parent[n]; p != 0; p = t.parent[p])
                            if (cond_count[t.item[p]] >= min_count_) path.push_back(t.item[p]);
                        if (path.empty()) continue;
                        std::reverse(path.begin(), path.end());
                        cond.insert(path.data(), static_cast<int>(path.size()), t.count[n]);
                    }
                    mine(cond);
                }
            }
            suffix_.pop_back();
        }
    }

private:
    void emit(int32_t support) {
        stats_.count_itemset(static_cast<int>(suffix_.size()));
        out_.write(suffix_.data(), static_cast<int>(suffix_.size()), support);
    }

    // Along a single path node counts never increase, so the support of any
    // subset of the path equals the count of its deepest node.
    void mine_single_path(const FPTree& t) {
        path_items_.clear();
        path_counts_.clear();
        for (int32_t n = t.first_child[0]; n != -1; n = t.first_child[n]) {
            path_items_.push_back(t.item[n]);
            path_counts_.push_back(t.count[n]);
        }
        enumerate_subsets(0);
    }

    void enumerate_subsets(size_t from) {
        for (size_t j = from; j < path_items_.size(); ++j) {
            if (path_counts_[j] < min_count_) break;
            suffix_.push_back(path_items_[j]);
            emit(path_counts_[j]);
            enumerate_subsets(j + 1);
            suffix_.pop_back();
        }
    }

    int min_count_;
    fpg::ItemsetWriter& out_;
    fpg::RunStats& stats_;
    std::vector<int32_t> suffix_;
    std::vector<int32_t> path_items_, path_counts_;
};

}  // namespace

int main(int argc, char** argv) {
    try {
        fpg::Options opt = fpg::parse_args(argc, argv, false);
        fpg::RunStats st;
        st.algorithm = "cpu";
        st.device = "CPU (sequential)";
        st.input = opt.input;

        fpg::Timer timer;
        fpg::Dataset data = fpg::load_transactions(opt.input);
        st.read_ms = timer.ms();
        st.transactions = data.num_transactions();
        st.distinct_items = data.num_items();
        st.total_items = data.items.size();
        st.min_count = fpg::resolve_min_count(opt.min_support_arg, st.transactions);

        timer.reset();
        std::vector<int32_t> counts(data.num_items(), 0);
        for (int32_t it : data.items) ++counts[it];
        fpg::ItemRanking ranking = fpg::rank_items(counts, st.min_count);
        st.frequent_items = ranking.num_frequent();

        FPTree tree(ranking.num_frequent());
        std::vector<int32_t> tx;
        for (size_t t = 0; t < data.num_transactions(); ++t) {
            tx.clear();
            for (uint32_t k = data.offsets[t]; k < data.offsets[t + 1]; ++k) {
                int32_t r = ranking.rank_of[data.items[k]];
                if (r >= 0) tx.push_back(r);
            }
            std::sort(tx.begin(), tx.end());
            tree.insert(tx.data(), static_cast<int>(tx.size()), 1);
        }
        st.tree_nodes = static_cast<size_t>(tree.size());
        st.build_ms = timer.ms();

        fpg::ItemsetWriter writer(opt.output, data, ranking);
        timer.reset();
        Miner miner(st.min_count, writer, st);
        miner.mine(tree);
        double mine_and_write = timer.ms();
        timer.reset();
        writer.close();
        st.write_ms = timer.ms();
        st.mine_ms = mine_and_write;
        st.peak_host_bytes = fpg::peak_host_memory_bytes();

        if (!opt.quiet) fpg::print_stats(st, false);
        if (opt.json) fpg::print_stats(st, true);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
