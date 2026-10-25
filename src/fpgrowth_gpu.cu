// Data-parallel FP-growth for CUDA GPUs.
//
// Phase 1  (GPU) item supports: one atomicAdd per item occurrence.
// Phase 2  (GPU) every transaction drops infrequent items and sorts the rest
//          by frequency rank (one thread per transaction).
// Phase 3  (GPU) FP-tree construction, one depth level at a time. At depth d
//          each transaction still longer than d forms the key
//          (node it reached at depth d-1, its d-th item); the keys are radix
//          sorted and every distinct key becomes a new tree node whose count
//          is the number of transactions sharing it. The resulting tree is
//          exactly the FP-tree (a prefix trie of the ranked transactions),
//          stored as flat parent/item/count arrays.
// Phase 4  (GPU) pattern growth. The conditional pattern base of an itemset P
//          is kept as a list of entries (node, count) where node points at the
//          top of a prefix path in the global FP-tree. For a batch of patterns:
//            a. one thread per entry walks its prefix path to the root and
//               atomically accumulates support[P][j] for every item j on it;
//            b. frequent (P, j) pairs become new itemsets P + {j};
//            c. every entry walks its path again and, for each frequent
//               extension j found at node a, emits (child, parent(a), count);
//               sort + reduce-by-key merges identical prefix paths, which is
//               the same compression a conditional FP-tree provides.
//          Child pattern bases are pushed on a host-side stack and processed
//          depth first; small batches are merged and large ones split so that
//          device memory stays bounded (--max-entries / --max-cands).
#include <exception>
#include <vector>

#include "common.h"
#include "cuda_compat.h"

namespace {

using gpu::DevBuf;

#define GRID_STRIDE(i, n) \
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < (n); \
         i += static_cast<size_t>(blockDim.x) * gridDim.x)

// Index p of the segment with off[p] <= e < off[p + 1] (off[0] = 0, off[n_seg] > e).
__device__ inline int find_segment(const uint32_t* off, int n_seg, uint32_t e) {
    int lo = 0, hi = n_seg;
    while (hi - lo > 1) {
        int mid = (lo + hi) >> 1;
        if (off[mid] <= e) lo = mid;
        else hi = mid;
    }
    return lo;
}

__device__ inline uint64_t pack(uint32_t hi, uint32_t lo) { return (static_cast<uint64_t>(hi) << 32) | lo; }

// ------------------------------------------------------------ phases 1 & 2

__global__ void k_count_items(const int32_t* items, size_t n, int32_t* counts) {
    GRID_STRIDE(i, n) atomicAdd(&counts[items[i]], 1);
}

__global__ void k_rank_and_sort(const uint32_t* off, size_t n_tx, const int32_t* items, const int32_t* rank_of,
                                int32_t* ranked, int32_t* len) {
    GRID_STRIDE(t, n_tx) {
        uint32_t b = off[t], e = off[t + 1];
        uint32_t m = 0;
        for (uint32_t k = b; k < e; ++k) {
            int32_t r = rank_of[items[k]];
            if (r < 0) continue;
            uint32_t p = b + m;
            while (p > b && ranked[p - 1] > r) {
                ranked[p] = ranked[p - 1];
                --p;
            }
            ranked[p] = r;
            ++m;
        }
        len[t] = static_cast<int32_t>(m);
    }
}

// ----------------------------------------------------------------- phase 3

__global__ void k_filter_active(const int32_t* act_in, size_t n, const int32_t* len, int depth, int32_t* act_out,
                                unsigned* cursor) {
    GRID_STRIDE(i, n) {
        int32_t t = act_in ? act_in[i] : static_cast<int32_t>(i);
        if (len[t] > depth) act_out[atomicAdd(cursor, 1u)] = t;
    }
}

__global__ void k_level_keys(const int32_t* act, size_t n, const uint32_t* off, const int32_t* ranked, int depth,
                             const int32_t* cur_node, uint64_t* keys, int32_t* vals) {
    GRID_STRIDE(i, n) {
        int32_t t = act[i];
        keys[i] = pack(static_cast<uint32_t>(cur_node[t]), static_cast<uint32_t>(ranked[off[t] + depth]));
        vals[i] = t;
    }
}

__global__ void k_run_flags(const uint64_t* keys, size_t n, uint32_t* flags) {
    GRID_STRIDE(i, n) flags[i] = (i == 0 || keys[i] != keys[i - 1]) ? 1u : 0u;
}

__global__ void k_make_nodes(const uint64_t* keys, const int32_t* vals, const uint32_t* flags, const uint32_t* pos,
                             size_t n, int32_t base, int32_t* node_parent, int32_t* node_item, int32_t* node_count,
                             int32_t* cur_node) {
    GRID_STRIDE(i, n) {
        int32_t id = base + static_cast<int32_t>(pos[i] + flags[i]) - 1;
        if (flags[i]) {
            node_parent[id] = static_cast<int32_t>(keys[i] >> 32);
            node_item[id] = static_cast<int32_t>(keys[i] & 0xffffffffu);
        }
        atomicAdd(&node_count[id], 1);
        cur_node[vals[i]] = id;
    }
}

// ----------------------------------------------------------------- phase 4

// Level-1 pattern bases: for each node n (item i) whose parent is not the
// root, entry (parent(n), count(n)) belongs to pattern {i}.
__global__ void k_level1_entries(size_t n_nodes, const int32_t* parent, const int32_t* item, const int32_t* count,
                                 uint64_t* keys, int32_t* vals, unsigned* cursor) {
    GRID_STRIDE(n, n_nodes) {
        if (n == 0 || parent[n] == 0) continue;
        unsigned w = atomicAdd(cursor, 1u);
        keys[w] = pack(static_cast<uint32_t>(item[n]), static_cast<uint32_t>(parent[n]));
        vals[w] = count[n];
    }
}

__global__ void k_count_extensions(size_t n_ent, int n_pat, const uint32_t* ent_off, const int32_t* ent_node,
                                   const int32_t* ent_cnt, const uint32_t* cand_off, const int32_t* parent,
                                   const int32_t* item, int32_t* sup, int32_t* nent) {
    GRID_STRIDE(e, n_ent) {
        int p = find_segment(ent_off, n_pat, static_cast<uint32_t>(e));
        uint32_t base = cand_off[p];
        int32_t c = ent_cnt[e];
        for (int32_t v = ent_node[e]; v != 0; v = parent[v]) {
            atomicAdd(&sup[base + item[v]], c);
            atomicAdd(&nent[base + item[v]], 1);
        }
    }
}

__global__ void k_frequent_flags(const int32_t* sup, size_t n, int min_count, uint32_t* flags) {
    GRID_STRIDE(s, n) flags[s] = sup[s] >= min_count ? 1u : 0u;
}

__global__ void k_collect_children(size_t n_cand, int n_pat, const uint32_t* cand_off, const uint32_t* flags,
                                   const uint32_t* pos, const int32_t* sup, const int32_t* nent, int32_t* ch_pat,
                                   int32_t* ch_item, int32_t* ch_sup, int32_t* ch_nent, int32_t* slot_child) {
    GRID_STRIDE(s, n_cand) {
        if (!flags[s]) {
            slot_child[s] = -1;
            continue;
        }
        int p = find_segment(cand_off, n_pat, static_cast<uint32_t>(s));
        uint32_t k = pos[s];
        ch_pat[k] = p;
        ch_item[k] = static_cast<int32_t>(s - cand_off[p]);
        ch_sup[k] = sup[s];
        ch_nent[k] = nent[s];
        slot_child[s] = static_cast<int32_t>(k);
    }
}

__global__ void k_emit_entries(size_t n_ent, int n_pat, const uint32_t* ent_off, const int32_t* ent_node,
                               const int32_t* ent_cnt, const uint32_t* cand_off, const int32_t* parent,
                               const int32_t* item, const int32_t* slot_child, int32_t lo, int32_t hi, uint64_t* keys,
                               int32_t* vals, unsigned* cursor) {
    GRID_STRIDE(e, n_ent) {
        int p = find_segment(ent_off, n_pat, static_cast<uint32_t>(e));
        uint32_t base = cand_off[p];
        int32_t c = ent_cnt[e];
        for (int32_t v = ent_node[e]; v != 0;) {
            int32_t up = parent[v];
            int32_t k = slot_child[base + item[v]];
            if (k >= lo && k < hi && up != 0) {
                unsigned w = atomicAdd(cursor, 1u);
                keys[w] = pack(static_cast<uint32_t>(k - lo), static_cast<uint32_t>(up));
                vals[w] = c;
            }
            v = up;
        }
    }
}

// --------------------------------------------------------------------- host

int bit_width(uint64_t x) {
    int b = 0;
    while (b < 64 && (uint64_t(1) << b) <= x) ++b;
    return b;
}

// A set of patterns together with their conditional pattern bases.
struct Batch {
    std::vector<int32_t> gid;     // output id of the itemset (-1 when not writing output)
    std::vector<int32_t> last;    // rank of the most recently added item; extensions have smaller rank
    std::vector<int32_t> len;     // itemset size
    std::vector<uint32_t> off{0}; // entry range of each pattern
    std::vector<int32_t> node, cnt;
    size_t cands = 0;             // sum of last[] = number of candidate counters

    size_t patterns() const { return last.size(); }
    size_t entries() const { return node.size(); }

    void add_pattern(int32_t g, int32_t l, int32_t n) {
        gid.push_back(g);
        last.push_back(l);
        len.push_back(n);
        cands += static_cast<size_t>(l);
    }
    void close_pattern() { off.push_back(static_cast<uint32_t>(node.size())); }

    void append(Batch&& o) {
        uint32_t shift = static_cast<uint32_t>(node.size());
        gid.insert(gid.end(), o.gid.begin(), o.gid.end());
        last.insert(last.end(), o.last.begin(), o.last.end());
        len.insert(len.end(), o.len.begin(), o.len.end());
        for (size_t p = 1; p < o.off.size(); ++p) off.push_back(o.off[p] + shift);
        node.insert(node.end(), o.node.begin(), o.node.end());
        cnt.insert(cnt.end(), o.cnt.begin(), o.cnt.end());
        cands += o.cands;
    }
};

class GpuFpGrowth {
public:
    GpuFpGrowth(const fpg::Options& opt, const fpg::Dataset& data, fpg::RunStats& st)
        : opt_(opt), data_(data), st_(st), writing_(!opt.output.empty()) {}

    void run() {
        fpg::Timer timer;
        build_tree();
        gpu::synchronize();
        st_.build_ms = timer.ms();

        timer.reset();
        mine();
        gpu::synchronize();
        st_.mine_ms = timer.ms();

        timer.reset();
        write_output();
        st_.write_ms = timer.ms();
        st_.peak_device_bytes = gpu::g_peak_bytes;
    }

private:
    // ------------------------------------------------------------ phases 1-3
    void build_tree() {
        const size_t T = data_.num_transactions();
        const size_t N = data_.items.size();
        min_count_ = st_.min_count;

        DevBuf<uint32_t> d_off;
        DevBuf<int32_t> d_items, d_counts;
        d_off.upload(data_.offsets);
        d_items.upload(data_.items);
        d_counts.zero(data_.num_items());
        gpu::launch(k_count_items, N, d_items.get(), N, d_counts.get());
        std::vector<int32_t> counts;
        d_counts.download(counts, data_.num_items());

        ranking_ = fpg::rank_items(counts, min_count_);
        const int F = ranking_.num_frequent();
        st_.frequent_items = F;

        DevBuf<int32_t> d_rank_of, d_ranked, d_len;
        d_rank_of.upload(ranking_.rank_of);
        d_ranked.reserve(N);
        d_len.reserve(T);
        gpu::launch(k_rank_and_sort, T, d_off.get(), T, d_items.get(), d_rank_of.get(), d_ranked.get(), d_len.get());
        d_items.release();

        const size_t max_nodes = N + 1;
        d_parent_.reserve(max_nodes);
        d_item_.reserve(max_nodes);
        d_count_.zero(max_nodes);
        int32_t root[3] = {0, -1, static_cast<int32_t>(T)};
        gpu::copy_h2d(d_parent_.get(), &root[0], sizeof(int32_t));
        gpu::copy_h2d(d_item_.get(), &root[1], sizeof(int32_t));
        gpu::copy_h2d(d_count_.get(), &root[2], sizeof(int32_t));
        n_nodes_ = 1;

        DevBuf<int32_t> d_cur(T), d_act(T), d_act2(T);
        DevBuf<unsigned> d_cursor(1);
        DevBuf<uint32_t> d_flags(T), d_pos(T);
        d_keys_.reserve(T);
        d_keys2_.reserve(T);
        d_vals_.reserve(T);
        d_vals2_.reserve(T);
        gpu::fill_zero(d_cur.get(), T * sizeof(int32_t));

        d_cursor.zero(1);
        gpu::launch(k_filter_active, T, static_cast<const int32_t*>(nullptr), T, d_len.get(), 0, d_act.get(),
                    d_cursor.get());
        size_t n_act = d_cursor.at(0);

        for (int depth = 0; n_act > 0; ++depth) {
            const int n = static_cast<int>(n_act);
            gpu::launch(k_level_keys, n_act, d_act.get(), n_act, d_off.get(), d_ranked.get(), depth, d_cur.get(),
                        d_keys_.get(), d_vals_.get());
            prim_.sort_pairs(d_keys_.get(), d_keys2_.get(), d_vals_.get(), d_vals2_.get(), n,
                             32 + bit_width(static_cast<uint64_t>(n_nodes_ - 1)));
            gpu::launch(k_run_flags, n_act, d_keys2_.get(), n_act, d_flags.get());
            prim_.exclusive_sum(d_flags.get(), d_pos.get(), n);
            const int32_t new_nodes = static_cast<int32_t>(d_pos.at(n_act - 1) + d_flags.at(n_act - 1));
            gpu::launch(k_make_nodes, n_act, d_keys2_.get(), d_vals2_.get(), d_flags.get(), d_pos.get(), n_act,
                        n_nodes_, d_parent_.get(), d_item_.get(), d_count_.get(), d_cur.get());
            n_nodes_ += new_nodes;

            d_cursor.zero(1);
            gpu::launch(k_filter_active, n_act, static_cast<const int32_t*>(d_act.get()), n_act, d_len.get(),
                        depth + 1, d_act2.get(), d_cursor.get());
            n_act = d_cursor.at(0);
            d_act.swap(d_act2);
        }
        st_.tree_nodes = static_cast<size_t>(n_nodes_);
    }

    // --------------------------------------------------------------- phase 4
    int32_t add_itemset(int32_t parent_gid, int32_t item, int32_t support, int len) {
        st_.count_itemset(len);
        if (!writing_) return -1;
        out_parent_.push_back(parent_gid);
        out_item_.push_back(item);
        out_sup_.push_back(support);
        return static_cast<int32_t>(out_item_.size() - 1);
    }

    void mine() {
        const int F = ranking_.num_frequent();
        if (F == 0) return;
        std::vector<int32_t> single_gid(F);
        for (int r = 0; r < F; ++r) single_gid[r] = add_itemset(-1, r, ranking_.support[r], 1);

        const size_t n_nodes = static_cast<size_t>(n_nodes_);
        d_keys_.reserve(n_nodes);
        d_keys2_.reserve(n_nodes);
        d_vals_.reserve(n_nodes);
        d_vals2_.reserve(n_nodes);
        d_cursor_.zero(1);
        gpu::launch(k_level1_entries, n_nodes, n_nodes, d_parent_.get(), d_item_.get(), d_count_.get(), d_keys_.get(),
                    d_vals_.get(), d_cursor_.get());
        const unsigned m = d_cursor_.at(0);
        prim_.sort_pairs(d_keys_.get(), d_keys2_.get(), d_vals_.get(), d_vals2_.get(), static_cast<int>(m),
                         64);
        std::vector<uint64_t> keys;
        std::vector<int32_t> vals;
        d_keys2_.download(keys, m);
        d_vals2_.download(vals, m);

        // Split the level-1 bases into batches that respect the budgets.
        Batch cur;
        size_t i = 0;
        while (i < m) {
            const int32_t it = static_cast<int32_t>(keys[i] >> 32);
            size_t j = i;
            while (j < m && static_cast<int32_t>(keys[j] >> 32) == it) ++j;
            if (cur.patterns() > 0 &&
                (cur.entries() + (j - i) > opt_.max_entries || cur.cands + it > opt_.max_cands)) {
                stack_.push_back(std::move(cur));
                cur = Batch();
            }
            cur.add_pattern(single_gid[it], it, 1);
            for (size_t k = i; k < j; ++k) {
                cur.node.push_back(static_cast<int32_t>(keys[k] & 0xffffffffu));
                cur.cnt.push_back(vals[k]);
            }
            cur.close_pattern();
            i = j;
        }
        if (cur.patterns() > 0) stack_.push_back(std::move(cur));

        while (!stack_.empty()) {
            Batch b = std::move(stack_.back());
            stack_.pop_back();
            while (!stack_.empty() && b.entries() + stack_.back().entries() <= opt_.max_entries &&
                   b.cands + stack_.back().cands <= opt_.max_cands) {
                b.append(std::move(stack_.back()));
                stack_.pop_back();
            }
            process(b);
            ++st_.batches;
        }
    }

    void process(const Batch& b) {
        const int P = static_cast<int>(b.patterns());
        const size_t E = b.entries();
        std::vector<uint32_t> cand_off(P + 1, 0);
        for (int p = 0; p < P; ++p) cand_off[p + 1] = cand_off[p] + static_cast<uint32_t>(b.last[p]);
        const size_t C = cand_off[P];
        if (E == 0 || C == 0) return;

        d_ent_off_.upload(b.off);
        d_ent_node_.upload(b.node);
        d_ent_cnt_.upload(b.cnt);
        d_cand_off_.upload(cand_off);
        d_sup_.zero(C);
        d_nent_.zero(C);
        gpu::launch(k_count_extensions, E, E, P, d_ent_off_.get(), d_ent_node_.get(), d_ent_cnt_.get(),
                    d_cand_off_.get(), d_parent_.get(), d_item_.get(), d_sup_.get(), d_nent_.get());

        d_flags_.reserve(C);
        d_pos_.reserve(C);
        gpu::launch(k_frequent_flags, C, d_sup_.get(), C, min_count_, d_flags_.get());
        prim_.exclusive_sum(d_flags_.get(), d_pos_.get(), static_cast<int>(C));
        const size_t n_child = d_pos_.at(C - 1) + d_flags_.at(C - 1);
        if (n_child == 0) return;

        d_ch_pat_.reserve(n_child);
        d_ch_item_.reserve(n_child);
        d_ch_sup_.reserve(n_child);
        d_ch_nent_.reserve(n_child);
        d_slot_child_.reserve(C);
        gpu::launch(k_collect_children, C, C, P, d_cand_off_.get(), d_flags_.get(), d_pos_.get(), d_sup_.get(),
                    d_nent_.get(), d_ch_pat_.get(), d_ch_item_.get(), d_ch_sup_.get(), d_ch_nent_.get(),
                    d_slot_child_.get());
        std::vector<int32_t> ch_pat, ch_item, ch_sup, ch_nent;
        d_ch_pat_.download(ch_pat, n_child);
        d_ch_item_.download(ch_item, n_child);
        d_ch_sup_.download(ch_sup, n_child);
        d_ch_nent_.download(ch_nent, n_child);

        std::vector<int32_t> ch_gid(n_child);
        for (size_t k = 0; k < n_child; ++k)
            ch_gid[k] = add_itemset(b.gid[ch_pat[k]], ch_item[k], ch_sup[k], b.len[ch_pat[k]] + 1);

        // Children whose last item has rank 0 cannot be extended further.
        size_t k = 0;
        while (k < n_child) {
            const size_t lo = k;
            size_t ent = 0, cands = 0;
            while (k < n_child) {
                const size_t e_k = ch_item[k] > 0 ? static_cast<size_t>(ch_nent[k]) : 0;
                const size_t c_k = static_cast<size_t>(ch_item[k]);
                if (k > lo && (ent + e_k > opt_.max_entries || cands + c_k > opt_.max_cands)) break;
                ent += e_k;
                cands += c_k;
                ++k;
            }
            if (ent > 0) emit_group(b, lo, k, ent, ch_pat, ch_item, ch_gid);
        }
    }

    void emit_group(const Batch& b, size_t lo, size_t hi, size_t max_out, const std::vector<int32_t>& ch_pat,
                    const std::vector<int32_t>& ch_item, const std::vector<int32_t>& ch_gid) {
        const int P = static_cast<int>(b.patterns());
        const size_t E = b.entries();
        d_keys_.reserve(max_out);
        d_keys2_.reserve(max_out);
        d_vals_.reserve(max_out);
        d_vals2_.reserve(max_out);
        d_cursor_.zero(1);
        gpu::launch(k_emit_entries, E, E, P, d_ent_off_.get(), d_ent_node_.get(), d_ent_cnt_.get(),
                    d_cand_off_.get(), d_parent_.get(), d_item_.get(), d_slot_child_.get(),
                    static_cast<int32_t>(lo), static_cast<int32_t>(hi), d_keys_.get(), d_vals_.get(),
                    d_cursor_.get());
        const int m = static_cast<int>(d_cursor_.at(0));
        if (m == 0) return;

        prim_.sort_pairs(d_keys_.get(), d_keys2_.get(), d_vals_.get(), d_vals2_.get(), m,
                         32 + bit_width(hi - lo - 1));
        const int u = prim_.reduce_by_key_sum(d_keys2_.get(), d_keys_.get(), d_vals2_.get(), d_vals_.get(), m);
        d_keys_.download(h_keys_, u);
        d_vals_.download(h_vals_, u);

        Batch nb;
        int64_t prev = -1;
        for (int i = 0; i < u; ++i) {
            const int64_t c = static_cast<int64_t>(h_keys_[i] >> 32);
            if (c != prev) {
                if (prev >= 0) nb.close_pattern();
                const size_t k = lo + static_cast<size_t>(c);
                nb.add_pattern(ch_gid[k], ch_item[k], b.len[ch_pat[k]] + 1);
                prev = c;
            }
            nb.node.push_back(static_cast<int32_t>(h_keys_[i] & 0xffffffffu));
            nb.cnt.push_back(h_vals_[i]);
        }
        nb.close_pattern();
        stack_.push_back(std::move(nb));
    }

    void write_output() {
        if (!writing_) return;
        fpg::ItemsetWriter writer(opt_.output, data_, ranking_);
        std::vector<int32_t> items;
        for (size_t g = 0; g < out_item_.size(); ++g) {
            items.clear();
            for (int32_t h = static_cast<int32_t>(g); h >= 0; h = out_parent_[h]) items.push_back(out_item_[h]);
            writer.write(items.data(), static_cast<int>(items.size()), out_sup_[g]);
        }
        writer.close();
    }

    const fpg::Options& opt_;
    const fpg::Dataset& data_;
    fpg::RunStats& st_;
    const bool writing_;
    int min_count_ = 0;
    fpg::ItemRanking ranking_;
    gpu::Primitives prim_;

    // FP-tree (node 0 is the root).
    DevBuf<int32_t> d_parent_, d_item_, d_count_;
    int32_t n_nodes_ = 0;

    // Mining buffers, reused across batches.
    DevBuf<uint32_t> d_ent_off_, d_cand_off_, d_flags_, d_pos_;
    DevBuf<int32_t> d_ent_node_, d_ent_cnt_, d_sup_, d_nent_, d_slot_child_;
    DevBuf<int32_t> d_ch_pat_, d_ch_item_, d_ch_sup_, d_ch_nent_;
    DevBuf<uint64_t> d_keys_, d_keys2_;
    DevBuf<int32_t> d_vals_, d_vals2_;
    DevBuf<unsigned> d_cursor_;
    std::vector<uint64_t> h_keys_;
    std::vector<int32_t> h_vals_;

    std::vector<Batch> stack_;
    std::vector<int32_t> out_parent_, out_item_, out_sup_;
};

}  // namespace

int main(int argc, char** argv) {
    try {
        fpg::Options opt = fpg::parse_args(argc, argv, true);
        fpg::RunStats st;
        st.algorithm = "gpu";
        st.device = gpu::device_name();
        st.input = opt.input;

        fpg::Timer timer;
        fpg::Dataset data = fpg::load_transactions(opt.input);
        st.read_ms = timer.ms();
        st.transactions = data.num_transactions();
        st.distinct_items = data.num_items();
        st.total_items = data.items.size();
        st.min_count = fpg::resolve_min_count(opt.min_support_arg, st.transactions);

        {
            GpuFpGrowth miner(opt, data, st);
            miner.run();
        }
        st.peak_host_bytes = fpg::peak_host_memory_bytes();

        if (!opt.quiet) fpg::print_stats(st, false);
        if (opt.json) fpg::print_stats(st, true);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
