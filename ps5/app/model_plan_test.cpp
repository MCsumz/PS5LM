// Host check for model_plan.cpp:
//   c++ -std=c++17 -O1 ps5/app/model_plan_test.cpp ps5/app/model_plan.cpp -o build/model_plan_test
//   build/model_plan_test          synthetic headers only
//   build/model_plan_test models   also the real GGUF files in models/
// Uses the budget the console reported (11.38 GiB free) and the two
// configurations measured there.
// SPDX-License-Identifier: GPL-3.0-or-later
#include "model_plan.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <string>

namespace {

// A GGUF header with string and uint32 metadata, no tensors.
struct Gguf {
    std::string kvs;
    uint64_t    n = 0;
    template <class T> static std::string raw(T v) { return std::string((const char *) &v, sizeof(v)); }
    static std::string str(const std::string & v) { return raw<uint64_t>(v.size()) + v; }
    Gguf & kv(const std::string & k, const std::string & v) { kvs += str(k) + raw<uint32_t>(8) + str(v); n++; return *this; }
    Gguf & kv(const std::string & k, uint32_t v) { kvs += str(k) + raw<uint32_t>(4) + raw(v); n++; return *this; }
    // magic, version 3, tensor count, metadata count, metadata
    std::string bytes() const { return "GGUF" + raw<uint32_t>(3) + raw<uint64_t>(0) + raw(n) + kvs; }
};

bool has_args(const ps5lm::Plan & p, const char * flag, const char * value) {
    auto it = std::find(p.args.begin(), p.args.end(), flag);
    return it != p.args.end() && it + 1 != p.args.end() && *(it + 1) == value;
}

}  // namespace

int main(int argc, char ** argv) {
    const double budget = 11.38;

    // A decision model (Clef is Qwen 3.5 9B with a decision head): found from
    // "<arch>.decision.type", a wide ubatch, a 4k cap and no chat sampling.
    Gguf clef;
    clef.kv("general.architecture", "qwen35").kv("general.name", "Clef Flash Qwen3.5 9B")
        .kv("qwen35.block_count", 32u).kv("qwen35.attention.head_count", 16u)
        .kv("qwen35.attention.head_count_kv", 4u).kv("qwen35.embedding_length", 4096u)
        .kv("qwen35.context_length", 262144u).kv("qwen35.decision.type", "clef");
    const ps5lm::ModelInfo d = ps5lm::read_model_header("clef-flash-Q4_K_M.gguf", clef.bytes(), 5ull << 30);
    assert(d.ok && d.decision == "clef");
    assert(!ps5lm::find_preset(d));
    const ps5lm::Plan dp = ps5lm::plan_model(d, budget);
    assert(dp.fits && dp.ctx == 4096 && has_args(dp, "-ub", "2048") && has_args(dp, "-b", "2048"));
    assert(dp.why.find("decision model (clef)") == 0);

    // The same model without the decision key is a chat model with the narrow batch.
    Gguf chat;
    chat.kv("general.architecture", "qwen35").kv("general.name", "Qwen3.5 9B")
        .kv("qwen35.block_count", 32u).kv("qwen35.attention.head_count", 16u)
        .kv("qwen35.attention.head_count_kv", 4u).kv("qwen35.embedding_length", 4096u)
        .kv("qwen35.context_length", 262144u);
    const ps5lm::ModelInfo c = ps5lm::read_model_header("Qwen3.5-9B-Q4_K_M.gguf", chat.bytes(), 5ull << 30);
    assert(c.ok && c.decision.empty() && ps5lm::find_preset(c));
    const ps5lm::Plan cp = ps5lm::plan_model(c, budget);
    assert(cp.fits && cp.ctx > 4096 && has_args(cp, "-ub", "64"));

    // Headroom: half a GiB left after loading when a cheaper cache buys it,
    // never at the cost of context.
    // A quantized cache also takes flash attention's f16 copy of one layer,
    // 4 heads x (256 + 256) x 64k x 2 bytes, sized at load (patches/0010).
    const uint64_t scratch = ps5lm::fa_scratch_bytes(c, "q8_0", 65536);
    assert(scratch == 256ull << 20 && ps5lm::fa_scratch_bytes(c, "q4_0", 65536) == scratch);
    assert(ps5lm::fa_scratch_bytes(c, "f16", 65536) == 0);
    assert(cp.kv_type == "f16" || cp.scratch_bytes == ps5lm::fa_scratch_bytes(c, cp.kv_type, cp.ctx));
    const double base = cp.weights_gib + cp.state_gib + cp.reserve_gib + scratch / 1073741824.0;
    const double q8 = base + ps5lm::kv_bytes_per_token(c, "q8_0") * 65536 / 1073741824.0;
    const double q4 = base + ps5lm::kv_bytes_per_token(c, "q4_0") * 65536 / 1073741824.0;
    assert(q8 - q4 > 0.1);
    const ps5lm::Plan roomy = ps5lm::plan_model(c, q8 + 0.6);
    assert(roomy.ctx == 65536 && roomy.kv_type == "q8_0");
    const ps5lm::Plan tight = ps5lm::plan_model(c, q8 + 0.1);
    assert(tight.ctx == 65536 && tight.kv_type == "q4_0");
    const ps5lm::Plan tightest = ps5lm::plan_model(c, q4 + 0.05);
    assert(tightest.ctx == 65536 && tightest.kv_type == "q4_0");
    const ps5lm::Plan forced = ps5lm::plan_model(c, q8 + 0.1, 65536, "q8_0");
    assert(forced.ctx == 65536 && forced.kv_type == "q8_0");

    if (argc < 2) {
        std::puts("model_plan_test: ok (synthetic only)");
        return 0;
    }
    bool saw_q2k = false, saw_iq2xxs = false;
    for (const auto & path : ps5lm::find_models({ argv[1] })) {
        const ps5lm::ModelInfo m = ps5lm::read_model_info(path);
        assert(m.ok);
        const ps5lm::Plan p = ps5lm::plan_model(m, budget);
        const ps5lm::Preset * pre = ps5lm::find_preset(m);
        std::printf("%-34s %s L%u kvL%u kvh%u k%u v%u state %.0f MiB | %s | %s\n", m.file_name.c_str(),
                    m.arch.c_str(), m.n_layer, m.n_kv_layers(), m.n_head_kv, m.key_length, m.value_length,
                    m.state_bytes / 1048576.0, p.why.c_str(), pre ? pre->label : "-");
        if (m.arch == "qwen35") {
            // Qwen 3.8 27B: 16 of 64 layers keep a KV cache, 4 heads of 256.
            assert(m.n_layer == 64 && m.n_kv_layers() == 16 && m.n_head_kv == 4 && m.key_length == 256);
            assert(kv_bytes_per_token(m, "f16") == 65536.0);
            assert(m.state_bytes > 140.0 * 1048576 && m.state_bytes < 170.0 * 1048576);
            assert(pre && std::string(pre->label) == "Qwen 3.8");
        }
        if (m.file_name.find("UD-Q2_K_XL") != std::string::npos) {
            // Ran on the console: 64k with a q4_0 cache.
            assert(p.fits && p.ctx == 65536 && p.kv_type == "q4_0");
            saw_q2k = true;
        }
        if (m.file_name.find("UD-IQ2_XXS") != std::string::npos) {
            // 4k at f16 ran; 64k needs at most q8_0 at this size.
            assert(p.fits && p.ctx == 65536 && p.kv_type == "q8_0");
            saw_iq2xxs = true;
        }
    }
    assert(saw_q2k && saw_iq2xxs);
    // Nothing fits in too little memory.
    ps5lm::ModelInfo big;
    big.ok = true; big.file_bytes = 20ull << 30; big.n_layer = 32; big.n_head_kv = 8;
    big.key_length = big.value_length = 128;
    assert(!ps5lm::plan_model(big, budget).fits);
    std::puts("model_plan_test: ok");
}
