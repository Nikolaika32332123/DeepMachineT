// Copyright (c) 2025 Nikolai Tocarev, nikolaos@byte-agi.ru
#ifndef DMT_LIB_EXPORTS
#define DMT_LIB_EXPORTS
#endif
#include "include/dmt_lib.h"
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <fstream>
#include <sstream>
#include <map>
#include <tuple>
#include <iomanip>
#include <numeric>
#include <set>
#include <chrono>
#include <cstdint>
#include <cstring>

#ifdef __GNUC__
#pragma GCC optimize("O3,unroll-loops,inline")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")
#endif

// Fast xorshift RNG
class FastRNG {
private:
    uint64_t state;
public:
    using result_type = uint32_t;
    static constexpr result_type min() { return 0; }
    static constexpr result_type max() { return UINT32_MAX; }

    FastRNG(uint64_t seed = 1) : state(seed) {
        if (state == 0) state = 1;
    }

    result_type operator()() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return static_cast<result_type>(state);
    }

    uint32_t rand_range(uint32_t max) {
        if (max == 0) return 0;
        return static_cast<uint32_t>((static_cast<uint64_t>((*this)()) * (static_cast<uint64_t>(max) + 1)) >> 32);
    }

    bool rand_bool() {
        return ((*this)() & 1) != 0;
    }

    void rand_bools(uint8_t* dest, int count) {
        for (int i = 0; i < count; ++i) dest[i] = rand_bool() ? 1 : 0;
    }
};

// ==================== DeepMachineT ====================
class DeepMachineT {
private:
    uint8_t depth_memory;
    int num_features;
    std::vector<int> layers;
    int len_layer_idx;
    std::vector<int> layer_input_sizes;
    std::vector<uint8_t> state;  // linear memory for cache locality
    std::vector<size_t> layer_offsets;
    FastRNG rng;

    inline size_t getIndex(int layer_idx, int fact_idx, int weight_idx) const {
        return layer_offsets[layer_idx] + static_cast<size_t>(fact_idx) * static_cast<size_t>(layer_input_sizes[layer_idx]) + static_cast<size_t>(weight_idx);
    }

    inline uint8_t* getFactWeights(int layer_idx, int fact_idx) {
        return state.data() + getIndex(layer_idx, fact_idx, 0);
    }

    inline const uint8_t* getFactWeights(int layer_idx, int fact_idx) const {
        return state.data() + getIndex(layer_idx, fact_idx, 0);
    }

    void initializeState() {
        size_t total_size = 0;
        layer_offsets.clear();
        layer_offsets.reserve(len_layer_idx + 1);
        layer_offsets.push_back(0);

        for (int i = 0; i < len_layer_idx; ++i) {
            size_t layer_size = static_cast<size_t>(layers[i]) * static_cast<size_t>(layer_input_sizes[i]);
            total_size += layer_size;
            layer_offsets.push_back(total_size);
        }

        state.clear();
        state.resize(total_size);

        size_t idx = 0;

        // First layer: random bits for num_features
        for (int i = 0; i < layers[0]; ++i) {
            for (int j = 0; j < num_features; ++j) state[idx++] = rng.rand_bool() ? 1 : 0;
        }

        // Remaining layers: random bits
        for (int i = 1; i < len_layer_idx; ++i) {
            int cols = layer_input_sizes[i];
            int rows = layers[i];
            for (int r = 0; r < rows; ++r) {
                for (int c = 0; c < cols; ++c) state[idx++] = rng.rand_bool() ? 1 : 0;
            }
        }
    }

    // Apply randomized noise to one weaken and (optionally) one reinforce fact
    inline void applyNoise(int layer_idx, int reward) {
        const int num_weights = layer_input_sizes[layer_idx];
        const int layer_size = layers[layer_idx];

        int noise_fact_idx_weaken = rng.rand_range(layer_size - 1);
        uint8_t* __restrict__ weaken_ptr = getFactWeights(layer_idx, noise_fact_idx_weaken);

        // Weaken: randomly flip some 1s to 0s
        for (int i = 0; i < num_weights; ++i) {
            if (weaken_ptr[i] == 1) weaken_ptr[i] = rng.rand_bool() ? 1 : 0;
        }

        if (reward == -1) {
            int noise_fact_idx_reinforce = rng.rand_range(layer_size - 1);
            uint8_t* __restrict__ reinforce_ptr = getFactWeights(layer_idx, noise_fact_idx_reinforce);

            // Reinforce: randomly flip some 0s to 1s
            for (int i = 0; i < num_weights; ++i) {
                if (reinforce_ptr[i] == 0) reinforce_ptr[i] = rng.rand_bool() ? 1 : 0;
            }
        }
    }

    // update: adjust weights according to pattern, deliberate, reward, feedback
    inline void update(int layer_idx, int fact_idx, const std::vector<int>& pattern,
                int deliberate, int reward, int feedback) {
        const int pattern_size = static_cast<int>(pattern.size());
        const int* __restrict__ pattern_ptr = pattern.data();
        uint8_t* __restrict__ weights_ptr = getFactWeights(layer_idx, fact_idx);
        const int depth_mem = static_cast<int>(depth_memory);

        const bool should_decrease = (reward == -1 && (feedback == 0 || (feedback == 1 && deliberate == 1)));
        const bool should_increase = (reward == 1);

        if (should_increase) {
            // increase counters where pattern==1, clamp to depth_mem. If pattern==0 and weight==1 -> clear to 0.
            for (int i = 0; i < pattern_size; ++i) {
                if (pattern_ptr[i] == 1) {
                    uint8_t val = weights_ptr[i];
                    weights_ptr[i] = static_cast<uint8_t>(val < depth_mem ? val + 1 : depth_mem);
                } else if (weights_ptr[i] == 1) {
                    weights_ptr[i] = 0;
                }
            }
        } else if (should_decrease) {
            // decrease counters where pattern==1, floor at 0. If pattern==0 and weight==1 -> clear to 0.
            for (int i = 0; i < pattern_size; ++i) {
                if (pattern_ptr[i] == 1) {
                    uint8_t val = weights_ptr[i];
                    weights_ptr[i] = static_cast<uint8_t>(val > 0 ? val - 1 : 0);
                } else if (weights_ptr[i] == 1) {
                    weights_ptr[i] = 0;
                }
            }
        } else {
            // default: if pattern==0 and weight==1 -> clear
            for (int i = 0; i < pattern_size; ++i) {
                if (pattern_ptr[i] == 0 && weights_ptr[i] == 1) weights_ptr[i] = 0;
            }
        }

        applyNoise(layer_idx, reward);
    }

public:
    DeepMachineT(int depth_memory_, int num_features_, const std::vector<int>& layers_)
        : depth_memory(static_cast<uint8_t>(depth_memory_)), num_features(num_features_), layers(layers_),
          len_layer_idx(static_cast<int>(layers.size())), rng(std::chrono::steady_clock::now().time_since_epoch().count()) {
        layer_input_sizes.reserve(layers.size());
        layer_input_sizes.push_back(num_features);
        for (size_t i = 0; i < layers.size() - 1; ++i) layer_input_sizes.push_back(layers[i]);
        initializeState();
    }

    void learnSL(const std::vector<std::vector<int>>& X,
               const std::vector<int>& y,
               int epochs) {
        const size_t data_size = X.size();
        for (int epoch = 0; epoch < epochs; ++epoch) {
            for (size_t i = 0; i < data_size; ++i) {
                auto [y_pred, pattern_trace, reasoning_trace, deliberate_trace] = predict(X[i]);

                int feedback = 0;
                // feedback = 1 if deliberate_trace contains values different from the first one
                if (deliberate_trace.size() > 1) {
                    int first_val = deliberate_trace[0];
                    for (size_t k = 1; k < deliberate_trace.size(); ++k) {
                        if (deliberate_trace[k] != first_val) { feedback = 1; break; }
                    }
                }

                int reward = (y_pred == y[i]) ? 1 : -1;

                for (int layer_idx = 0; layer_idx < len_layer_idx; ++layer_idx) {
                    update(layer_idx, reasoning_trace[layer_idx], pattern_trace[layer_idx],
                           deliberate_trace[layer_idx], reward, feedback);
                }
            }
        }
    }

    void learnRL(const std::vector<int>& X_input, int reward) {
        auto [y_pred, pattern_trace, reasoning_trace, deliberate_trace] = predict(X_input);

        int feedback = 0;
        if (deliberate_trace.size() > 1) {
            int first_val = deliberate_trace[0];
            for (size_t k = 1; k < deliberate_trace.size(); ++k) {
                if (deliberate_trace[k] != first_val) { feedback = 1; break; }
            }
        }

        for (int layer_idx = 0; layer_idx < len_layer_idx; ++layer_idx) {
            update(layer_idx, reasoning_trace[layer_idx], pattern_trace[layer_idx],
                   deliberate_trace[layer_idx], reward, feedback);
        }
    }

    // Returns: predicted id, pattern trace, reasoning trace (fact ids), deliberate trace
    std::tuple<int, std::vector<std::vector<int>>, std::vector<int>, std::vector<int>>
    predict(const std::vector<int>& X_input) {
        const std::vector<int>* current_pattern = &X_input;
        std::vector<int> pattern_buffer;

        std::vector<std::vector<int>> pattern_trace;
        pattern_trace.reserve(len_layer_idx + 1);
        pattern_trace.push_back(X_input);

        std::vector<int> reasoning_trace;
        reasoning_trace.reserve(len_layer_idx);

        std::vector<int> deliberate_trace;
        deliberate_trace.reserve(len_layer_idx);

        for (int layer_idx = 0; layer_idx < len_layer_idx; ++layer_idx) {
            auto [active_fact_id, deliberate] = predictFact(*current_pattern, layer_idx);
            deliberate_trace.push_back(deliberate);
            reasoning_trace.push_back(active_fact_id);

            if (layer_idx == len_layer_idx - 1) {
                return std::make_tuple(active_fact_id, pattern_trace, reasoning_trace, deliberate_trace);
            }

            int next_layer_size = layers[layer_idx];
            pattern_buffer.assign(next_layer_size, 0);
            pattern_buffer[active_fact_id] = 1;
            pattern_trace.push_back(pattern_buffer);
            current_pattern = &pattern_trace.back();
        }

        return std::make_tuple(0, pattern_trace, reasoning_trace, deliberate_trace);
    }

    // Predict best fact in a layer and whether the chosen fact contains explicit 1s (deliberate)
    inline std::pair<int, int> predictFact(const std::vector<int>& pattern, int layer_idx) {
        int candidate_id = 0;
        int candidate_score = -1; // start from -1 so zero-score facts can win if first encountered
        const int layer_size = layers[layer_idx];
        const int input_size = layer_input_sizes[layer_idx];
        const int* __restrict__ pattern_ptr = pattern.data();

        for (int fact_idx = 0; fact_idx < layer_size; ++fact_idx) {
            int score = 0;
            const uint8_t* __restrict__ weights_ptr = getFactWeights(layer_idx, fact_idx);
            for (int i = 0; i < input_size; ++i) {
                // count matching pattern bits where weight > 0
                if (pattern_ptr[i] == 1 && weights_ptr[i] > 0) ++score;
            }

            if (score > candidate_score) {
                candidate_score = score;
                candidate_id = fact_idx;
                if (score == input_size) break; // perfect match
            }
        }

        // Check whether the chosen weights contain any explicit 1 (deliberate)
        const uint8_t* __restrict__ chosen_weights = getFactWeights(layer_idx, candidate_id);
        int deliberate = 0;
        for (int i = 0; i < layer_input_sizes[layer_idx]; ++i) {
            if (chosen_weights[i] == 1) { deliberate = 1; break; }
        }

        return std::make_pair(candidate_id, deliberate);
    }

    // Save/load weights to binary file
    bool saveWeights(const char* filename) {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) return false;

        uint32_t version = 1;
        file.write(reinterpret_cast<const char*>(&version), sizeof(version));
        file.write(reinterpret_cast<const char*>(&depth_memory), sizeof(depth_memory));
        file.write(reinterpret_cast<const char*>(&num_features), sizeof(num_features));
        file.write(reinterpret_cast<const char*>(&len_layer_idx), sizeof(len_layer_idx));

        file.write(reinterpret_cast<const char*>(layers.data()), sizeof(int) * len_layer_idx);
        file.write(reinterpret_cast<const char*>(layer_input_sizes.data()), sizeof(int) * len_layer_idx);
        file.write(reinterpret_cast<const char*>(layer_offsets.data()), sizeof(size_t) * (len_layer_idx + 1));
        file.write(reinterpret_cast<const char*>(state.data()), sizeof(uint8_t) * state.size());

        file.close();
        return file.good();
    }

    bool loadWeights(const char* filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) return false;

        uint32_t version;
        file.read(reinterpret_cast<char*>(&version), sizeof(version));
        if (version != 1) { file.close(); return false; }

        uint8_t loaded_depth_memory;
        int loaded_num_features;
        int loaded_len_layer_idx;

        file.read(reinterpret_cast<char*>(&loaded_depth_memory), sizeof(loaded_depth_memory));
        file.read(reinterpret_cast<char*>(&loaded_num_features), sizeof(loaded_num_features));
        file.read(reinterpret_cast<char*>(&loaded_len_layer_idx), sizeof(loaded_len_layer_idx));

        if (loaded_depth_memory != depth_memory || loaded_num_features != num_features || loaded_len_layer_idx != len_layer_idx) {
            file.close(); return false;
        }

        std::vector<int> loaded_layers(loaded_len_layer_idx);
        file.read(reinterpret_cast<char*>(loaded_layers.data()), sizeof(int) * loaded_len_layer_idx);
        for (int i = 0; i < loaded_len_layer_idx; ++i) {
            if (loaded_layers[i] != layers[i]) { file.close(); return false; }
        }

        std::vector<int> loaded_layer_input_sizes(loaded_len_layer_idx);
        file.read(reinterpret_cast<char*>(loaded_layer_input_sizes.data()), sizeof(int) * loaded_len_layer_idx);

        std::vector<size_t> loaded_layer_offsets(loaded_len_layer_idx + 1);
        file.read(reinterpret_cast<char*>(loaded_layer_offsets.data()), sizeof(size_t) * (loaded_len_layer_idx + 1));

        size_t state_size = loaded_layer_offsets[loaded_len_layer_idx];
        state.resize(state_size);
        file.read(reinterpret_cast<char*>(state.data()), sizeof(uint8_t) * state_size);

        layer_input_sizes = std::move(loaded_layer_input_sizes);
        layer_offsets = std::move(loaded_layer_offsets);

        file.close();
        return file.good();
    }

    // getters
    int getNumFeatures() const { return num_features; }
    int getNumLayers() const { return len_layer_idx; }
    int getLayerSize(int layer_idx) const { if (layer_idx >= 0 && layer_idx < len_layer_idx) return layers[layer_idx]; return -1; }
    int getDepthMemory() const { return static_cast<int>(depth_memory); }
};

// ==================== C API wrappers ====================

extern "C" {

DMT_API DMT_Handle DMT_Create(int depth_memory, int num_features, const int* layers, int num_layers) {
    if (!layers || num_layers <= 0) return nullptr;
    std::vector<int> layers_vec(layers, layers + num_layers);
    try {
        DeepMachineT* model = new DeepMachineT(depth_memory, num_features, layers_vec);
        return reinterpret_cast<DMT_Handle>(model);
    } catch (...) {
        return nullptr;
    }
}

DMT_API void DMT_Destroy(DMT_Handle handle) {
    if (handle) delete reinterpret_cast<DeepMachineT*>(handle);
}

DMT_API void DMT_LearnSL(DMT_Handle handle, const int* X, const int* y, int num_samples, int epochs) {
    if (!handle || !X || !y || num_samples <= 0 || epochs <= 0) return;

    DeepMachineT* model = reinterpret_cast<DeepMachineT*>(handle);
    int num_features = model->getNumFeatures();

    std::vector<std::vector<int>> X_vec;
    X_vec.reserve(num_samples);
    std::vector<int> y_vec;
    y_vec.reserve(num_samples);

    for (int i = 0; i < num_samples; ++i) {
        std::vector<int> sample(X + i * num_features, X + (i + 1) * num_features);
        X_vec.push_back(std::move(sample));
        y_vec.push_back(y[i]);
    }

    model->learnSL(X_vec, y_vec, epochs);
}

DMT_API void DMT_LearnRL(DMT_Handle handle, const int* X_input, int reward) {
    if (!handle || !X_input) return;
    DeepMachineT* model = reinterpret_cast<DeepMachineT*>(handle);
    int num_features = model->getNumFeatures();
    std::vector<int> X_vec(X_input, X_input + num_features);
    model->learnRL(X_vec, reward);
}

DMT_API int DMT_Predict(DMT_Handle handle, const int* X_input, int* y_pred) {
    if (!handle || !X_input || !y_pred) return -1;
    DeepMachineT* model = reinterpret_cast<DeepMachineT*>(handle);
    int num_features = model->getNumFeatures();
    std::vector<int> X_vec(X_input, X_input + num_features);
    auto [pred, _, __, ___] = model->predict(X_vec);
    *y_pred = pred;
    return 0;
}

DMT_API int DMT_SaveWeights(DMT_Handle handle, const char* filename) {
    if (!handle || !filename) return -1;
    DeepMachineT* model = reinterpret_cast<DeepMachineT*>(handle);
    return model->saveWeights(filename) ? 0 : -1;
}

DMT_API int DMT_LoadWeights(DMT_Handle handle, const char* filename) {
    if (!handle || !filename) return -1;
    DeepMachineT* model = reinterpret_cast<DeepMachineT*>(handle);
    return model->loadWeights(filename) ? 0 : -1;
}

DMT_API int DMT_GetNumFeatures(DMT_Handle handle) {
    if (!handle) return -1;
    DeepMachineT* model = reinterpret_cast<DeepMachineT*>(handle);
    return model->getNumFeatures();
}

DMT_API int DMT_GetNumLayers(DMT_Handle handle) {
    if (!handle) return -1;
    DeepMachineT* model = reinterpret_cast<DeepMachineT*>(handle);
    return model->getNumLayers();
}

DMT_API int DMT_GetLayerSize(DMT_Handle handle, int layer_idx) {
    if (!handle) return -1;
    DeepMachineT* model = reinterpret_cast<DeepMachineT*>(handle);
    return model->getLayerSize(layer_idx);
}

DMT_API int DMT_GetDepthMemory(DMT_Handle handle) {
    if (!handle) return -1;
    DeepMachineT* model = reinterpret_cast<DeepMachineT*>(handle);
    return model->getDepthMemory();
}

} // extern "C"