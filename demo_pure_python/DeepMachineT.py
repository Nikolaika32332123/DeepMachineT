# Copyright (c) 2025 Nikolai Tocarev, nikolaos@byte-agi.ru
import random

class DeepMachineT:
    def __init__(self, depth_memory, num_features, layers):
        self.depth_memory = depth_memory
        self.num_features = num_features
        self.layers = layers
        self.len_layer_idx = len(layers)
        self.layer_input_sizes = [num_features, *self.layers[:-1]]
        self.state = self._initialize_state()
    
    def _initialize_state(self):
        state = []
        first_layer = [[random.choice([0, 1]) for _ in range(self.num_features)] for _ in range(self.layers[0])]
        state.append(first_layer)
        for i in range(1, self.len_layer_idx):
            cols = self.layers[i - 1]
            rows = self.layers[i]
            state.append([[random.choice([0, 1]) for _ in range(cols)] for _ in range(rows)])
        return state

    def _apply_noise(self, layer_idx, reward):
        num_weights = self.layer_input_sizes[layer_idx]
        noise_fact_idx_weaken = random.randint(0, self.layers[layer_idx] - 1)
        for weight_idx in range(num_weights):
            old_val = self.state[layer_idx][noise_fact_idx_weaken][weight_idx]
            if old_val == 1:
                delta = random.choice([0, -1])
                new_val = old_val + delta
                self.state[layer_idx][noise_fact_idx_weaken][weight_idx] = max(0, min(1, new_val))
        if reward == -1:
            noise_fact_idx_reinforce = random.randint(0, self.layers[layer_idx] - 1)
            for weight_idx in range(num_weights):
                old_val = self.state[layer_idx][noise_fact_idx_reinforce][weight_idx]
                if old_val == 0:
                    delta = random.choice([1, 0])
                    new_val = old_val + delta
                    self.state[layer_idx][noise_fact_idx_reinforce][weight_idx] = max(0, min(1, new_val))

    def _update(self, layer_idx, fact_idx, pattern, deliberate, reward, feedback):
        for i, pattern_bit in enumerate(pattern):
            current_value = self.state[layer_idx][fact_idx][i]
            new_value = current_value
            if pattern_bit == 1:
                if reward == 1:
                    new_value = current_value + 1
                elif reward == -1:
                    if feedback == 0 or (feedback == 1 and deliberate == 1):
                        new_value = current_value - 1
            elif current_value == 1:
                 new_value = 0
            self.state[layer_idx][fact_idx][i] = max(0, min(self.depth_memory, new_value))
        self._apply_noise(layer_idx, reward)

    def learn(self, X, y, epochs):
        for epoch in range(epochs):
            for x_pred, y_true in zip(X, y):
                y_pred, pattern_trace, reasoning_trace, deliberate_trace = self.predict(x_pred)
                feedback = int(min(deliberate_trace) != max(deliberate_trace))
                reward = 1 if y_pred == y_true else -1
                for layer_idx in range(self.len_layer_idx):
                    self._update(layer_idx, reasoning_trace[layer_idx], pattern_trace[layer_idx], deliberate_trace[layer_idx], reward, feedback)
    
    def predict(self, X_input):
        current_pattern = X_input
        pattern_trace = [X_input]
        reasoning_trace = []
        deliberate_trace = []
        for layer_idx in range(self.len_layer_idx):
            active_fact_id, deliberate = self.predict_fact(current_pattern, layer_idx)
            deliberate_trace.append(deliberate)
            reasoning_trace.append(active_fact_id)
            if layer_idx == self.len_layer_idx - 1:
                return active_fact_id, pattern_trace, reasoning_trace, deliberate_trace
            next_layer_size = self.layers[layer_idx]
            next_pattern = [0] * next_layer_size
            next_pattern[active_fact_id] = 1
            pattern_trace.append(next_pattern)
            current_pattern = next_pattern
    
    def predict_fact(self, pattern, layer_idx):
        candidate_id = 0
        candidate_score = 0
        layer_facts = self.state[layer_idx]
        for fact_idx, fact_weights in enumerate(layer_facts):
            score = 0
            for feature_idx in range(self.layer_input_sizes[layer_idx]):
                weight = fact_weights[feature_idx]
                input_bit = pattern[feature_idx]
                if input_bit == 1 and weight > 0:
                    score += 1
            if score > candidate_score:
                candidate_score = score
                candidate_id = fact_idx
        deliberate = 0
        winner_weights = layer_facts[candidate_id]
        for weight in winner_weights:
            if weight == 1:
                deliberate = 1
                break
        return candidate_id, deliberate