import ctypes
import os
import platform
import json
import struct
import webbrowser
import tempfile
from typing import List, Optional

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

class DMT:
    """Wrapper for the DeepMachineT library"""
    def __init__(self, depth_memory: int, num_features: int, layers: List[int], lib_path: Optional[str] = None):
        self.depth_memory = depth_memory
        self.num_features = num_features
        self.layers = layers
        if not lib_path:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            lib_name = "dmt_lib.dll" if platform.system() == "Windows" else "libdmt.so"
            lib_path = os.path.join(base_dir, "..", "..", "..", "bin", lib_name)
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"Library not found at: {lib_path}")
        self._lib = ctypes.CDLL(lib_path)
        self._handle = None
        self._setup_api()
        self._create_model()

    def _setup_api(self):
        """Setting up C-function signatures."""
        self._lib.DMT_Create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        self._lib.DMT_Create.restype = ctypes.c_void_p
        self._lib.DMT_Destroy.argtypes = [ctypes.c_void_p]
        self._lib.DMT_LearnSL.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
        self._lib.DMT_LearnRL.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        self._lib.DMT_Predict.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
        self._lib.DMT_Predict.restype = ctypes.c_int
        self._lib.DMT_SaveWeights.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self._lib.DMT_SaveWeights.restype = ctypes.c_int
        self._lib.DMT_LoadWeights.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self._lib.DMT_LoadWeights.restype = ctypes.c_int

    def _create_model(self):
        arr_layers = (ctypes.c_int * len(self.layers))(*self.layers)
        self._handle = self._lib.DMT_Create(self.depth_memory, self.num_features, arr_layers, len(self.layers))
        if not self._handle:
            raise RuntimeError("Failed to create DMT model via DLL.")

    def _to_c_array(self, data: List[int]):
        return (ctypes.c_int * len(data))(*data)
    
    def _get_compact_state(self) -> List[List[List[int]]]:
        """
        Get the current state of the model weights as layers[layer][neuron][weights...],
        via save_weights_json / bin_to_json.
        """
        import tempfile
        
        fd, temp_json = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            self.save(temp_json, as_json=True)
            with open(temp_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data["layers"]
        finally:
            if os.path.exists(temp_json):
                os.remove(temp_json)

    def learn_sl(self, X: List[List[int]], y: List[int], epochs: int = 100, verbose: bool = True):
        if len(X) != len(y):
            raise ValueError("Size mismatch between X and y")
        flat_X = [item for sublist in X for item in sublist]
        c_X = self._to_c_array(flat_X)
        c_y = self._to_c_array(y)
        num_samples = len(X)
        iterator = range(epochs)
        if verbose and tqdm:
            iterator = tqdm(iterator, desc="Training")
        elif verbose:
            print(f"Training started for {epochs} epochs...")
        for _ in iterator:
            self._lib.DMT_LearnSL(self._handle, c_X, c_y, num_samples, 1)

    def learn_rl(self, x: List[int], reward: int):
        self._lib.DMT_LearnRL(self._handle, self._to_c_array(x), reward)

    def predict(self, x: List[int]) -> int:
        res = ctypes.c_int()
        if self._lib.DMT_Predict(self._handle, self._to_c_array(x), ctypes.byref(res)) != 0:
            raise RuntimeError("Prediction failed")
        return res.value

    def save(self, path: str, as_json: bool = False):
        """Saves the weights. If as_json=True, converts the binary to JSON."""
        bin_path = path if not as_json else path.replace('.json', '.bin')
        if self._lib.DMT_SaveWeights(self._handle, bin_path.encode()) != 0:
            raise RuntimeError(f"Failed to save to {bin_path}")
        if as_json:
            self._bin_to_json(bin_path, path)
            if os.path.exists(bin_path): 
                os.remove(bin_path)

    def load(self, path: str):
        """Loads weights (supports .bin and .json)."""
        if path.endswith('.json'):
            with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp:
                self._json_to_bin(path, tmp.name)
                tmp_name = tmp.name
            try:
                if self._lib.DMT_LoadWeights(self._handle, tmp_name.encode()) != 0:
                    raise RuntimeError("Failed to load weights from JSON converted bin")
            finally:
                os.remove(tmp_name)
        else:
            if self._lib.DMT_LoadWeights(self._handle, path.encode()) != 0:
                raise RuntimeError(f"Failed to load weights from {path}")

    def evaluate(self, X: List[List[int]], y: List[int]) -> float:
        correct = sum(1 for i, sample in enumerate(X) if self.predict(sample) == y[i])
        return (correct / len(X)) * 100 if X else 0.0

    @staticmethod
    def _bin_to_json(bin_path: str, json_path: str):
        with open(bin_path, 'rb') as f:
            ver = struct.unpack('I', f.read(4))[0]
            d_mem = struct.unpack('B', f.read(1))[0]
            n_feat = struct.unpack('i', f.read(4))[0]
            num_layers = struct.unpack('i', f.read(4))[0]
            layers_sizes = struct.unpack(f'{num_layers}i', f.read(4 * num_layers))
            inputs_sizes = struct.unpack(f'{num_layers}i', f.read(4 * num_layers))
            offsets = struct.unpack(f'{num_layers + 1}Q', f.read(8 * (num_layers + 1)))
            raw_weights = list(f.read())
            layers_data = []
            idx = 0
            for l_size, in_size in zip(layers_sizes, inputs_sizes):
                layer = []
                for _ in range(l_size):
                    w_chunk = raw_weights[idx : idx + in_size]
                    layer.append(w_chunk)
                    idx += in_size
                layers_data.append(layer)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({"architecture": "DeepMachineT", "layers": layers_data}, f, separators=(',', ':'))

    def _json_to_bin(self, json_path: str, bin_path: str):
        with open(json_path, 'r') as f:
            data = json.load(f)["layers"]
        layers_sizes = [len(l) for l in data]
        input_sizes = [len(l[0]) if l else 0 for l in data]
        flat_weights = [w for layer in data for neuron in layer for w in neuron]
        offsets = [0]
        current_offset = 0
        for layer in data:
            for neuron in layer:
                current_offset += len(neuron)
            offsets.append(current_offset)
        with open(bin_path, 'wb') as f:
            f.write(struct.pack('I', 1)) 
            f.write(struct.pack('B', self.depth_memory))
            f.write(struct.pack('i', self.num_features))
            f.write(struct.pack('i', len(data)))
            f.write(struct.pack(f'{len(data)}i', *layers_sizes))
            f.write(struct.pack(f'{len(data)}i', *input_sizes))
            f.write(struct.pack(f'{len(offsets)}Q', *offsets))
            f.write(bytearray(flat_weights))


    def compress_pruning(self, save_path: str):
        """Deletes unused neurons and saves the JSON."""
        fd, tmp_name = tempfile.mkstemp(suffix='.json')
        os.close(fd)
        try:
            self.save(tmp_name, as_json=True)
            with open(tmp_name, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
            layers = data["layers"]
            new_layers = []
            active_indices = None
            stats = []
            for i, layer in enumerate(layers):
                is_last = (i == len(layers) - 1)
                new_layer = []
                current_active = []
                orig_count = len(layer)
                for n_idx, weights in enumerate(layer):
                    if active_indices is not None:
                        weights = [weights[k] for k in active_indices if k < len(weights)]
                    if is_last or any(w > 1 for w in weights):
                        new_layer.append(weights)
                        current_active.append(n_idx)
                new_layers.append(new_layer)
                active_indices = current_active
                new_count = len(new_layer)
                stats.append((orig_count, new_count))
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump({"architecture": "DeepMachineT", "layers": new_layers}, f, separators=(',', ':'))
            print(f"Compressed model saved to {save_path}")
            print("-" * 40)
            print(f"{'Layer':<10} | {'Original':<10} | {'Compressed':<10} | {'Reduction':<10}")
            print("-" * 40)
            for i, (orig, new) in enumerate(stats):
                reduction = 100 * (1 - new / orig) if orig > 0 else 0
                layer_name = f"Layer {i}" if i < len(stats)-1 else "Output"
                print(f"{layer_name:<10} | {orig:<10} | {new:<10} | {reduction:>8.1f}%")
            print("-" * 40)
        finally:
            if os.path.exists(tmp_name):
                try:
                    os.remove(tmp_name)
                except OSError:
                    pass