"""ai.py - rede neural, população, evolução e (de)serialização em JSON.

Somente NumPy. A população guarda os pesos de TODAS as redes em arrays
empilhados (N, entradas, saídas), então o forward pass de centenas de
carros é um único matmul em lote por camada.
"""
import json
import os
import time

import numpy as np

MODEL_VERSION = 1
OUTPUT_NAMES = ["STEERING", "THROTTLE", "BRAKE", "HANDBRAKE"]
# bias inicial da camada de saída: começa acelerando um pouco, sem freios
OUT_BIAS = np.array([0.0, 1.0, -2.0, -3.0])


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def outputs_to_actions(z, allow_reverse=False):
    """Saídas brutas (..., 4) -> ações.
    STEERING [-1,1] | THROTTLE [0,1] (ou [-1,1] com ré) | BRAKE [0,1] | HANDBRAKE 0/1
    """
    a = np.empty(z.shape)
    a[..., 0] = np.tanh(z[..., 0])
    t = sigmoid(z[..., 1])
    a[..., 1] = 2 * t - 1 if allow_reverse else t
    a[..., 2] = sigmoid(z[..., 2])
    a[..., 3] = (sigmoid(z[..., 3]) > 0.5).astype(float)
    return a


class Network:
    """Uma única rede (MLP, tanh nas ocultas). Usada em WATCH/TEST e nos arquivos."""

    def __init__(self, layers, weights=None, biases=None, allow_reverse=False,
                 meta=None, input_names=None, rng=None):
        self.layers = [int(n) for n in layers]
        if self.layers[-1] != len(OUTPUT_NAMES):
            raise ValueError("a camada de saída precisa ter %d neurônios" % len(OUTPUT_NAMES))
        rng = rng or np.random.default_rng()
        if weights is None:
            weights = [rng.normal(0, 1 / np.sqrt(a), (a, b))
                       for a, b in zip(self.layers[:-1], self.layers[1:])]
            biases = [np.zeros(b) for b in self.layers[1:]]
            biases[-1] = OUT_BIAS.copy()
        self.weights = [np.asarray(w, float) for w in weights]
        self.biases = [np.asarray(b, float) for b in biases]
        self.allow_reverse = bool(allow_reverse)
        self.meta = dict(meta or {})
        self.input_names = list(input_names or [])

    def forward(self, x):
        a = np.asarray(x, float)
        last = len(self.weights) - 1
        for l, (w, b) in enumerate(zip(self.weights, self.biases)):
            a = a @ w + b
            if l < last:
                a = np.tanh(a)
        return outputs_to_actions(a, self.allow_reverse)

    # ---------- JSON ----------
    def to_dict(self):
        last = len(self.weights) - 1
        return {
            "format": "parking_ai_mlp",
            "version": MODEL_VERSION,
            "architecture": self.layers,
            "hidden_activation": "tanh",
            "outputs": OUTPUT_NAMES,
            "output_activations": ["tanh", "sigmoid", "sigmoid", "sigmoid>0.5"],
            "allow_reverse": self.allow_reverse,
            "inputs": self.input_names,
            "meta": self.meta,
            "layers": [
                {"index": l, "inputs": w.shape[0], "neurons": w.shape[1],
                 "activation": "tanh" if l < last else "output",
                 "weights": w.tolist(), "biases": b.tolist()}
                for l, (w, b) in enumerate(zip(self.weights, self.biases))
            ],
        }

    @classmethod
    def from_dict(cls, d):
        if d.get("format") != "parking_ai_mlp":
            raise ValueError("arquivo não é um modelo do PARKING AI")
        if d.get("version", 0) > MODEL_VERSION:
            raise ValueError("modelo de versão mais nova (%s) que este código" % d.get("version"))
        ls = d["layers"]
        return cls(d["architecture"], [l["weights"] for l in ls], [l["biases"] for l in ls],
                   d.get("allow_reverse", False), d.get("meta"), d.get("inputs"))

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)
        return path

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def make_meta(**kw):
    kw.setdefault("created", time.strftime("%Y-%m-%d %H:%M:%S"))
    return kw


class Population:
    """N redes com pesos empilhados + evolução por seleção/mutação."""

    def __init__(self, layers, n, rng=None, allow_reverse=False, input_names=None):
        self.layers = [int(x) for x in layers]
        self.n = int(n)
        self.rng = rng or np.random.default_rng()
        self.allow_reverse = allow_reverse
        self.input_names = list(input_names or [])
        self.W = [self.rng.normal(0, 1 / np.sqrt(a), (self.n, a, b))
                  for a, b in zip(self.layers[:-1], self.layers[1:])]
        self.B = [np.zeros((self.n, b)) for b in self.layers[1:]]
        self.B[-1][:] = OUT_BIAS

    @classmethod
    def seeded(cls, net, n, rate, strength, rng=None, allow_reverse=False):
        """População inicial = cópias (mutadas) de uma rede existente. O índice 0 é idêntico."""
        p = cls(net.layers, n, rng, allow_reverse, net.input_names)
        for l in range(len(net.weights)):
            p.W[l][:] = net.weights[l]
            p.B[l][:] = net.biases[l]
            if n > 1:
                p.W[l][1:] += (p.rng.random(p.W[l][1:].shape) < rate) * p.rng.normal(0, strength, p.W[l][1:].shape)
                p.B[l][1:] += (p.rng.random(p.B[l][1:].shape) < rate) * p.rng.normal(0, strength, p.B[l][1:].shape)
        return p

    def forward(self, x, idx=None):
        """x: (k, entradas). idx: índices das redes correspondentes (None = todas)."""
        a = np.asarray(x, float)
        last = len(self.W) - 1
        for l in range(len(self.W)):
            w = self.W[l] if idx is None else self.W[l][idx]
            b = self.B[l] if idx is None else self.B[l][idx]
            a = np.matmul(a[:, None, :], w)[:, 0, :] + b
            if l < last:
                a = np.tanh(a)
        return outputs_to_actions(a, self.allow_reverse)

    def get_network(self, i, meta=None):
        return Network(self.layers, [w[i].copy() for w in self.W], [b[i].copy() for b in self.B],
                       self.allow_reverse, meta, self.input_names)

    def evolve(self, fitness, elite_pct=0.05, rate=0.1, strength=0.3, random_fraction=0.03):
        """Mantém a elite intacta (primeiras posições) e preenche o resto com
        filhos mutados de pais sorteados entre a elite; uma fração pequena é
        substituída por redes novas (diversidade)."""
        fitness = np.asarray(fitness, float)
        order = np.argsort(-fitness)
        ne = int(np.clip(round(self.n * elite_pct), 1, self.n))
        elite = order[:ne]
        nc = self.n - ne
        if nc <= 0:
            return
        parents = self.rng.choice(elite, size=nc)
        for l in range(len(self.W)):
            cw, cb = self.W[l][parents], self.B[l][parents]
            cw = cw + (self.rng.random(cw.shape) < rate) * self.rng.normal(0, strength, cw.shape)
            cb = cb + (self.rng.random(cb.shape) < rate) * self.rng.normal(0, strength, cb.shape)
            self.W[l] = np.concatenate([self.W[l][elite], cw])
            self.B[l] = np.concatenate([self.B[l][elite], cb])
        nr = min(int(self.n * random_fraction), nc)
        if nr > 0:
            fresh = Population(self.layers, nr, self.rng, self.allow_reverse)
            for l in range(len(self.W)):
                self.W[l][-nr:] = fresh.W[l]
                self.B[l][-nr:] = fresh.B[l]
