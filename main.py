"""main.py - PARKING AI: simulador, física, pista, editor, treino e interface.

Uso:
    python main.py                      # abre a interface (menu)
    python main.py --train --track simples.json --pop 500 --gens 100 --hidden 16,16
                                        # treino HEADLESS pelo terminal (sem janela)

Convenções: coordenadas em pixels, eixo Y para baixo. Ângulo 0 = apontando para a
direita; ângulos crescem no sentido horário (na tela). Todo retângulo (vaga, carro,
obstáculo, parede) é {x, y, l, w, rot}: l = comprimento ao longo de `rot`, w = largura.
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import pygame

import ai

BASE = os.path.dirname(os.path.abspath(__file__))
TRACKS_DIR = os.path.join(BASE, "tracks")
MODELS_DIR = os.path.join(BASE, "models")
CONFIG_PATH = os.path.join(BASE, "config.json")

# ----------------------------------------------------------------- constantes
DT = 1 / 30                                   # passo de física (s)
CAR_L, CAR_W = 44.0, 22.0
WHEELBASE = 28.0
MAX_SPEED, MAX_REV = 170.0, 60.0
ACCEL, BRAKE_DEC, HANDBRAKE_DEC = 220.0, 520.0, 300.0
ROLL_FRICTION, DRAG = 40.0, 0.15
MAX_STEER, STEER_RATE = 0.55, 2.5             # rad, rad/s
SENSOR_RANGE = 160.0
SENSOR_NAMES = ["frente", "frente-esq", "frente-dir", "esquerda", "direita",
                "traseira", "tras-esq", "tras-dir"]
SENSOR_ANGLES = np.radians([0, -45, 45, -90, 90, 180, -135, 135])
_ca, _sa = np.abs(np.cos(SENSOR_ANGLES)), np.abs(np.sin(SENSOR_ANGLES))
SENSOR_EDGE = np.minimum(CAR_L / 2 / np.maximum(_ca, 1e-9), CAR_W / 2 / np.maximum(_sa, 1e-9))

# ---- Camadas de informação (ver seção 12 do projeto) ----
# A) conhecida diretamente pela IA: estado do carro + dados matemáticos da vaga
# B) descoberta por sensores: paredes / carros / obstáculos
# C) somente visual (track.visual): NUNCA entra nas entradas da rede
INPUT_NAMES = (["velocidade", "vel_angular", "cos_orientacao", "sin_orientacao",
                "dist_vaga", "sin_ang_vaga", "cos_ang_vaga", "sin_dif_orient", "cos_dif_orient",
                "dist_lateral", "dist_longitudinal"]
               + ["sensor_" + n for n in SENSOR_NAMES])
NIN = len(INPUT_NAMES)

MAX_STEPS = 900                # 30 s
STALL_STEPS = 180              # 6 s sem progresso
PARK_FRAMES = 15               # condições mantidas por 0,5 s
PARK_DIST, PARK_ANG, PARK_SPEED, PARK_TOL = 10.0, math.radians(12), 6.0, 2.0

DEFAULT_SIM = dict(car_pen=500.0, obs_pen=500.0, out_pen=300.0, park_reward=1000.0,
                   enter_reward=50.0, align_reward=100.0, stop_reward=50.0,
                   k_approach=0.5, k_align=40.0, time_pen=0.02, stall_pen=20.0)

DEFAULT_CFG = dict(track="simples.json", model="", hidden="16,16", pop=300, mut_rate=0.10,
                   mut_strength=0.30, elite=0.05, max_gen=500, reverse=False,
                   car_pen=500, obs_pen=500, out_pen=300, resume=False, headless=False)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ----------------------------------------------------------------- geometria
def rect_corners(cx, cy, l, w, rot):
    """Cantos (…,4,2) de retângulos. rot em radianos. Aceita escalares ou arrays."""
    cx, cy, rot = np.asarray(cx, float), np.asarray(cy, float), np.asarray(rot, float)
    c, s = np.cos(rot), np.sin(rot)
    fx, fy, rx, ry = c * l / 2, s * l / 2, -s * w / 2, c * w / 2
    pts = [(cx + fx + rx, cy + fy + ry), (cx + fx - rx, cy + fy - ry),
           (cx - fx - rx, cy - fy - ry), (cx - fx + rx, cy - fy + ry)]
    return np.stack([np.stack(p, -1) for p in pts], -2)


def make_solid(kind, o):
    rot = math.radians(o.get("rot", 0))
    c, s = math.cos(rot), math.sin(rot)
    return dict(kind=kind, cx=o["x"], cy=o["y"], c=c, s=s, hl=o["l"] / 2, hw=o["w"] / 2,
                corners=rect_corners(o["x"], o["y"], o["l"], o["w"], rot),
                ax0=np.array([c, s]), ax1=np.array([-s, c]))


def sat_hit(cc, a0, a1, rc, r0, r1):
    """Teorema dos eixos separadores: k retângulos (cc:(k,4,2)) contra 1 retângulo fixo."""
    k = len(cc)
    hit = np.ones(k, bool)
    for ax in (a0, a1, np.broadcast_to(r0, (k, 2)), np.broadcast_to(r1, (k, 2))):
        pc = cc[..., 0] * ax[:, None, 0] + cc[..., 1] * ax[:, None, 1]
        pr = rc[None, :, 0] * ax[:, None, 0] + rc[None, :, 1] * ax[:, None, 1]
        hit &= (pc.max(1) >= pr.min(1)) & (pr.max(1) >= pc.min(1))
    return hit


def ray_obb(o, d, s):
    """Distância de raios (o,d) até um retângulo; inf se não acerta."""
    px, py = o[:, 0] - s["cx"], o[:, 1] - s["cy"]
    lx, ly = px * s["c"] + py * s["s"], -px * s["s"] + py * s["c"]
    dx = d[:, 0] * s["c"] + d[:, 1] * s["s"]
    dy = -d[:, 0] * s["s"] + d[:, 1] * s["c"]
    dx = np.where(np.abs(dx) < 1e-9, 1e-9, dx)
    dy = np.where(np.abs(dy) < 1e-9, 1e-9, dy)
    t1, t2 = (-s["hl"] - lx) / dx, (s["hl"] - lx) / dx
    t3, t4 = (-s["hw"] - ly) / dy, (s["hw"] - ly) / dy
    tmin = np.maximum(np.minimum(t1, t2), np.minimum(t3, t4))
    tmax = np.minimum(np.maximum(t1, t2), np.maximum(t3, t4))
    hit = (tmax >= np.maximum(tmin, 0))
    return np.where(hit, np.maximum(tmin, 0), np.inf)


# ----------------------------------------------------------------- pista
class Track:
    def __init__(self, d=None):
        d = d or {}
        self.name = d.get("name", "nova")
        self.size = list(d.get("size", [1000, 700]))
        self.slot = dict(x=700, y=350, l=64, w=34, rot=0)
        self.slot.update(d.get("slot", {}))
        self.start = dict(x=120, y=350, rot=0)
        self.start.update(d.get("start", {}))
        self.cars = [dict(c) for c in d.get("cars", [])]
        self.obstacles = [dict(c) for c in d.get("obstacles", [])]
        self.walls = [dict(c) for c in d.get("walls", [])]
        self.visual = [dict(c) for c in d.get("visual", [])]   # SOMENTE VISUAL

    def to_dict(self):
        return {"format": "parking_ai_track", "version": 1, "name": self.name, "size": self.size,
                "slot": self.slot,
                "start": {k: self.start[k] for k in ("x", "y", "rot")},
                "cars": self.cars, "obstacles": self.obstacles, "walls": self.walls,
                "visual": [dict(v, visual_only=True) for v in self.visual]}

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=1)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            t = cls(json.load(f))
        if not t.name or t.name == "nova":
            t.name = os.path.splitext(os.path.basename(path))[0]
        return t


def slot_lines(s, color=(240, 240, 240)):
    """Linhas brancas decorativas (visuais) ao redor de uma vaga."""
    c = rect_corners(s["x"], s["y"], s["l"], s["w"], math.radians(s["rot"]))
    return [{"kind": "line", "p1": c[i].tolist(), "p2": c[(i + 1) % 4].tolist(),
             "color": list(color), "width": 3, "visual_only": True} for i in range(4)]


def make_track(name, slot, start, cars=(), walls=(), obstacles=()):
    t = Track({"name": name, "slot": slot, "start": start})
    for x, y, rot in cars:
        t.cars.append(dict(x=x, y=y, l=CAR_L, w=CAR_W, rot=rot))
    for x, y, l, w, rot in walls:
        t.walls.append(dict(x=x, y=y, l=l, w=w, rot=rot))
    for kind, x, y, l, w, rot in obstacles:
        t.obstacles.append(dict(kind=kind, x=x, y=y, l=l, w=w, rot=rot))
    t.visual = slot_lines(slot)
    return t


def ensure_default_tracks():
    os.makedirs(TRACKS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    if glob.glob(os.path.join(TRACKS_DIR, "*.json")):
        return
    perp = lambda w: dict(x=500, y=560, l=64, w=w, rot=90)
    edge = lambda w, gap: (w / 2 + CAR_W / 2 + gap)
    tracks = [
        make_track("simples", dict(x=700, y=350, l=64, w=34, rot=0), dict(x=120, y=300, rot=0)),
        make_track("vaga_entre_carros", perp(34), dict(x=250, y=330, rot=0),
                   cars=[(500 - edge(34, 4), 560, 90), (500 + edge(34, 4), 560, 90)],
                   walls=[(500, 612, 300, 12, 0)]),
        make_track("vaga_apertada", perp(28), dict(x=250, y=330, rot=0),
                   cars=[(500 - edge(28, 2), 560, 90), (500 + edge(28, 2), 560, 90)],
                   walls=[(500, 612, 300, 12, 0)]),
        make_track("baliza", dict(x=500, y=590, l=72, w=32, rot=0), dict(x=200, y=470, rot=0),
                   cars=[(500 - 36 - 22 - 4, 590, 0), (500 + 36 + 22 + 4, 590, 0)],
                   walls=[(500, 632, 500, 12, 0)],
                   obstacles=[("cone", 350, 520, 14, 14, 0)]),
    ]
    for t in tracks:
        t.save(os.path.join(TRACKS_DIR, t.name + ".json"))


def default_track():
    t = make_track("nova", dict(x=700, y=350, l=64, w=34, rot=0), dict(x=120, y=350, rot=0))
    return t


# ----------------------------------------------------------------- simulação (vetorizada)
class Sim:
    """Simula n carros ao mesmo tempo na mesma pista (arrays NumPy)."""

    def __init__(self, track, n, allow_reverse=False, cfg=None):
        self.t, self.n, self.allow_reverse = track, int(n), bool(allow_reverse)
        self.cfg = dict(DEFAULT_SIM)
        self.cfg.update(cfg or {})
        self.W, self.H = track.size
        s = track.slot
        self.sx, self.sy, self.srot, self.SL, self.SW = s["x"], s["y"], math.radians(s["rot"]), s["l"], s["w"]
        self.solids = ([make_solid("car", o) for o in track.cars]
                       + [make_solid("obs", o) for o in track.obstacles + track.walls])
        self.reset()

    def reset(self):
        n, st = self.n, self.t.start
        f = lambda v: np.full(n, float(v))
        z = lambda: np.zeros(n)
        b = lambda: np.zeros(n, bool)
        self.x, self.y, self.h = f(st["x"]), f(st["y"]), f(math.radians(st["rot"]))
        self.v, self.steer, self.av = z(), z(), z()
        self.alive = np.ones(n, bool)
        self.fitness = z()
        self.steps = 0
        self.park_cnt = np.zeros(n, int)
        self.stall = np.zeros(n, int)
        self.entered, self.aligned, self.stopped = b(), b(), b()
        self.parked, self.hit_car, self.hit_obs, self.out, self.stalled = b(), b(), b(), b(), b()
        self.park_time = np.full(n, np.nan)
        d = np.hypot(self.sx - self.x, self.sy - self.y)
        ae = np.abs(wrap(self.srot - self.h))
        self.prev_dist, self.prev_aerr = d, ae
        self.best_m = d + 40 * ae

    # ---- sensores e observação
    def sense(self, idx):
        k = len(idx)
        ang = self.h[idx][:, None] + SENSOR_ANGLES[None, :]
        d = np.stack([np.cos(ang), np.sin(ang)], -1).reshape(-1, 2)
        o = np.repeat(np.stack([self.x[idx], self.y[idx]], -1), 8, axis=0) + d * np.tile(SENSOR_EDGE, k)[:, None]
        t = np.full(k * 8, SENSOR_RANGE)
        for s in self.solids:
            t = np.minimum(t, ray_obb(o, d, s))
        dx = np.where(np.abs(d[:, 0]) < 1e-9, 1e-9, d[:, 0])
        dy = np.where(np.abs(d[:, 1]) < 1e-9, 1e-9, d[:, 1])
        tx = np.where(dx > 0, (self.W - o[:, 0]) / dx, -o[:, 0] / dx)
        ty = np.where(dy > 0, (self.H - o[:, 1]) / dy, -o[:, 1] / dy)
        t = np.minimum(t, np.maximum(np.minimum(tx, ty), 0))
        return t.reshape(k, 8), o.reshape(k, 8, 2), d.reshape(k, 8, 2)

    def observe(self, idx):
        x, y, h, v, av = self.x[idx], self.y[idx], self.h[idx], self.v[idx], self.av[idx]
        c, s = np.cos(h), np.sin(h)
        dx, dy = self.sx - x, self.sy - y
        dist = np.hypot(dx, dy)
        lon = dx * c + dy * s
        lat = -dx * s + dy * c
        ang = np.arctan2(lat, lon)
        dh = wrap(self.srot - h)
        sens = np.clip(self.sense(idx)[0] / SENSOR_RANGE, 0, 1)
        cols = [v / MAX_SPEED, np.clip(av / 3, -1, 1), c, s, np.tanh(dist / 150),
                np.sin(ang), np.cos(ang), np.sin(dh), np.cos(dh),
                np.tanh(lat / 100), np.tanh(lon / 100)]
        return np.column_stack(cols + [sens])

    # ---- passo
    def step(self, act):
        ia = np.flatnonzero(self.alive)
        if ia.size == 0:
            return
        cfg = self.cfg
        self.steps += 1
        a = act[ia]
        x, y, h, v, st = self.x[ia], self.y[ia], self.h[ia], self.v[ia], self.steer[ia]
        # ---- física (modelo de bicicleta simplificado)
        tgt = np.clip(a[:, 0], -1, 1) * MAX_STEER / (1 + (v / 110) ** 2)
        st = st + np.clip(tgt - st, -STEER_RATE * DT, STEER_RATE * DT)
        thr = np.clip(a[:, 1], -1 if self.allow_reverse else 0, 1)
        brk = np.clip(a[:, 2], 0, 1)
        hb = a[:, 3] > 0.5
        v = v + thr * ACCEL * DT
        dec = (ROLL_FRICTION + DRAG * np.abs(v) + brk * BRAKE_DEC + hb * HANDBRAKE_DEC) * DT
        v = np.sign(v) * np.maximum(np.abs(v) - dec, 0)
        v = np.clip(v, -MAX_REV, MAX_SPEED)
        av = v / WHEELBASE * np.tan(st) * np.where(hb & (np.abs(v) > 20), 1.6, 1.0)
        h = h + av * DT
        x = x + np.cos(h) * v * DT
        y = y + np.sin(h) * v * DT
        # ---- colisões
        cc = rect_corners(x, y, CAR_L, CAR_W, h)
        a0 = np.stack([np.cos(h), np.sin(h)], -1)
        a1 = np.stack([-np.sin(h), np.cos(h)], -1)
        hc, ho = np.zeros(len(ia), bool), np.zeros(len(ia), bool)
        for s in self.solids:
            hit = sat_hit(cc, a0, a1, s["corners"], s["ax0"], s["ax1"])
            if s["kind"] == "car":
                hc |= hit
            else:
                ho |= hit
        out = ((cc[..., 0] < 0) | (cc[..., 0] > self.W) | (cc[..., 1] < 0) | (cc[..., 1] > self.H)).any(1)
        # ---- geometria em relação à vaga
        dx, dy = x - self.sx, y - self.sy
        c, s = math.cos(self.srot), math.sin(self.srot)
        lon, lat = dx * c + dy * s, -dx * s + dy * c
        dist = np.hypot(dx, dy)
        aerr = np.abs(wrap(self.srot - h))
        speed = np.abs(v)
        inside_c = (np.abs(lon) <= self.SL / 2) & (np.abs(lat) <= self.SW / 2)
        ccx, ccy = cc[..., 0] - self.sx, cc[..., 1] - self.sy
        clon, clat = ccx * c + ccy * s, -ccx * s + ccy * c
        inside_full = ((np.abs(clon) <= self.SL / 2 + PARK_TOL) & (np.abs(clat) <= self.SW / 2 + PARK_TOL)).all(1)
        # ---- recompensas
        f = self.fitness[ia].copy()
        f += cfg["k_approach"] * (self.prev_dist[ia] - dist)
        prox = np.clip(1 - dist / 150, 0, 1)
        f += cfg["k_align"] * prox * (self.prev_aerr[ia] - aerr)
        ent, ali, sto = self.entered[ia], self.aligned[ia], self.stopped[ia]
        n_ent = inside_c & ~ent
        n_ali = inside_c & (aerr < 0.2) & ~ali
        n_sto = inside_c & (aerr < 0.2) & (speed < 8) & ~sto
        f += cfg["enter_reward"] * n_ent + cfg["align_reward"] * n_ali + cfg["stop_reward"] * n_sto
        self.entered[ia], self.aligned[ia], self.stopped[ia] = ent | n_ent, ali | n_ali, sto | n_sto
        # penalidades pequenas
        f -= cfg["time_pen"]
        f -= np.where((dist < 100) & (speed > 80), (speed - 80) * 0.01, 0)      # velocidade excessiva
        f -= np.where(inside_c, 0.005 * speed, 0)                               # movimento desnecessário
        f -= 0.03 * np.minimum(np.clip(a[:, 1], 0, 1), brk) + 0.002 * np.abs(av)
        # ---- estacionamento válido
        ok = inside_full & (dist < PARK_DIST) & (aerr < PARK_ANG) & (speed < PARK_SPEED)
        cnt = np.where(ok, self.park_cnt[ia] + 1, 0)
        collided = hc | ho | out
        parked = (cnt >= PARK_FRAMES) & ~collided
        f += parked * (cfg["park_reward"] + 200 * (1 - self.steps / MAX_STEPS))
        # ---- penalidades de colisão
        f -= hc * cfg["car_pen"] + (ho & ~hc) * cfg["obs_pen"] + (out & ~hc & ~ho) * cfg["out_pen"]
        # ---- travamento
        m = dist + 40 * aerr
        improved = m < self.best_m[ia] - 1
        self.best_m[ia] = np.where(improved, m, self.best_m[ia])
        stall = np.where(improved, 0, self.stall[ia] + 1)
        stalled = (stall >= STALL_STEPS) & ~parked & ~collided
        f -= stalled * cfg["stall_pen"]
        # ---- grava estado
        self.x[ia], self.y[ia], self.h[ia], self.v[ia], self.steer[ia], self.av[ia] = x, y, h, v, st, av
        self.fitness[ia] = f
        self.prev_dist[ia], self.prev_aerr[ia] = dist, aerr
        self.park_cnt[ia], self.stall[ia] = cnt, stall
        self.hit_car[ia] |= hc
        self.hit_obs[ia] |= ho & ~hc
        self.out[ia] |= out & ~hc & ~ho
        self.stalled[ia] |= stalled
        self.parked[ia] |= parked
        self.park_time[ia[parked]] = self.steps * DT
        dead = collided | parked | stalled
        if self.steps >= MAX_STEPS:
            dead[:] = True
        self.alive[ia[dead]] = False

    def reason(self, i):
        if self.parked[i]: return "ESTACIONOU"
        if self.hit_car[i]: return "colisão com carro"
        if self.hit_obs[i]: return "colisão com obstáculo"
        if self.out[i]: return "saiu da área"
        if self.stalled[i]: return "sem progresso"
        return "tempo esgotado" if not self.alive[i] else "em andamento"


# ----------------------------------------------------------------- treinador
class Trainer:
    def __init__(self, track, cfg, resume_net=None, log=None):
        self.track, self.cfg, self.log = track, cfg, log
        n = int(cfg["pop"])
        rng = np.random.default_rng()
        if resume_net is not None:
            if resume_net.layers[0] != NIN:
                raise ValueError("modelo tem %d entradas, o simulador usa %d" % (resume_net.layers[0], NIN))
            self.pop = ai.Population.seeded(resume_net, n, cfg["mut_rate"], cfg["mut_strength"], rng,
                                            resume_net.allow_reverse)
            reverse = resume_net.allow_reverse
        else:
            hidden = parse_hidden(cfg["hidden"])
            self.pop = ai.Population([NIN] + hidden + [4], n, rng, cfg["reverse"], INPUT_NAMES)
            reverse = cfg["reverse"]
        self.pop.input_names = list(INPUT_NAMES)
        self.sim_cfg = dict(car_pen=cfg["car_pen"], obs_pen=cfg["obs_pen"], out_pen=cfg["out_pen"])
        self.sim = Sim(track, n, reverse, self.sim_cfg)
        self.gen, self.finished = 1, False
        self.history, self.stats = [], None
        self.best_fit, self.best_net, self.gen_best_net = -1e18, None, None
        self.saved_auto = None

    def step(self):
        ia = np.flatnonzero(self.sim.alive)
        if ia.size == 0:
            return False
        act = np.zeros((self.sim.n, 4))
        act[ia] = self.pop.forward(self.sim.observe(ia), ia)
        self.sim.step(act)
        return True

    def run(self, budget):
        """Roda passos por `budget` segundos; encerra gerações quando terminam."""
        t0 = time.time()
        while not self.finished and time.time() - t0 < budget:
            if not self.step():
                self.end_generation()

    def leader(self):
        s = self.sim
        if s.alive.any():
            return int(np.argmax(np.where(s.alive, s.fitness, -1e18)))
        return int(np.argmax(s.fitness))

    def end_generation(self):
        s, f = self.sim, self.sim.fitness
        bi = int(np.argmax(f))
        meta = ai.make_meta(generation=self.gen, fitness=float(f[bi]), track=self.track.name,
                            parked=bool(s.parked[bi]), population=self.sim.n,
                            reverse=bool(self.sim.allow_reverse))
        self.gen_best_net = self.pop.get_network(bi, meta)
        pt = s.park_time[s.parked]
        self.stats = dict(gen=self.gen, best=float(f.max()), mean=float(f.mean()),
                          best_time=float(pt.min()) if pt.size else None,
                          park_rate=float(s.parked.mean()),
                          collisions=int((s.hit_car | s.hit_obs).sum()),
                          car_collisions=int(s.hit_car.sum()), out=int(s.out.sum()))
        self.history.append((self.stats["best"], self.stats["mean"], self.stats["park_rate"]))
        if f.max() > self.best_fit:
            self.best_fit, self.best_net = float(f.max()), self.gen_best_net
            self.saved_auto = self.best_net.save(os.path.join(MODELS_DIR, "best_auto.json"))
        if self.log:
            self.log(self.stats)
        if self.gen >= self.cfg["max_gen"]:
            self.finished = True
            return
        self.pop.evolve(f, self.cfg["elite"], self.cfg["mut_rate"], self.cfg["mut_strength"])
        self.sim.reset()
        self.gen += 1


def parse_hidden(s):
    try:
        h = [int(v) for v in str(s).replace(" ", "").split(",") if v]
    except ValueError:
        raise ValueError("camadas ocultas inválidas: use algo como 16,16")
    if not h or any(v < 1 or v > 512 for v in h):
        raise ValueError("camadas ocultas inválidas: use algo como 16,16")
    return h


def next_model_path(prefix="parking_ai_v"):
    i = 1
    while os.path.exists(os.path.join(MODELS_DIR, "%s%d.json" % (prefix, i))):
        i += 1
    return os.path.join(MODELS_DIR, "%s%d.json" % (prefix, i))


def load_cfg():
    cfg = dict(DEFAULT_CFG)
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg.update({k: v for k, v in json.load(f).items() if k in cfg})
    except (OSError, ValueError):
        pass
    return cfg


def save_cfg(cfg):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=1)
    except OSError:
        pass


def list_files(d):
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(d, "*.json")))


# ----------------------------------------------------------------- desenho
class View:
    def __init__(self, rect, size):
        self.rect = pygame.Rect(rect)
        self.set_size(size)

    def set_size(self, size):
        tw, th = size
        self.s = min(self.rect.w / tw, self.rect.h / th)
        self.ox = self.rect.x + (self.rect.w - tw * self.s) / 2
        self.oy = self.rect.y + (self.rect.h - th * self.s) / 2

    def tp(self, x, y):
        return (self.ox + x * self.s, self.oy + y * self.s)

    def inv(self, mx, my):
        return ((mx - self.ox) / self.s, (my - self.oy) / self.s)


FONT = {}
def font(size=15):
    if size not in FONT:
        FONT[size] = pygame.font.SysFont("dejavusansmono,consolas,menlo,couriernew", size)
    return FONT[size]


def txt(surf, s, x, y, color=(225, 225, 225), size=15):
    surf.blit(font(size).render(str(s), True, color), (x, y))


CAR_COLORS = [(200, 70, 70), (70, 120, 200), (200, 170, 60), (140, 90, 190), (80, 170, 120), (190, 190, 190)]
VIS_COLORS = [(240, 240, 240), (240, 210, 60), (90, 150, 240), (230, 80, 80), (90, 200, 110), (255, 140, 40)]


def poly(surf, view, pts, color, width=0):
    pygame.draw.polygon(surf, color, [view.tp(*p) for p in pts], width)


def draw_car(surf, view, x, y, h, color, l=CAR_L, w=CAR_W, outline=None):
    poly(surf, view, rect_corners(x, y, l, w, h), color)
    poly(surf, view, rect_corners(x + math.cos(h) * l * 0.18, y + math.sin(h) * l * 0.18, l * 0.28, w * 0.78, h),
         (30, 40, 55))
    poly(surf, view, rect_corners(x, y, l, w, h), outline or (15, 15, 15), 1)
    fx, fy = x + math.cos(h) * l / 2, y + math.sin(h) * l / 2
    pygame.draw.circle(surf, (255, 240, 150), view.tp(fx, fy), max(2, int(2.5 * view.s)))


def draw_arrow(surf, view, p1, p2, color, width):
    a, b = view.tp(*p1), view.tp(*p2)
    pygame.draw.line(surf, color, a, b, max(1, int(width * view.s)))
    ang = math.atan2(b[1] - a[1], b[0] - a[0])
    hs = max(8, 10 * view.s + width)
    pts = [b, (b[0] - hs * math.cos(ang - 0.4), b[1] - hs * math.sin(ang - 0.4)),
           (b[0] - hs * math.cos(ang + 0.4), b[1] - hs * math.sin(ang + 0.4))]
    pygame.draw.polygon(surf, color, pts)


def draw_track(surf, tr, view, show_start=False):
    surf.fill((22, 24, 28))
    W, H = tr.size
    pygame.draw.rect(surf, (58, 60, 66), (*view.tp(0, 0), W * view.s, H * view.s))
    # --- camada SOMENTE VISUAL (a IA nunca vê isto)
    for v in tr.visual:
        col = tuple(v.get("color", (240, 240, 240)))
        if v["kind"] == "area":
            pygame.draw.rect(surf, col, (*view.tp(v["x"], v["y"]), v["w"] * view.s, v["h"] * view.s))
        elif v["kind"] == "line":
            pygame.draw.line(surf, col, view.tp(*v["p1"]), view.tp(*v["p2"]), max(1, int(v.get("width", 3) * view.s)))
        elif v["kind"] == "arrow":
            draw_arrow(surf, view, v["p1"], v["p2"], col, v.get("width", 3))
    # --- vaga (dado matemático real usado pela IA)
    s = tr.slot
    sc = rect_corners(s["x"], s["y"], s["l"], s["w"], math.radians(s["rot"]))
    poly(surf, view, sc, (62, 92, 70))
    poly(surf, view, sc, (90, 220, 110), 1)
    r = math.radians(s["rot"])
    draw_arrow(surf, view, (s["x"] - math.cos(r) * s["l"] * 0.25, s["y"] - math.sin(r) * s["l"] * 0.25),
               (s["x"] + math.cos(r) * s["l"] * 0.25, s["y"] + math.sin(r) * s["l"] * 0.25), (90, 220, 110), 1.5)
    # --- sólidos
    for w_ in tr.walls:
        poly(surf, view, rect_corners(w_["x"], w_["y"], w_["l"], w_["w"], math.radians(w_["rot"])), (130, 130, 140))
    for o in tr.obstacles:
        c = rect_corners(o["x"], o["y"], o["l"], o["w"], math.radians(o["rot"]))
        col = {"cone": (255, 130, 30), "barrier": (240, 200, 40)}.get(o.get("kind"), (150, 105, 70))
        poly(surf, view, c, col)
        poly(surf, view, c, (20, 20, 20), 1)
    for i, c in enumerate(tr.cars):
        draw_car(surf, view, c["x"], c["y"], math.radians(c["rot"]), CAR_COLORS[i % len(CAR_COLORS)], c["l"], c["w"])
    if show_start:
        st = tr.start
        draw_car(surf, view, st["x"], st["y"], math.radians(st["rot"]), (60, 150, 230), CAR_L, CAR_W, (255, 255, 255))
    pygame.draw.rect(surf, (200, 200, 210), (*view.tp(0, 0), W * view.s, H * view.s), 2)


def draw_sim_car(surf, view, sim, i, color=(60, 150, 230), sensors=False):
    if sensors:
        d, o, dirs = sim.sense(np.array([i]))
        for k in range(8):
            end = o[0, k] + dirs[0, k] * d[0, k]
            fr = d[0, k] / SENSOR_RANGE
            col = (int(255 * (1 - fr)), int(220 * fr), 60)
            pygame.draw.line(surf, col, view.tp(*o[0, k]), view.tp(*end), 1)
            pygame.draw.circle(surf, col, view.tp(*end), 3)
    draw_car(surf, view, sim.x[i], sim.y[i], sim.h[i], color, outline=(255, 255, 255))


# ----------------------------------------------------------------- prompt de texto
class Prompt:
    def __init__(self, label, text, cb):
        self.label, self.text, self.cb = label, text, cb

    def handle(self, e, app):
        if e.type != pygame.KEYDOWN:
            return
        if e.key == pygame.K_ESCAPE:
            app.prompt = None
        elif e.key == pygame.K_RETURN:
            app.prompt = None
            self.cb(self.text)
        elif e.key == pygame.K_BACKSPACE:
            self.text = self.text[:-1]
        elif e.unicode and e.unicode.isprintable():
            self.text += e.unicode

    def draw(self, surf):
        r = pygame.Rect(240, 320, 800, 90)
        pygame.draw.rect(surf, (20, 22, 28), r)
        pygame.draw.rect(surf, (120, 200, 255), r, 2)
        txt(surf, self.label + "  (Enter = ok, Esc = cancela)", r.x + 14, r.y + 12)
        txt(surf, self.text + "_", r.x + 14, r.y + 46, (255, 255, 160), 20)


# ----------------------------------------------------------------- MENU
class MenuMode:
    fps = 30
    ITEMS = [("track", "Pista", "choice"), ("model", "Modelo (WATCH/TEST/retomar)", "choice"),
             ("hidden", "Camadas ocultas (ex: 16,16)", "text"),
             ("pop", "POPULATION_SIZE", "num", 25, 2, 5000),
             ("mut_rate", "MUTATION_RATE", "num", 0.01, 0.0, 1.0),
             ("mut_strength", "MUTATION_STRENGTH", "num", 0.05, 0.0, 3.0),
             ("elite", "ELITE_PERCENTAGE", "num", 0.01, 0.01, 0.5),
             ("max_gen", "MAX_GENERATIONS", "num", 50, 1, 100000),
             ("reverse", "Marcha à ré permitida", "bool"),
             ("car_pen", "Penalidade colisão com carro", "num", 50, 0, 5000),
             ("obs_pen", "Penalidade colisão com obstáculo", "num", 50, 0, 5000),
             ("out_pen", "Penalidade sair da área", "num", 50, 0, 5000),
             ("resume", "Retomar treino a partir do modelo", "bool"),
             ("headless", "Treino HEADLESS (sem render)", "bool")]

    def __init__(self, app):
        self.app, self.sel = app, 0
        self.tracks = list_files(TRACKS_DIR)
        self.models = ["(nenhum)"] + list_files(MODELS_DIR)
        if app.cfg["track"] not in self.tracks and self.tracks:
            app.cfg["track"] = self.tracks[0]

    def shown(self, it):
        v = self.app.cfg[it[0]]
        if it[2] == "bool": return "SIM" if v else "não"
        if it[2] == "choice" and it[0] == "model": return v or "(nenhum)"
        if it[2] == "num": return ("%d" % v) if it[3] >= 1 else ("%.2f" % v)
        return str(v)

    def change(self, d):
        it = self.ITEMS[self.sel]
        cfg = self.app.cfg
        if it[2] == "bool":
            cfg[it[0]] = not cfg[it[0]]
        elif it[2] == "num":
            v = cfg[it[0]] + d * it[3]
            v = min(max(v, it[4]), it[5])
            cfg[it[0]] = int(round(v)) if it[3] >= 1 else round(v, 3)
        elif it[2] == "choice":
            opts = self.tracks if it[0] == "track" else self.models
            if not opts: return
            cur = cfg[it[0]] if cfg[it[0]] in opts else opts[0]
            cfg[it[0]] = opts[(opts.index(cur) + d) % len(opts)]
            if cfg[it[0]] == "(nenhum)": cfg[it[0]] = ""

    def handle(self, e):
        if e.type != pygame.KEYDOWN: return
        app = self.app
        k = e.key
        if k == pygame.K_ESCAPE: app.quit()
        elif k == pygame.K_UP: self.sel = (self.sel - 1) % len(self.ITEMS)
        elif k == pygame.K_DOWN: self.sel = (self.sel + 1) % len(self.ITEMS)
        elif k == pygame.K_LEFT: self.change(-1)
        elif k == pygame.K_RIGHT: self.change(1)
        elif k == pygame.K_RETURN and self.ITEMS[self.sel][2] == "text":
            def cb(t):
                try:
                    parse_hidden(t); app.cfg["hidden"] = t.replace(" ", "")
                except ValueError as ex: app.flash(str(ex))
            app.prompt = Prompt("Camadas ocultas (separadas por vírgula)", app.cfg["hidden"], cb)
        elif k == pygame.K_t: app.start(TrainMode)
        elif k == pygame.K_w: app.start(PlayMode, "watch")
        elif k == pygame.K_x: app.start(PlayMode, "test")
        elif k == pygame.K_d: app.start(PlayMode, "manual")
        elif k == pygame.K_e:
            p = os.path.join(TRACKS_DIR, app.cfg["track"])
            app.start(Editor, p if os.path.exists(p) else None)

    def update(self): pass

    def draw(self, surf):
        surf.fill((20, 22, 28))
        txt(surf, "PARKING AI", 40, 24, (120, 220, 255), 40)
        txt(surf, "laboratório de IA evolutiva para estacionamento", 44, 72, (150, 150, 160))
        y = 120
        for i, it in enumerate(self.ITEMS):
            sel = i == self.sel
            if sel: pygame.draw.rect(surf, (40, 50, 70), (30, y - 3, 760, 26))
            txt(surf, ("> " if sel else "  ") + it[1], 40, y, (255, 255, 255) if sel else (185, 185, 195))
            txt(surf, self.shown(it), 470, y, (255, 230, 120) if sel else (210, 200, 150))
            y += 28
        cfg = self.app.cfg
        try: arch = [NIN] + parse_hidden(cfg["hidden"]) + [4]
        except ValueError: arch = ["?"]
        txt(surf, "Arquitetura: " + " → ".join(map(str, arch)), 40, y + 10, (140, 220, 160))
        acts = [("T", "TRAIN (treinar)"), ("W", "WATCH (ver modelo)"), ("X", "TEST (todas as pistas)"),
                ("E", "EDITOR de pistas"), ("D", "dirigir manualmente"), ("Esc", "sair")]
        yy = 120
        txt(surf, "AÇÕES", 850, yy - 30, (120, 220, 255))
        for k, s in acts:
            txt(surf, "[%s]" % k, 850, yy, (255, 230, 120)); txt(surf, s, 910, yy); yy += 30
        for i, s in enumerate(["↑↓ escolhe item   ←→ altera valor", "Enter edita texto",
                               "Entradas da rede: %d" % NIN, "Saídas: STEERING THROTTLE",
                               "        BRAKE HANDBRAKE", "", "Modelos ficam em models/", "Pistas ficam em tracks/"]):
            txt(surf, s, 850, 340 + i * 22, (150, 150, 160))


# ----------------------------------------------------------------- TREINO
class TrainMode:
    fps = 30

    def __init__(self, app):
        self.app = app
        cfg = app.cfg
        self.track = Track.load(os.path.join(TRACKS_DIR, cfg["track"]))
        net = None
        if cfg["resume"]:
            if not cfg["model"]:
                raise ValueError("escolha um modelo no menu para retomar")
            net = ai.Network.load(os.path.join(MODELS_DIR, cfg["model"]))
        self.tr = Trainer(self.track, cfg, net)
        self.view = View((10, 10, 900, 760), self.track.size)
        self.headless, self.viewmode, self.paused = cfg["headless"], 0, False
        self.last_draw = 0

    def handle(self, e):
        if e.type != pygame.KEYDOWN: return
        k, app, tr = e.key, self.app, self.tr
        if k == pygame.K_ESCAPE: app.back()
        elif k == pygame.K_SPACE: self.paused = not self.paused
        elif k == pygame.K_h: self.headless = not self.headless
        elif k == pygame.K_v: self.viewmode = (self.viewmode + 1) % 3
        elif k == pygame.K_s and tr.best_net:
            app.flash("salvo: " + os.path.basename(tr.best_net.save(next_model_path())))
        elif k == pygame.K_g and tr.gen_best_net:
            p = os.path.join(MODELS_DIR, "best_generation_%d.json" % tr.stats["gen"])
            app.flash("salvo: " + os.path.basename(tr.gen_best_net.save(p)))

    def update(self):
        if not self.paused and not self.tr.finished:
            self.tr.run(0.5 if self.headless else 0.03)

    def draw(self, surf):
        tr, sim = self.tr, self.tr.sim
        if self.headless and time.time() - self.last_draw < 0.2:
            return
        self.last_draw = time.time()
        if self.headless:
            surf.fill((20, 22, 28))
            txt(surf, "HEADLESS — renderização desligada (H liga)", 40, 300, (150, 150, 160), 22)
        else:
            draw_track(surf, self.track, self.view)
            if self.viewmode == 1:
                for i in np.flatnonzero(sim.alive):
                    draw_car(surf, self.view, sim.x[i], sim.y[i], sim.h[i], (60, 120, 200), outline=(30, 60, 100))
            if self.viewmode in (0, 1):
                draw_sim_car(surf, self.view, sim, tr.leader(), (255, 200, 60), sensors=True)
        # painel
        px = 920
        pygame.draw.rect(surf, (20, 22, 28), (px - 6, 0, 370, 780))
        st = tr.stats
        alive = int(sim.alive.sum())
        lines = [("TREINO", (120, 220, 255)), ("Pista: " + self.track.name, None),
                 ("Geração: %d / %d" % (tr.gen, tr.cfg["max_gen"]), None),
                 ("Rede: " + "→".join(map(str, tr.pop.layers)), None),
                 ("Vivos: %d / %d   passo %d" % (alive, sim.n, sim.steps), None),
                 ("Fitness líder: %.1f" % sim.fitness[tr.leader()], None), ("", None),
                 ("ÚLTIMA GERAÇÃO", (120, 220, 255))]
        if st:
            lines += [("Melhor fitness: %.1f" % st["best"], None), ("Fitness médio: %.1f" % st["mean"], None),
                      ("Melhor tempo: " + ("%.1fs" % st["best_time"] if st["best_time"] else "-"), None),
                      ("Taxa estacionou: %.1f%%" % (100 * st["park_rate"]), None),
                      ("Colisões: %d" % st["collisions"], None),
                      ("  com carros: %d" % st["car_collisions"], None),
                      ("Saíram da área: %d" % st["out"], None),
                      ("Melhor de todos: %.1f" % tr.best_fit, (255, 230, 120))]
        else:
            lines.append(("(aguardando 1ª geração)", None))
        y = 14
        for s, c in lines:
            txt(surf, s, px, y, c or (225, 225, 225)); y += 22
        # gráfico
        gy, gh, gw = 470, 150, 340
        pygame.draw.rect(surf, (32, 34, 42), (px, gy, gw, gh))
        hist = tr.history[-200:]
        if len(hist) > 1:
            allv = [v for h in hist for v in h[:2]]
            lo, hi = min(allv), max(allv)
            if hi - lo < 1: hi = lo + 1
            for j, col in ((0, (255, 210, 80)), (1, (90, 170, 255))):
                pts = [(px + i * gw / (len(hist) - 1), gy + gh - (h[j] - lo) / (hi - lo) * (gh - 6) - 3)
                       for i, h in enumerate(hist)]
                pygame.draw.lines(surf, col, False, pts, 2)
        txt(surf, "amarelo=melhor  azul=médio", px, gy + gh + 4, (150, 150, 160), 13)
        for i, s in enumerate(["[Espaço] pausa   [H] headless", "[V] vista: melhor/todos/nenhum",
                               "[S] salvar melhor (parking_ai_vN)", "[G] salvar melhor da geração",
                               "melhor de todos: models/best_auto.json", "[Esc] voltar ao menu"]):
            txt(surf, s, px, 650 + i * 20, (150, 150, 160), 14)
        if tr.finished: txt(surf, "MAX_GENERATIONS atingido!", px, 630, (255, 120, 120))
        elif self.paused: txt(surf, "PAUSADO", px, 630, (255, 120, 120))


# ----------------------------------------------------------------- WATCH / TEST / MANUAL
class PlayMode:
    fps = 30

    def __init__(self, app, kind):
        self.app, self.kind = app, kind
        cfg = app.cfg
        files = [cfg["track"]] if kind != "test" else list_files(TRACKS_DIR)
        self.tracks = [Track.load(os.path.join(TRACKS_DIR, f)) for f in files]
        self.net = None
        if kind != "manual":
            if not cfg["model"]:
                raise ValueError("escolha um modelo no menu (setas em 'Modelo')")
            self.net = ai.Network.load(os.path.join(MODELS_DIR, cfg["model"]))
            if self.net.layers[0] != NIN:
                raise ValueError("modelo tem %d entradas; o simulador usa %d" % (self.net.layers[0], NIN))
        self.ti, self.speed, self.sensors, self.trail_on, self.pause = 0, 1, True, True, False
        self.results, self.summary = [], False
        self.view = View((10, 10, 900, 760), (1000, 700))
        self.begin()

    def begin(self):
        self.track = self.tracks[self.ti]
        self.view.set_size(self.track.size)
        rev = True if self.kind == "manual" else self.net.allow_reverse
        self.sim = Sim(self.track, 1, rev, dict(car_pen=self.app.cfg["car_pen"], obs_pen=self.app.cfg["obs_pen"],
                                                out_pen=self.app.cfg["out_pen"]))
        self.trail, self.end_timer = [], 0

    def handle(self, e):
        if e.type != pygame.KEYDOWN: return
        k = e.key
        if k == pygame.K_ESCAPE: self.app.back()
        elif k == pygame.K_s: self.sensors = not self.sensors
        elif k == pygame.K_t: self.trail_on = not self.trail_on
        elif k == pygame.K_r: self.begin()
        elif k == pygame.K_p: self.pause = not self.pause
        elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS): self.speed = min(self.speed * 2, 16)
        elif k in (pygame.K_MINUS, pygame.K_KP_MINUS): self.speed = max(self.speed // 2, 1)

    def manual_action(self):
        k = pygame.key.get_pressed()
        steer = (k[pygame.K_RIGHT] - k[pygame.K_LEFT])
        thr = k[pygame.K_UP] - k[pygame.K_DOWN]
        return np.array([[steer, thr, float(k[pygame.K_b] or k[pygame.K_LSHIFT]), float(k[pygame.K_SPACE])]])

    def next_episode(self):
        if self.kind == "test":
            s = self.sim
            self.results.append((self.track.name, s.reason(0), float(s.fitness[0]), s.park_time[0]))
            self.ti += 1
            if self.ti >= len(self.tracks):
                self.summary = True
                return
        self.begin()

    def update(self):
        if self.summary or self.pause: return
        sim = self.sim
        for _ in range(self.speed):
            if not sim.alive.any(): break
            idx = np.array([0])
            act = self.manual_action() if self.kind == "manual" else self.net.forward(sim.observe(idx))
            sim.step(act)
            self.trail.append((sim.x[0], sim.y[0]))
        if not sim.alive.any():
            self.end_timer += 1
            if self.end_timer > 40 // (1 if self.kind != "manual" else 1):
                self.next_episode()

    def draw(self, surf):
        sim = self.sim
        if self.summary:
            surf.fill((20, 22, 28))
            txt(surf, "RESULTADO DO TEST — modelo " + self.app.cfg["model"], 40, 30, (120, 220, 255), 22)
            ok = sum(r[1] == "ESTACIONOU" for r in self.results)
            for i, (n, r, f, t) in enumerate(self.results):
                good = r == "ESTACIONOU"
                txt(surf, "%-24s %-24s fitness %8.1f  %s" % (n, r, f, "%.1fs" % t if good else ""), 40, 90 + i * 26,
                    (120, 230, 140) if good else (240, 150, 130))
            txt(surf, "Estacionou em %d de %d pistas.   [Esc] voltar" % (ok, len(self.results)), 40, 110 + len(self.results) * 26)
            return
        draw_track(surf, self.track, self.view)
        if self.trail_on and len(self.trail) > 1:
            pygame.draw.lines(surf, (255, 255, 120), False, [self.view.tp(*p) for p in self.trail[::2] + [self.trail[-1]]], 1)
        draw_sim_car(surf, self.view, sim, 0, sensors=self.sensors and sim.alive[0])
        px = 920
        pygame.draw.rect(surf, (20, 22, 28), (px - 6, 0, 370, 780))
        title = {"watch": "WATCH", "test": "TEST %d/%d" % (self.ti + 1, len(self.tracks)), "manual": "MANUAL"}[self.kind]
        lines = [(title, (120, 220, 255)), ("Pista: " + self.track.name, None)]
        if self.net: lines.append(("Modelo: " + self.app.cfg["model"], None))
        state = sim.reason(0)
        lines += [("Estado: " + state, (120, 230, 140) if sim.parked[0] else None),
                  ("Velocidade: %.0f px/s" % sim.v[0], None), ("Fitness: %.1f" % sim.fitness[0], None),
                  ("Tempo: %.1fs" % (sim.steps * DT), None), ("Ré: " + ("sim" if sim.allow_reverse else "não"), None),
                  ("Vel. sim: x%d" % self.speed, None), ("", None), ("SENSORES (0-1)", (120, 220, 255))]
        if sim.alive[0]:
            for n, v in zip(SENSOR_NAMES, sim.sense(np.array([0]))[0][0] / SENSOR_RANGE):
                lines.append(("%-11s %.2f" % (n, v), None))
        y = 14
        for s, c in lines:
            txt(surf, s, px, y, c or (225, 225, 225)); y += 22
        keys = ["[S] sensores  [T] trajetória", "[+/-] velocidade  [P] pausa", "[R] reinicia   [Esc] menu"]
        if self.kind == "manual": keys.insert(0, "setas dirige, Espaço=freio de mão, B=freio")
        for i, s in enumerate(keys):
            txt(surf, s, px, 690 + i * 20, (150, 150, 160), 14)


# ----------------------------------------------------------------- EDITOR
def in_rect(o, p, pad=0.0):
    r = math.radians(o.get("rot", 0))
    dx, dy = p[0] - o["x"], p[1] - o["y"]
    lon, lat = dx * math.cos(r) + dy * math.sin(r), -dx * math.sin(r) + dy * math.cos(r)
    return abs(lon) <= o["l"] / 2 + pad and abs(lat) <= o["w"] / 2 + pad


def seg_dist(p, a, b):
    ax, ay, bx, by = a[0], a[1], b[0], b[1]
    dx, dy = bx - ax, by - ay
    t = 0 if dx == dy == 0 else max(0, min(1, ((p[0] - ax) * dx + (p[1] - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(p[0] - ax - t * dx, p[1] - ay - t * dy)


class Editor:
    fps = 60
    TOOLS = ["Selecionar/mover", "Vaga (dado da IA)", "Carro (obstáculo)", "Obstáculo", "Parede (arrastar)",
             "Início do carro IA", "Linha (VISUAL)", "Área colorida (VISUAL)", "Seta (VISUAL)"]
    OKINDS = [("cone", 14, 14), ("barrier", 80, 12), ("object", 30, 30)]

    def __init__(self, app, path=None):
        self.app = app
        self.t = Track.load(path) if path else default_track()
        self.t.start.setdefault("l", CAR_L); self.t.start.setdefault("w", CAR_W)
        self.view = View((10, 10, 900, 760), self.t.size)
        self.tool, self.sel, self.drag, self.anchor = 0, None, None, None
        self.snap, self.okind, self.col = True, 0, 0
        pygame.key.set_repeat(300, 40)

    def leave(self):
        pygame.key.set_repeat()

    # ---- utilidades
    def sn(self, v):
        return round(v / 5) * 5 if self.snap else v

    def world(self, pos, snap=True):
        x, y = self.view.inv(*pos)
        return (self.sn(x), self.sn(y)) if snap else (x, y)

    def pick(self, p):
        t = self.t
        for lst in (t.cars, t.obstacles, t.walls):
            for o in reversed(lst):
                if in_rect(o, p, 2): return o
        if in_rect(t.start, p, 2): return t.start
        for v in reversed(t.visual):
            if v["kind"] in ("line", "arrow") and seg_dist(p, v["p1"], v["p2"]) < 8 / self.view.s: return v
            if v["kind"] == "area" and v["x"] <= p[0] <= v["x"] + v["w"] and v["y"] <= p[1] <= v["y"] + v["h"]: return v
        if in_rect(t.slot, p): return t.slot
        return None

    def delete(self, o):
        t = self.t
        for lst in (t.cars, t.obstacles, t.walls, t.visual):
            for i, x in enumerate(lst):
                if x is o:
                    del lst[i]
                    if self.sel is o: self.sel = None
                    return

    def shift(self, o, dx, dy, orig):
        if "p1" in o:
            o["p1"] = [orig["p1"][0] + dx, orig["p1"][1] + dy]
            o["p2"] = [orig["p2"][0] + dx, orig["p2"][1] + dy]
        else:
            o["x"], o["y"] = orig["x"] + dx, orig["y"] + dy

    # ---- eventos
    def handle(self, e):
        app, t = self.app, self.t
        if e.type == pygame.KEYDOWN:
            k, mods = e.key, pygame.key.get_mods()
            ctrl, shift = mods & pygame.KMOD_CTRL, mods & pygame.KMOD_SHIFT
            if k == pygame.K_ESCAPE:
                self.leave(); app.back(); return
            if ctrl and k == pygame.K_s:
                def cb(name):
                    name = "".join(c for c in name if c.isalnum() or c in "_-") or "pista"
                    t.name = name
                    t.save(os.path.join(TRACKS_DIR, name + ".json"))
                    app.cfg["track"] = name + ".json"
                    app.flash("pista salva: tracks/%s.json" % name)
                app.prompt = Prompt("Salvar pista como (nome)", t.name, cb); return
            if ctrl and k == pygame.K_o:
                def cb(name):
                    try:
                        self.t = Track.load(os.path.join(TRACKS_DIR, name.replace(".json", "") + ".json"))
                        self.t.start.setdefault("l", CAR_L); self.t.start.setdefault("w", CAR_W)
                        self.view.set_size(self.t.size); self.sel = None
                    except (OSError, ValueError) as ex: app.flash("erro: %s" % ex)
                app.prompt = Prompt("Abrir pista (" + ", ".join(n[:-5] for n in list_files(TRACKS_DIR)) + ")", "", cb); return
            if ctrl and k == pygame.K_n:
                self.t = default_track(); self.t.start.update(l=CAR_L, w=CAR_W); self.view.set_size(self.t.size); self.sel = None; return
            if pygame.K_1 <= k <= pygame.K_9: self.tool = k - pygame.K_1; self.anchor = None
            elif k == pygame.K_g: self.snap = not self.snap
            elif k == pygame.K_TAB:
                if self.tool == 3: self.okind = (self.okind + 1) % len(self.OKINDS)
                else: self.col = (self.col + 1) % len(VIS_COLORS)
            elif k in (pygame.K_DELETE, pygame.K_BACKSPACE) and self.sel: self.delete(self.sel)
            elif k in (pygame.K_F1, pygame.K_F2, pygame.K_F3, pygame.K_F4):
                i = 0 if k in (pygame.K_F1, pygame.K_F2) else 1
                t.size[i] = int(min(2000, max(300, t.size[i] + (50 if k in (pygame.K_F2, pygame.K_F4) else -50))))
                self.view.set_size(t.size)
            o = self.sel
            if o and "rot" in o:
                if k in (pygame.K_q, pygame.K_e):
                    step = 1 if shift else 45 if ctrl else 5
                    o["rot"] = (o["rot"] + (step if k == pygame.K_e else -step) + 180) % 360 - 180
                if "l" in o and k in (pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN) and o is not t.start:
                    d = 2 if k in (pygame.K_RIGHT, pygame.K_UP) else -2
                    if k in (pygame.K_LEFT, pygame.K_RIGHT): o["l"] = max(4, o["l"] + d)
                    else: o["w"] = max(4, o["w"] + d)
        elif e.type == pygame.MOUSEBUTTONDOWN and e.pos[0] < 915:
            p, raw = self.world(e.pos), self.world(e.pos, False)
            if e.button == 3:
                o = self.pick(raw)
                if o and o is not t.slot and o is not t.start: self.delete(o)
            elif e.button == 1:
                if self.tool == 0:
                    self.sel = self.pick(raw)
                    if self.sel: self.drag = (raw, dict(self.sel))
                elif self.tool == 1: t.slot.update(x=p[0], y=p[1]); self.sel = t.slot
                elif self.tool == 2:
                    t.cars.append(dict(x=p[0], y=p[1], l=CAR_L, w=CAR_W, rot=0)); self.sel = t.cars[-1]
                elif self.tool == 3:
                    kd, l, w = self.OKINDS[self.okind]
                    t.obstacles.append(dict(kind=kd, x=p[0], y=p[1], l=l, w=w, rot=0)); self.sel = t.obstacles[-1]
                elif self.tool == 5: t.start.update(x=p[0], y=p[1]); self.sel = t.start
                else: self.anchor = p
        elif e.type == pygame.MOUSEMOTION and self.drag and e.buttons[0]:
            raw = self.world(e.pos, False)
            (gx, gy), orig = self.drag
            self.shift(self.sel, self.sn(raw[0] - gx), self.sn(raw[1] - gy), orig)
        elif e.type == pygame.MOUSEBUTTONUP and e.button == 1:
            self.drag = None
            if self.anchor is not None:
                p, a = self.world(e.pos), self.anchor
                self.anchor = None
                if math.hypot(p[0] - a[0], p[1] - a[1]) < 5: return
                col = list(VIS_COLORS[self.col])
                if self.tool == 4:
                    t.walls.append(dict(x=(a[0] + p[0]) / 2, y=(a[1] + p[1]) / 2, l=math.hypot(p[0] - a[0], p[1] - a[1]),
                                        w=12, rot=round(math.degrees(math.atan2(p[1] - a[1], p[0] - a[0])), 1)))
                elif self.tool == 6: t.visual.append(dict(kind="line", p1=list(a), p2=list(p), color=col, width=3))
                elif self.tool == 8: t.visual.append(dict(kind="arrow", p1=list(a), p2=list(p), color=col, width=3))
                elif self.tool == 7:
                    t.visual.append(dict(kind="area", x=min(a[0], p[0]), y=min(a[1], p[1]),
                                         w=abs(p[0] - a[0]), h=abs(p[1] - a[1]), color=col))

    def update(self): pass

    def draw(self, surf):
        t = self.t
        draw_track(surf, t, self.view, show_start=True)
        # seleção
        o = self.sel
        if o is not None:
            if "p1" in o:
                for q in (o["p1"], o["p2"]): pygame.draw.circle(surf, (80, 255, 255), self.view.tp(*q), 6, 2)
            elif o.get("kind") == "area":
                pygame.draw.rect(surf, (80, 255, 255), (*self.view.tp(o["x"], o["y"]), o["w"] * self.view.s, o["h"] * self.view.s), 2)
            else:
                poly(surf, self.view, rect_corners(o["x"], o["y"], o["l"], o["w"], math.radians(o["rot"])), (80, 255, 255), 2)
                r = math.radians(o["rot"])
                draw_arrow(surf, self.view, (o["x"], o["y"]), (o["x"] + math.cos(r) * 25, o["y"] + math.sin(r) * 25), (80, 255, 255), 1.5)
        # prévia
        if self.anchor:
            m = self.world(pygame.mouse.get_pos())
            a = self.view.tp(*self.anchor)
            if self.tool == 7:
                b = self.view.tp(*m)
                pygame.draw.rect(surf, VIS_COLORS[self.col], (min(a[0], b[0]), min(a[1], b[1]), abs(b[0] - a[0]), abs(b[1] - a[1])), 1)
            else:
                pygame.draw.line(surf, (255, 255, 255) if self.tool != 4 else (180, 180, 190), a, self.view.tp(*m), 2)
        # painel
        px = 920
        pygame.draw.rect(surf, (20, 22, 28), (px - 6, 0, 370, 780))
        txt(surf, "EDITOR — " + t.name, px, 12, (120, 220, 255))
        txt(surf, "tamanho %dx%d  grade:%s" % (t.size[0], t.size[1], "ON" if self.snap else "off"), px, 34, (170, 170, 180), 14)
        for i, n in enumerate(self.TOOLS):
            sel = i == self.tool
            if sel: pygame.draw.rect(surf, (40, 60, 90), (px - 4, 62 + i * 24 - 2, 360, 22))
            extra = ""
            if sel and i == 3: extra = " [%s]" % self.OKINDS[self.okind][0]
            txt(surf, "%d %s%s" % (i + 1, n, extra), px, 62 + i * 24, (255, 255, 255) if sel else (185, 185, 195), 14)
        if self.tool >= 6:
            pygame.draw.rect(surf, VIS_COLORS[self.col], (px + 250, 62 + 6 * 24, 40, 16))
        help_ = ["Clique esq: usa a ferramenta", "Clique dir: apaga", "Selecionado:", " Q/E gira (Shift=1° Ctrl=45°)",
                 " ←→ comprimento  ↑↓ largura", " Del apaga", "Tab: tipo de obstáculo / cor",
                 "G grade   F1-F4 tamanho da pista", "Ctrl+S salva  Ctrl+O abre  Ctrl+N nova", "Esc volta ao menu", "",
                 "A vaga (verde) é dado da IA;", "linhas/áreas/setas são só visuais.",
                 "A seta da vaga = direção de entrada."]
        for i, s in enumerate(help_):
            txt(surf, s, px, 300 + i * 20, (150, 150, 160), 14)
        txt(surf, "carros:%d obst:%d paredes:%d visuais:%d" % (len(t.cars), len(t.obstacles), len(t.walls), len(t.visual)),
            px, 600, (170, 170, 180), 13)
        txt(surf, "pistas: " + ", ".join(n[:-5] for n in list_files(TRACKS_DIR))[:70], px, 622, (140, 140, 150), 13)


# ----------------------------------------------------------------- APP
class App:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("PARKING AI")
        self.screen = pygame.display.set_mode((1280, 780))
        self.clock = pygame.time.Clock()
        self.cfg = load_cfg()
        self.prompt, self.msg = None, ("", 0)
        self.mode = MenuMode(self)

    def flash(self, s): self.msg = (s, time.time() + 4)

    def start(self, cls, *a):
        save_cfg(self.cfg)
        try:
            self.mode = cls(self, *a)
        except Exception as ex:
            self.flash("erro: %s" % ex)

    def back(self):
        m = self.mode
        if isinstance(m, TrainMode) and m.tr.best_net:
            self.flash("melhor rede salva em models/best_auto.json")
        self.mode = MenuMode(self)

    def quit(self):
        save_cfg(self.cfg)
        pygame.quit()
        sys.exit()

    def run(self):
        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT: self.quit()
                elif self.prompt: self.prompt.handle(e, self)
                else: self.mode.handle(e)
            self.mode.update()
            self.mode.draw(self.screen)
            if self.prompt: self.prompt.draw(self.screen)
            if time.time() < self.msg[1]:
                pygame.draw.rect(self.screen, (60, 30, 30), (0, 748, 1280, 32))
                txt(self.screen, self.msg[0], 20, 755, (255, 220, 160))
            pygame.display.flip()
            self.clock.tick(self.mode.fps)


# ----------------------------------------------------------------- CLI headless
def cli_train(a):
    cfg = dict(DEFAULT_CFG)
    cfg.update(pop=a.pop, max_gen=a.gens, hidden=a.hidden, reverse=a.reverse)
    track = Track.load(a.track if os.path.exists(a.track) else os.path.join(TRACKS_DIR, a.track))
    net = ai.Network.load(a.resume) if a.resume else None
    def log(s):
        bt = "%.1fs" % s["best_time"] if s["best_time"] else "-"
        print("ger %4d | melhor %8.1f | médio %8.1f | tempo %5s | estacionou %5.1f%% | colisões %4d (carros %d)"
              % (s["gen"], s["best"], s["mean"], bt, 100 * s["park_rate"], s["collisions"], s["car_collisions"]), flush=True)
    tr = Trainer(track, cfg, net, log)
    print("pista=%s pop=%d rede=%s" % (track.name, cfg["pop"], tr.pop.layers))
    try:
        while not tr.finished:
            tr.run(1.0)
    except KeyboardInterrupt:
        print("interrompido")
    if tr.best_net:
        out = a.out or next_model_path()
        tr.best_net.save(out)
        print("melhor rede salva em", out)


def main():
    ap = argparse.ArgumentParser(description="PARKING AI")
    ap.add_argument("--train", action="store_true", help="treino headless no terminal")
    ap.add_argument("--track", default="simples.json")
    ap.add_argument("--pop", type=int, default=500)
    ap.add_argument("--gens", type=int, default=100)
    ap.add_argument("--hidden", default="16,16")
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--resume", default="", help="modelo .json para continuar treinando")
    ap.add_argument("--out", default="", help="onde salvar a melhor rede")
    a = ap.parse_args()
    ensure_default_tracks()
    if a.train:
        cli_train(a)
    else:
        App().run()


if __name__ == "__main__":
    main()
