"""
Project PROMETHEUS: Universal Continuous Energy Materials Discovery Engine
Plasticity-Rule Optimized Machine that Evolves Thinking, 
Heuristics, and Executable Universal Systems

This system evolves complete learning systems to tackle:
  - Discovering cheaper, cleaner, and more efficient ways to generate and store energy.
  - Contains the entire periodic table of elements (H to Bismuth + Lanthanides).
  - Evaluates Generation (Bandgap), Storage (Capacity/Cleanliness), and Cost.
  - Implements a CONTINUOUS LEARNING LOOP: The AI trains on its own discoveries
    to iteratively generate STRICTLY BETTER materials every iteration.
"""

import jax
import jax.numpy as jnp
import numpy as np
import random as pyrandom
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any
from copy import deepcopy
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================
def _setup_jax_device():
    try:
        devices = jax.devices()
        gpu_devices = [d for d in devices if d.platform == 'gpu']
        cpu_devices = [d for d in devices if d.platform == 'cpu']
        if gpu_devices:
            jax.config.update("jax_platform_name", "gpu")
            print(f"[PROMETHEUS] CUDA/GPU detected. Using device: {gpu_devices[0]}")
            return gpu_devices[0]
        else:
            jax.config.update("jax_platform_name", "cpu")
            print(f"[PROMETHEUS] No GPU detected. Falling back to CPU.")
            return cpu_devices[0] if cpu_devices else devices[0]
    except Exception as e:
        print(f"[PROMETHEUS] Device setup warning: {e}.")
        return None

jax.config.update("jax_enable_x64", True)
DEFAULT_DEVICE = _setup_jax_device()

def to_device(arr):
    if DEFAULT_DEVICE is not None:
        return jax.device_put(arr, DEFAULT_DEVICE)
    return arr

# ============================================================================
# PART 0: MATERIAL RECORD (for printing + saving full structure & results)
# ============================================================================
@dataclass
class SiteInfo:
    """One crystallographic site in an ABX3-style material."""
    role: str           # 'A-site cation' | 'B-site metal' | 'X-site anion'
    element: str        # element symbol, e.g. 'Ca'
    radius: float       # Å
    electronegativity: float
    mass: float         # g/mol
    cost: float         # $/kg
    toxicity: float     # 0-1
    # What the inverse-designer asked for at this site (the continuous
    # feature vector the gradient descent converged to):
    target_radius: float = 0.0
    target_en: float = 0.0
    target_mass: float = 0.0
    target_cost: float = 0.0

@dataclass
class MaterialRecord:
    """Full record of one inverse-designed material: structure + results."""
    iteration: int
    formula: str                       # e.g. 'Ca-Ti-O-S'
    formula_compact: str               # e.g. 'CaTiOS' (no separators)
    sites: List[SiteInfo] = field(default_factory=list)
    # AI's own prediction (from EvolvedCognitiveSystem.think):
    ai_bandgap: float = 0.0
    ai_storage: float = 0.0
    ai_pv_score: float = 0.0
    ai_battery_score: float = 0.0
    # Oracle's true physics (from _physics_oracle):
    true_bandgap: float = 0.0
    true_storage: float = 0.0
    true_stability: float = 0.0
    true_pv_score: float = 0.0
    true_battery_score: float = 0.0
    true_stability_score: float = 0.0
    true_cost_penalty: float = 0.0
    # Gates:
    chemistry_valid: bool = False
    chemistry_note: str = ""
    coherent: bool = False
    pv_disagreement: float = 0.0
    battery_disagreement: float = 0.0
    # Status:
    declared_new_best: bool = False
    status_note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'iteration': self.iteration,
            'formula': self.formula,
            'formula_compact': self.formula_compact,
            'sites': [
                {
                    'role': s.role, 'element': s.element,
                    'radius': s.radius, 'electronegativity': s.electronegativity,
                    'mass': s.mass, 'cost': s.cost, 'toxicity': s.toxicity,
                    'target_radius': s.target_radius, 'target_en': s.target_en,
                    'target_mass': s.target_mass, 'target_cost': s.target_cost,
                } for s in self.sites
            ],
            'ai_prediction': {
                'bandgap_eV': self.ai_bandgap,
                'storage_mAh_g': self.ai_storage,
                'pv_score': self.ai_pv_score,
                'battery_score': self.ai_battery_score,
            },
            'oracle_true': {
                'bandgap_eV': self.true_bandgap,
                'storage_mAh_g': self.true_storage,
                'stability_eV': self.true_stability,
                'pv_score': self.true_pv_score,
                'battery_score': self.true_battery_score,
                'stability_score': self.true_stability_score,
                'cost_penalty': self.true_cost_penalty,
            },
            'gates': {
                'chemistry_valid': self.chemistry_valid,
                'chemistry_note': self.chemistry_note,
                'coherent': self.coherent,
                'pv_disagreement': self.pv_disagreement,
                'battery_disagreement': self.battery_disagreement,
            },
            'status': {
                'declared_new_best': self.declared_new_best,
                'note': self.status_note,
            },
        }

# ============================================================================
# PART 1: SYMBOLIC PLASTICITY RULE EVOLUTION
# ============================================================================

class PlasticityNode:
    BINARY_OPS = {'+', '-', '*', '/'}
    UNARY_OPS = {'exp', 'sin', 'cos', 'abs', 'tanh', 'sigmoid', 'relu', 'sign', 'neg'}
    VARIABLES = {'pre', 'post', 'w', 'err', 'lr', 'mem', 'reward', 'mod'}
    CONSTANTS = [0.01, 0.1, 0.5, 1.0, -1.0, 2.0, 0.001]
    
    def __init__(self, value: str, children: List['PlasticityNode'] = None):
        self.value = value
        self.children = children or []
    
    def to_expr(self) -> str:
        v = self.value
        if v in self.VARIABLES: return f"ctx['{v}']"
        if v in self.CONSTANTS: return str(v)
        if v == '+': return f"({self.children[0].to_expr()} + {self.children[1].to_expr()})"
        if v == '-':
            if len(self.children) == 1: return f"(-{self.children[0].to_expr()})"
            return f"({self.children[0].to_expr()} - {self.children[1].to_expr()})"
        if v == '*': return f"({self.children[0].to_expr()} * {self.children[1].to_expr()})"
        if v == '/': return f"({self.children[0].to_expr()} / (jnp.abs({self.children[1].to_expr()}) + 1e-8))"
        if v == 'exp': return f"jnp.exp(jnp.clip({self.children[0].to_expr()}, -10, 10))"
        if v == 'sin': return f"jnp.sin({self.children[0].to_expr()})"
        if v == 'cos': return f"jnp.cos({self.children[0].to_expr()})"
        if v == 'abs': return f"jnp.abs({self.children[0].to_expr()})"
        if v == 'tanh': return f"jnp.tanh({self.children[0].to_expr()})"
        if v == 'sigmoid': return f"jax.nn.sigmoid({self.children[0].to_expr()})"
        if v == 'relu': return f"jax.nn.relu({self.children[0].to_expr()})"
        if v == 'sign': return f"jnp.sign({self.children[0].to_expr()})"
        if v == 'neg': return f"(-{self.children[0].to_expr()})"
        return "0.0"
    
    def mutate(self, depth: int = 0):
        if depth > 5: return
        if pyrandom.random() < 0.3 and self.children:
            self.children[pyrandom.randint(0, len(self.children) - 1)].mutate(depth + 1)
        elif pyrandom.random() < 0.4:
            if self.value in self.BINARY_OPS: self.value = pyrandom.choice(list(self.BINARY_OPS))
            elif self.value in self.UNARY_OPS: self.value = pyrandom.choice(list(self.UNARY_OPS))
            else: self.value = pyrandom.choice(list(self.VARIABLES) + [str(c) for c in self.CONSTANTS])
        elif pyrandom.random() < 0.5 and len(self.children) < 2:
            self.children.append(PlasticityNode(pyrandom.choice(list(self.VARIABLES) + [str(c) for c in self.CONSTANTS])))
    
    def copy(self) -> 'PlasticityNode':
        return PlasticityNode(self.value, [c.copy() for c in self.children])

def random_plasticity_tree(max_depth: int = 3) -> PlasticityNode:
    func = pyrandom.choice(list(PlasticityNode.BINARY_OPS) + list(PlasticityNode.UNARY_OPS))
    if func in PlasticityNode.BINARY_OPS:
        child1 = PlasticityNode(pyrandom.choice(list(PlasticityNode.VARIABLES)))
        child2 = random_plasticity_tree(max_depth - 1) if max_depth > 0 else PlasticityNode(pyrandom.choice(list(PlasticityNode.VARIABLES) + [str(c) for c in PlasticityNode.CONSTANTS]))
        return PlasticityNode(func, [child1, child2])
    else:
        child = random_plasticity_tree(max_depth - 1) if max_depth > 0 else PlasticityNode(pyrandom.choice(list(PlasticityNode.VARIABLES)))
        return PlasticityNode(func, [child])

def compile_plasticity(rule: PlasticityNode):
    expr = rule.to_expr()
    code = f"""
def _plasticity_fn(ctx):
    try:
        val = {expr}
        return jnp.nan_to_num(val, nan=0.0, posinf=0.1, neginf=-0.1)
    except:
        return ctx['w'] * 0.0
"""
    local_ns = {'jnp': jnp, 'jax': jax}
    exec(code, local_ns)
    return local_ns['_plasticity_fn']

# ============================================================================
# PART 2: COGNITIVE PROGRAM GENOME
# ============================================================================

COGNITIVE_OPS = {
    'matmul': 0, 'attention': 1, 'mem_write': 2, 'mem_read': 3, 'compare': 4,
    'gate': 5, 'accumulate': 6, 'threshold': 7, 'rotate': 8, 'blend': 9,
    'norm': 10, 'routing': 11,
}

@dataclass
class CognitiveStep:
    op: int
    input_src: int
    weight_dim: int
    plastic: bool

@dataclass
class SystemGenome:
    plasticity_rule: PlasticityNode
    cognitive_steps: List[CognitiveStep]
    memory_size: int
    n_think_steps: int
    hidden_dim: int
    learning_rate: float
    plasticity_mask: List[bool]
    
    def complexity(self) -> int:
        return len(self.cognitive_steps) * self.hidden_dim + self.memory_size
    
    def copy(self) -> 'SystemGenome':
        return SystemGenome(
            plasticity_rule=self.plasticity_rule.copy(),
            cognitive_steps=deepcopy(self.cognitive_steps),
            memory_size=self.memory_size,
            n_think_steps=self.n_think_steps,
            hidden_dim=self.hidden_dim,
            learning_rate=self.learning_rate,
            plasticity_mask=list(self.plasticity_mask),
        )

def random_genome() -> SystemGenome:
    n_steps = pyrandom.randint(2, 5)
    hidden = pyrandom.choice([32, 64, 128])
    steps = [CognitiveStep(
        op=pyrandom.randint(0, len(COGNITIVE_OPS) - 1),
        input_src=pyrandom.randint(0, 3),
        weight_dim=pyrandom.choice([32, 64, 128]),
        plastic=pyrandom.random() < 0.6,
    ) for _ in range(n_steps)]
    
    return SystemGenome(
        plasticity_rule=random_plasticity_tree(max_depth=3),
        cognitive_steps=steps,
        memory_size=pyrandom.choice([16, 32, 64]),
        n_think_steps=pyrandom.randint(1, 4),
        hidden_dim=hidden,
        learning_rate=10 ** pyrandom.uniform(-4, -1),
        plasticity_mask=[s.plastic for s in steps],
    )

# ============================================================================
# PART 3: THE EVOLVED COGNITIVE SYSTEM
# ============================================================================

class EvolvedCognitiveSystem:
    def __init__(self, genome: SystemGenome, input_dim: int, output_dim: int):
        self.genome = genome
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        key = jax.random.PRNGKey(pyrandom.randint(0, 10000))
        self.weights = []
        for step in genome.cognitive_steps:
            key, subkey = jax.random.split(key)
            w = jax.random.normal(subkey, (step.weight_dim, genome.hidden_dim)) * 0.3
            self.weights.append(to_device(w))
        
        key, sk1, sk2 = jax.random.split(key, 3)
        self.w_in = to_device(jax.random.normal(sk1, (input_dim, genome.hidden_dim)) * 0.3)
        self.w_out = to_device(jax.random.normal(sk2, (genome.hidden_dim, output_dim)) * 0.3)
        self.memory = to_device(jnp.zeros((genome.memory_size, genome.hidden_dim)))
        self.plasticity_fn = compile_plasticity(genome.plasticity_rule)
    
    def _apply_op(self, op: int, h: jnp.ndarray, w: jnp.ndarray, memory: jnp.ndarray, step_idx: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if w.shape[0] != h.shape[-1]:
            w = to_device(jax.random.normal(jax.random.PRNGKey(step_idx), (h.shape[-1], w.shape[1])) * 0.3)
        
        if op == 0: h = jnp.dot(h, w) + h
        elif op == 1:
            scores = jnp.dot(h, memory.T)
            attn = jax.nn.softmax(scores)
            h = jnp.dot(attn, memory) + h
        elif op == 2:
            mem_idx = step_idx % memory.shape[0]
            memory = memory.at[mem_idx].set(h)
        elif op == 3:
            mem_idx = step_idx % memory.shape[0]
            h = memory[mem_idx] + h
        elif op == 4: h = h - jnp.mean(memory, axis=0)
        elif op == 5: h = h * jax.nn.sigmoid(jnp.dot(h, w))
        elif op == 6: h = h + jnp.dot(h, w)
        elif op == 7: h = jnp.where(jnp.abs(h) > 0.5, h, 0.0)
        elif op == 8: h = jnp.roll(h, shift=step_idx)
        elif op == 9: h = 0.5 * h + 0.5 * jnp.dot(h, w)
        elif op == 10: h = h / (jnp.linalg.norm(h) + 1e-8)
        elif op == 11:
            gate = jax.nn.softmax(jnp.dot(h, w)[:2])
            h = gate[0] * h + gate[1] * jnp.dot(h, w)
        
        h = jnp.nan_to_num(h, nan=0.0, posinf=1.0, neginf=-1.0)
        h = jnp.clip(h, -5.0, 5.0)
        return h, memory
    
    def think(self, x: jnp.ndarray) -> jnp.ndarray:
        x = to_device(x)
        h = jnp.dot(x, self.w_in)
        
        for think_iter in range(self.genome.n_think_steps):
            for i, step in enumerate(self.genome.cognitive_steps):
                if step.input_src == 0: inp = h
                elif step.input_src == 1:
                    inp = jnp.mean(self.memory, axis=0)
                    if inp.shape[-1] != h.shape[-1]: inp = jnp.zeros_like(h)
                elif step.input_src == 2: inp = h
                else: inp = jnp.zeros_like(h)
                h, self.memory = self._apply_op(step.op, inp, self.weights[i], self.memory, think_iter * len(self.genome.cognitive_steps) + i)
        
        out = jnp.dot(h, self.w_out)
        out = jnp.nan_to_num(out, nan=0.0, posinf=10.0, neginf=-10.0)
        return jnp.clip(out, -20.0, 20.0)
    
    def learn(self, x: jnp.ndarray, y: jnp.ndarray) -> float:
        x = to_device(x)
        y = to_device(y)
        pred = self.think(x)
        err = jnp.clip(pred - y, -5.0, 5.0)
        loss = float(jnp.mean(err ** 2))
        if not np.isfinite(loss): loss = 100.0
        
        reward = -loss
        
        for i, step in enumerate(self.genome.cognitive_steps):
            if not self.genome.plasticity_mask[i]: continue
            w = self.weights[i]
            pre = jnp.zeros((w.shape[0],))
            pre = pre.at[:min(w.shape[0], x.shape[0])].set(x[:min(w.shape[0], x.shape[0])])
            post = jnp.dot(pre, w)
            err_broadcast = jnp.broadcast_to(jnp.mean(err), w.shape) if err.size != w.size else err.reshape(w.shape) if err.size == w.size else jnp.broadcast_to(jnp.mean(err), w.shape)
            
            ctx = {
                'pre': jnp.broadcast_to(pre[:, None], w.shape) if pre.ndim == 1 else pre,
                'post': jnp.broadcast_to(post[None, :], w.shape) if post.ndim == 1 else post,
                'w': w, 'err': err_broadcast,
                'lr': to_device(jnp.array(self.genome.learning_rate)),
                'mem': jnp.broadcast_to(jnp.mean(self.memory), w.shape),
                'reward': to_device(jnp.array(reward)),
                'mod': to_device(jnp.array(1.0)),
            }
            try:
                delta = self.plasticity_fn(ctx)
                if delta.shape != w.shape:
                    delta = jnp.broadcast_to(jnp.mean(delta), w.shape)
                delta = jnp.nan_to_num(delta, nan=0.0, posinf=0.01, neginf=-0.01)
                delta = jnp.clip(delta, -0.1, 0.1)
                self.weights[i] = jnp.clip(w + delta, -2.0, 2.0)
            except Exception:
                pass
        
        try:
            err_in = jnp.dot(err, self.w_out.T)
            ctx_in = {
                'pre': jnp.broadcast_to(x[:, None], self.w_in.shape) if x.ndim == 1 else x,
                'post': jnp.dot(x, self.w_in), 'w': self.w_in,
                'err': jnp.broadcast_to(err_in, self.w_in.shape),
                'lr': to_device(jnp.array(self.genome.learning_rate)),
                'mem': jnp.broadcast_to(jnp.mean(self.memory), self.w_in.shape),
                'reward': to_device(jnp.array(reward)), 'mod': to_device(jnp.array(1.0)),
            }
            delta_in = self.plasticity_fn(ctx_in)
            delta_in = jnp.nan_to_num(delta_in, nan=0.0, posinf=0.01, neginf=-0.01)
            delta_in = jnp.clip(delta_in, -0.1, 0.1)
            if delta_in.shape == self.w_in.shape:
                self.w_in = jnp.clip(self.w_in + delta_in, -2.0, 2.0)
        except:
            pass
        return loss

# ============================================================================
# PART 4: UNIVERSAL CHEMISTRY ORACLE & ENERGY EVALUATOR
# ============================================================================

class UniversalEnergyEvaluator:
    """
    Evaluates systems on REAL continuous physics for ANY chemical molecule.
    Input features per site (4 sites): [radius, electronegativity, mass, cost]
    Output metrics (7-vector): [Bandgap (eV), Storage (mAh/g), Stability (eV),
                                PV score, Battery score, Stability score,
                                Cost penalty]
    The PV and Battery scores are reported SEPARATELY and never blended.
    """
    def __init__(self, input_dim=16, output_dim=7):
        self.input_dim = input_dim  # 4 sites * 4 features
        self.output_dim = output_dim  # (eg, storage, stability, pv_score, battery_score, stability_score, cost_penalty) 
        
        # The Entire Periodic Table (H to Bi, + Lanthanides)
        # Format: 'Element': (Radius(Å), Electronegativity, Mass(g/mol), Cost($/kg), Toxicity(0-1))
        self.periodic_table = {
            'H': (0.31, 2.20, 1.008, 1.0, 0.0), 'He': (0.28, 0.0, 4.00, 50.0, 0.0),
            'Li': (1.28, 0.98, 6.94, 50.0, 0.2), 'Be': (0.96, 1.57, 9.01, 800.0, 0.8),
            'B': (0.84, 2.04, 10.81, 1000.0, 0.3), 'C': (0.76, 2.55, 12.01, 10.0, 0.0),
            'N': (0.71, 3.04, 14.01, 1.0, 0.0), 'O': (0.66, 3.44, 16.00, 1.0, 0.0),
            'F': (0.57, 3.98, 19.00, 50.0, 0.3), 'Ne': (0.58, 0.0, 20.18, 1000.0, 0.0),
            'Na': (1.66, 0.93, 22.99, 5.0, 0.3), 'Mg': (1.41, 1.31, 24.30, 5.0, 0.1),
            'Al': (1.21, 1.61, 26.98, 3.0, 0.2), 'Si': (1.11, 1.90, 28.08, 2.0, 0.1),
            'P': (1.07, 2.19, 30.97, 20.0, 0.3), 'S': (1.05, 2.58, 32.06, 1.0, 0.2),
            'Cl': (1.02, 3.16, 35.45, 2.0, 0.4), 'K': (2.03, 0.82, 39.09, 10.0, 0.2),
            'Ca': (1.76, 1.00, 40.08, 2.0, 0.1), 'Sc': (1.70, 1.36, 44.95, 5000.0, 0.3),
            'Ti': (1.60, 1.54, 47.86, 10.0, 0.2), 'V': (1.53, 1.63, 50.94, 50.0, 0.6),
            'Cr': (1.39, 1.66, 52.00, 10.0, 0.8), 'Mn': (1.39, 1.55, 54.93, 2.0, 0.7),
            'Fe': (1.32, 1.83, 55.84, 1.0, 0.3), 'Co': (1.26, 1.88, 58.93, 30.0, 0.6),
            'Ni': (1.24, 1.91, 58.69, 15.0, 0.7), 'Cu': (1.32, 1.90, 63.54, 8.0, 0.6),
            'Zn': (1.22, 1.65, 65.38, 3.0, 0.5), 'Ga': (1.22, 1.81, 69.72, 200.0, 0.4),
            'Ge': (1.20, 2.01, 72.63, 1000.0, 0.3), 'As': (1.19, 2.18, 74.92, 50.0, 0.9),
            'Se': (1.20, 2.55, 78.97, 30.0, 0.5), 'Br': (1.20, 2.96, 79.90, 5.0, 0.4),
            'Rb': (2.20, 0.82, 85.46, 1000.0, 0.3), 'Sr': (1.95, 0.95, 87.62, 50.0, 0.3),
            'Y': (1.90, 1.22, 88.90, 30.0, 0.2), 'Zr': (1.75, 1.33, 91.22, 30.0, 0.2),
            'Nb': (1.64, 1.6, 92.90, 40.0, 0.3), 'Mo': (1.54, 2.16, 95.95, 20.0, 0.4),
            'Tc': (1.47, 1.9, 98.0, 10000.0, 0.8), 'Ru': (1.46, 2.2, 101.07, 1000.0, 0.3),
            'Rh': (1.42, 2.28, 102.90, 5000.0, 0.3), 'Pd': (1.39, 2.20, 106.42, 2000.0, 0.3),
            'Ag': (1.45, 1.93, 107.87, 500.0, 0.4), 'Cd': (1.44, 1.69, 112.41, 10.0, 0.9),
            'In': (1.42, 1.78, 114.81, 200.0, 0.5), 'Sn': (1.39, 1.96, 118.71, 20.0, 0.4),
            'Sb': (1.39, 2.05, 121.76, 10.0, 0.6), 'Te': (1.38, 2.1, 127.60, 50.0, 0.6),
            'I': (1.39, 2.66, 126.90, 20.0, 0.3), 'Xe': (1.40, 2.6, 131.29, 1000.0, 0.0),
            'Cs': (2.44, 0.79, 132.90, 1000.0, 0.3), 'Ba': (2.15, 0.89, 137.32, 10.0, 0.3),
            'La': (1.95, 1.10, 138.90, 5.0, 0.2), 'Ce': (1.85, 1.12, 140.11, 5.0, 0.2),
            'Pr': (2.47, 1.13, 140.90, 50.0, 0.2), 'Nd': (2.06, 1.14, 144.24, 30.0, 0.2),
            'W': (1.39, 2.36, 183.84, 50.0, 0.4), 'Pt': (1.39, 2.28, 195.08, 3000.0, 0.3),
            'Au': (1.36, 2.54, 196.97, 4000.0, 0.2), 'Hg': (1.32, 2.00, 200.59, 100.0, 1.0),
            'Tl': (1.45, 1.62, 204.38, 100.0, 0.9), 'Pb': (1.46, 1.87, 207.2, 5.0, 0.9),
            'Bi': (1.48, 2.02, 208.98, 15.0, 0.5)
        }

        # ====================================================================
        # FIXED PHYSICAL TARGETS (these are LAWS, not learnable parameters)
        # --------------------------------------------------------------------
        # Previous versions let the targets drift toward whatever the last
        # discovery happened to land on. That is textbook goalpost-moving:
        # "new best" is then guaranteed by construction. The targets below
        # are external, physically-motivated constants and NEVER change.
        #   - target_bandgap = 1.34 eV : the Shockley-Queisser optimum for a
        #     single-junction photovoltaic absorber. NOT learnable.
        #   - target_storage = 450 mAh/g : a reasonable gravimetric-capacity
        #     ceiling for light-ion (Li/Na/Mg) insertion cathodes. NOT learnable.
        # ====================================================================
        self.FIXED_TARGET_BANDGAP = 1.34     # eV  -- Shockley-Queisser optimum
        self.FIXED_TARGET_STORAGE = 450.0    # mAh/g -- light-ion cathode ceiling
        self.FIXED_OPTIMAL_STABILITY = -4.5  # eV  -- Goldschmidt-ideal reference

        # ====================================================================
        # ADAPTIVE TRAINING EMPHASIS (this is what actually learns)
        # --------------------------------------------------------------------
        # The TARGETS are frozen. What the AI is allowed to adapt is *how
        # much attention it pays to each sub-objective during training*.
        # These sample weights cannot move the goalposts -- they only
        # decide which past examples to emphasise. Reported separately,
        # never blended into one shifting "energy score".
        # ====================================================================
        self.training_weights = {
            'pv_bandgap':     1.0,   # emphasis on PV-relevant examples
            'battery_storage':1.0,   # emphasis on battery-relevant examples
            'stability':      1.0,   # emphasis on structural stability
            'cost':           1.0,   # emphasis on low-cost examples
        }
        # Audit trail of weight revisions (only the EMPHASIS, never the targets).
        self.training_weight_history: List[Dict[str, float]] = [dict(self.training_weights)]
        self.discovery_history: List[Tuple] = []

        self.X, self.y, self.material_map = self._generate_materials_data()

    def _physics_oracle(self, features):
        """
        Continuous semi-empirical physics formulas for universal energy materials.

        Input : 16 features (4 sites: [radius, EN, mass, cost])
        Output: (bandgap_eV, storage_mAh_g, stability_eV,
                 pv_score, battery_score, stability_score, cost_penalty)
                 -- all metrics reported SEPARATELY. We NEVER blend
                 PV bandgap and battery storage into one composite
                 number, because they describe different physics.

        Scoring targets are FROZEN constants (Shockley-Queisser 1.34 eV
        for PV, 450 mAh/g for batteries). The oracle does not consult
        any mutable target dictionary.
        """
        # Extract sites (ABX3-style architecture mapping for stability)
        r1, en1, m1, c1 = features[0], features[1], features[2], features[3]
        r2, en2, m2, c2 = features[4], features[5], features[6], features[7]
        r3, en3, m3, c3 = features[8], features[9], features[10], features[11]
        r4, en4, m4, c4 = features[12], features[13], features[14], features[15]

        # 1. Structural Stability (Goldschmidt generalized)
        t = (r1 + r3) / (1.414 * (r2 + r3) + 1e-6)
        mu = r2 / (r3 + 1e-6)
        stability = -4.5 + 25.0 * (t - 0.9)**2 + 10.0 * (mu - 0.5)**2 + 0.5 * (en2 - en3)**2
        stability = jnp.clip(stability, -10.0, 5.0)

        # 2. Bandgap (PV generation efficiency) -- controlled by the
        # anion-site (site 3) radius + Goldschmidt distortion terms.
        base_eg = 5.0 - 1.5 * r3
        eg = base_eg + 2.0 * jnp.abs(t - 0.9) + 1.0 * jnp.abs(mu - 0.5)
        eg = jnp.clip(eg, 0.1, 6.0)

        # 3. Storage Capacity (gravimetric, light-ion insertion)
        avg_mass = (m1 + m2 + m3 + m4) / 4.0
        storage_cap = 1000.0 / (avg_mass + 1.0) * jnp.abs(en2 - en3)
        storage_cap = jnp.clip(storage_cap, 0.0, 500.0)

        # 4. Cost & cleanliness
        avg_cost = (c1 + c2 + c3 + c4) / 4.0
        cost_penalty = jnp.clip(avg_cost / 100.0, 0.0, 1.0)

        # ----------------------------------------------------------------
        # SEPARATE sub-scores (NOT blended). Each is on [0, 1].
        # PV      : how close bandgap is to the FROZEN 1.34 eV optimum.
        # Battery : how close storage is to the FROZEN 450 mAh/g ceiling.
        # Stability: structural stability quality, clipped.
        # ----------------------------------------------------------------
        pv_score       = 1.0 / (1.0 + (eg - self.FIXED_TARGET_BANDGAP)**2)
        battery_score  = 1.0 / (1.0 + (storage_cap - self.FIXED_TARGET_STORAGE)**2 / 1000.0)
        stability_score = 1.0 + jnp.clip(stability, -1.0, 0.0)  # in [0, 1]

        return eg, storage_cap, stability, pv_score, battery_score, stability_score, cost_penalty

    def _generate_materials_data(self):
        X_list, y_list, map_list = [], [], []
        keys = list(self.periodic_table.keys())

        # Generate random combinations of elements from the entire periodic table.
        # NOTE: y is now a 7-vector: (eg, storage, stability, pv_score,
        #       battery_score, stability_score, cost_penalty).
        for _ in range(500):
            e1, e2, e3, e4 = pyrandom.sample(keys, 4)
            p1 = self.periodic_table[e1]
            p2 = self.periodic_table[e2]
            p3 = self.periodic_table[e3]
            p4 = self.periodic_table[e4]

            features = jnp.array([p1[0], p1[1], p1[2], p1[3],
                                  p2[0], p2[1], p2[2], p2[3],
                                  p3[0], p3[1], p3[2], p3[3],
                                  p4[0], p4[1], p4[2], p4[3]])

            eg, cap, stab, pv_s, bat_s, stab_s, cost_pen = self._physics_oracle(features)
            y_vec = jnp.array([eg, cap, stab, pv_s, bat_s, stab_s, cost_pen])
            X_list.append(features)
            y_list.append(y_vec)
            map_list.append(f"{e1}-{e2}-{e3}-{e4}")

        return to_device(jnp.array(X_list)), to_device(jnp.array(y_list)), map_list

    # ====================================================================
    # SELF-IMPROVEMENT (REAL THIS TIME) -- ADAPT TRAINING EMPHASIS ONLY
    # --------------------------------------------------------------------
    # The TARGETS are frozen (Shockley-Queisser 1.34 eV, 450 mAh/g).
    # The SCORING FORMULA is frozen (PV, battery, stability reported
    # separately, never blended).
    #
    # The only thing this method touches is `training_weights`: how much
    # the AI emphasises each sub-objective during retraining. This cannot
    # move the goalposts because the goal (the frozen targets and the
    # separate sub-scores) is unchanged. It only decides which past
    # examples the AI pays more attention to.
    # ====================================================================
    def self_improve_method(self, discovery: Tuple) -> Dict[str, float]:
        """
        Adapt the AI's training emphasis based on the latest discovery.

        The discovery tuple is:
            (formula, eg, storage, stability, pv_score, battery_score,
             stability_score, cost_penalty, ai_pv_pred, ai_battery_pred,
             coherent)

        Adaptation rules (none of these move the targets):
          1. If the AI's prediction DISAGREES with the oracle on a
             sub-objective, raise that sub-objective's training weight
             so the AI gets more practice on it.
          2. If the AI's prediction AGREES (within tolerance), keep the
             weight where it is -- do not reward by changing the goal.
          3. All weights are clamped to [0.25, 3.0] so the AI never
             fully ignores any sub-objective.
        """
        self.discovery_history.append(discovery)
        # discovery layout (see continuous_discovery_loop):
        # 0:formula 1:eg 2:storage 3:stability
        # 4:pv_score 5:battery_score 6:stability_score 7:cost_penalty
        # 8:ai_pv_pred 9:ai_battery_pred 10:coherent(bool)
        eg          = discovery[1]
        storage     = discovery[2]
        pv_score    = discovery[4]
        bat_score   = discovery[5]
        stab_score  = discovery[6]
        cost_pen    = discovery[7]
        ai_pv       = discovery[8]
        ai_bat      = discovery[9]

        w = dict(self.training_weights)

        # 1. If AI's PV prediction is far from oracle's PV score, emphasise PV.
        pv_disagreement = abs(float(ai_pv) - float(pv_score))
        if pv_disagreement > 0.15:
            w['pv_bandgap'] = min(3.0, w['pv_bandgap'] * 1.10)
        else:
            # gentle decay so weights don't stay maxed forever
            w['pv_bandgap'] = max(0.25, w['pv_bandgap'] * 0.97)

        # 2. Same for battery.
        bat_disagreement = abs(float(ai_bat) - float(bat_score))
        if bat_disagreement > 0.15:
            w['battery_storage'] = min(3.0, w['battery_storage'] * 1.10)
        else:
            w['battery_storage'] = max(0.25, w['battery_storage'] * 0.97)

        # 3. Stability and cost emphasis drift gently based on how bad the
        #    latest material was on those axes. (This does NOT change the
        #    target -- it changes how much training time the AI spends on
        #    stable / cheap examples.)
        w['stability'] = min(3.0, max(0.25,
            w['stability'] * (1.0 + 0.05 * (1.0 - float(stab_score)))))
        w['cost']      = min(3.0, max(0.25,
            w['cost']      * (1.0 + 0.05 * float(cost_pen))))

        self.training_weights = w
        self.training_weight_history.append(dict(w))
        return w

    def rescore_database(self, X=None):
        """
        Recompute y for the database using the (frozen-formula) oracle.
        Because the oracle's scoring formula and targets are FROZEN, this
        is now deterministic and idempotent -- it does not change labels
        based on adaptation. It exists so that when the database grows
        (new discoveries are appended), the new rows get scored by the
        same oracle.

        Returns y of shape (N, 7):
            (eg, storage, stability, pv_score, battery_score,
             stability_score, cost_penalty)
        """
        X_in = self.X if X is None else X
        try:
            batched_oracle = jax.vmap(self._physics_oracle)
            results = batched_oracle(X_in)              # tuple of 7 (N,) arrays
            new_y = to_device(jnp.stack(results, axis=-1))  # -> (N, 7)
        except Exception:
            rows = []
            for i in range(X_in.shape[0]):
                m = self._physics_oracle(X_in[i])
                rows.append(jnp.array([m[0], m[1], m[2], m[3], m[4], m[5], m[6]]))
            new_y = to_device(jnp.stack(rows))
        if X is None:
            self.y = new_y
        return new_y

    # ====================================================================
    # CHEMISTRY-AWARE ELEMENT MATCHING (kills the H-H-H-H collapse)
    # --------------------------------------------------------------------
    # Previously every site was matched to the nearest element by raw
    # Euclidean distance on (r, EN, mass, cost), with NO chemistry
    # constraints. Gradient descent happily drifted to a corner where
    # every site's nearest element was H (smallest, lightest, cheapest).
    # That is not a real material.
    #
    # We now restrict each site to elements that are chemically plausible
    # for its ROLE in an ABX3-style structure:
    #   site 0 (A-site): large, low-EN cation   -- alkalis, alkaline earths, lanthanides
    #   site 1 (B-site): mid-radius transition / post-transition metal
    #   site 2 (X-site #1): small, high-EN anion  -- O, S, N, F, Cl, etc.
    #   site 3 (X-site #2): small, high-EN anion  -- same set, must differ from site 2
    # Plus a hard diversity constraint: no element may be reused across sites.
    # ====================================================================
    def _classify_element_role(self, sym: str) -> str:
        """Return one of: 'cation_large', 'cation_metal', 'anion', 'other'."""
        # A-site: alkali, alkaline-earth, lanthanides (large, low-EN)
        cation_large = {
            'Li','Na','K','Rb','Cs','Ca','Sr','Ba','Ra',
            'La','Ce','Pr','Nd','Sc','Y',
        }
        # B-site: transition metals, post-transition metals (mid-radius)
        cation_metal = {
            'Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Zr','Nb','Mo','Tc',
            'Ru','Rh','Pd','Ag','Cd','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg',
            'Al','Ga','In','Tl','Sn','Pb','Bi','Sb','Ge',
        }
        # X-site: small, high-electronegativity anions
        anion = {'O','S','Se','Te','N','P','As','F','Cl','Br','I','C','H','B'}

        if sym in cation_large: return 'cation_large'
        if sym in cation_metal: return 'cation_metal'
        if sym in anion:        return 'anion'
        return 'other'

    def match_features_to_real_material(self, x_discovery) -> Tuple[str, bool, str]:
        """
        Match a 16-feature inverse-designed vector back to a real
        ABX3-style formula, with chemistry + diversity constraints.

        Returns:
            formula      : "A-B-X-X" string of element symbols
            is_valid     : True iff every site was matched to a
                           role-appropriate element AND no element is
                           reused across sites (i.e. the formula is a
                           real, distinct compound, not H-H-H-H).
            failure_note : empty string if valid, else a description of
                           which constraint was violated.
        """
        # Build role-resolved element pools from the periodic table.
        all_elems = list(self.periodic_table.keys())
        pool_A = [e for e in all_elems if self._classify_element_role(e) == 'cation_large']
        pool_B = [e for e in all_elems if self._classify_element_role(e) == 'cation_metal']
        pool_X = [e for e in all_elems if self._classify_element_role(e) == 'anion']
        # Sites 2 and 3 are both anions.

        used = set()
        chosen = []
        failure = ""

        # Site 0: A-site cation
        e0, d0 = self._nearest_in_pool(x_discovery, 0, pool_A, used)
        if e0 is None:
            failure = "no A-site cation candidate matched"
            return "??-??-??-??", False, failure
        used.add(e0); chosen.append(e0)

        # Site 1: B-site metal
        e1, d1 = self._nearest_in_pool(x_discovery, 1, pool_B, used)
        if e1 is None:
            failure = "no B-site metal candidate matched"
            return f"{e0}-??-??-??", False, failure
        used.add(e1); chosen.append(e1)

        # Site 2: X-site anion (must differ from site 3)
        e2, d2 = self._nearest_in_pool(x_discovery, 2, pool_X, used)
        if e2 is None:
            failure = "no X-site anion candidate matched"
            return f"{e0}-{e1}-??-??", False, failure
        used.add(e2); chosen.append(e2)

        # Site 3: X-site anion (must differ from site 2)
        e3, d3 = self._nearest_in_pool(x_discovery, 3, pool_X, used)
        if e3 is None:
            # Allow site 3 == site 2 only if no distinct anion is available;
            # mark as invalid (still a real element, but degenerate).
            e3 = e2
            failure = "site 3 anion == site 2 anion (degenerate)"
            chosen.append(e3)
            return f"{chosen[0]}-{chosen[1]}-{chosen[2]}-{chosen[3]}", False, failure
        used.add(e3); chosen.append(e3)

        formula = f"{chosen[0]}-{chosen[1]}-{chosen[2]}-{chosen[3]}"
        # Final diversity check: at least 3 distinct elements among 4 sites.
        distinct = len(set(chosen))
        if distinct < 3:
            return formula, False, f"only {distinct} distinct elements (degenerate)"
        return formula, True, ""

    def _nearest_in_pool(self, x_discovery, site_idx: int, pool: List[str],
                         used: set) -> Tuple[str, float]:
        """Find the nearest unused element in `pool` to the features at
        `site_idx` (4 features per site). Returns (symbol, distance) or
        (None, inf) if no unused element exists in the pool."""
        if not pool:
            return None, float('inf')
        base = site_idx * 4
        r_t  = float(x_discovery[base + 0])
        en_t = float(x_discovery[base + 1])
        m_t  = float(x_discovery[base + 2])
        c_t  = float(x_discovery[base + 3])

        best_elem = None
        best_dist = float('inf')
        for elem in pool:
            if elem in used:
                continue
            r, en, m, c, _tox = self.periodic_table[elem]
            dist = ((r - r_t)**2
                    + (en - en_t)**2
                    + (m - m_t)**2 / 1000.0
                    + (c - c_t)**2 / 10000.0)
            if dist < best_dist:
                best_dist = dist
                best_elem = elem
        return best_elem, best_dist

    # ====================================================================
    # FULL STRUCTURE + ASCII FORMATTING (for printing + saving)
    # ====================================================================
    ROLE_LABELS = {
        'cation_large': 'A-site cation   (large, low-EN)',
        'cation_metal': 'B-site metal    (transition / post-trans.)',
        'anion':        'X-site anion    (small, high-EN)',
        'other':        'unclassified    ',
    }

    def build_material_record(self, x_discovery, iteration: int,
                              formula: str, is_chem_valid: bool, chem_note: str,
                              ai_pred, true_metrics,
                              coherent: bool, pv_disagree: float, bat_disagree: float,
                              declared_new_best: bool, status_note: str) -> MaterialRecord:
        """Construct a full MaterialRecord from the loop's outputs."""
        sites = []
        site_roles = ['A-site cation', 'B-site metal', 'X-site anion', 'X-site anion']
        symbols = formula.split('-') if formula and '?' not in formula else ['?','?','?','?']
        for j in range(4):
            sym = symbols[j] if j < len(symbols) else '?'
            if sym != '?' and sym in self.periodic_table:
                r, en, m, c, tox = self.periodic_table[sym]
            else:
                r = en = m = c = tox = 0.0
            base = j * 4
            sites.append(SiteInfo(
                role=site_roles[j], element=sym,
                radius=float(r), electronegativity=float(en),
                mass=float(m), cost=float(c), toxicity=float(tox),
                target_radius=float(x_discovery[base + 0]),
                target_en=float(x_discovery[base + 1]),
                target_mass=float(x_discovery[base + 2]),
                target_cost=float(x_discovery[base + 3]),
            ))
        true_eg, true_cap, true_stab, true_pv, true_bat, true_stab_s, true_cost_pen = (
            float(v) for v in true_metrics
        )
        compact = formula.replace('-', '') if '?' not in formula else 'UNKNOWN'
        return MaterialRecord(
            iteration=iteration,
            formula=formula,
            formula_compact=compact,
            sites=sites,
            ai_bandgap=float(ai_pred[0]),
            ai_storage=float(ai_pred[1]),
            ai_pv_score=float(ai_pred[3]),
            ai_battery_score=float(ai_pred[4]),
            true_bandgap=true_eg,
            true_storage=true_cap,
            true_stability=true_stab,
            true_pv_score=true_pv,
            true_battery_score=true_bat,
            true_stability_score=true_stab_s,
            true_cost_penalty=true_cost_pen,
            chemistry_valid=is_chem_valid,
            chemistry_note=chem_note,
            coherent=coherent,
            pv_disagreement=pv_disagree,
            battery_disagreement=bat_disagree,
            declared_new_best=declared_new_best,
            status_note=status_note,
        )

    def format_full_structure(self, rec: MaterialRecord) -> str:
        """Detailed multi-line text dump of one material: structure + results."""
        lines = []
        lines.append("=" * 90)
        lines.append(f"MATERIAL RECORD  (iteration {rec.iteration})")
        lines.append("=" * 90)
        lines.append(f"Formula (dashed) : {rec.formula}")
        lines.append(f"Formula (compact): {rec.formula_compact}")
        lines.append(f"Architecture     : ABX3-style perovskite (A = cation, B = metal, X1/X2 = anions)")
        lines.append("")
        lines.append("FULL SITE-BY-SITE STRUCTURE:")
        lines.append("-" * 90)
        header = f"{'Site':<6}{'Role':<40}{'Element':<8}{'r(Å)':>7}{'EN':>6}{'Mass':>8}{'Cost':>8}{'Tox':>6}"
        lines.append(header)
        lines.append("-" * 90)
        for j, s in enumerate(rec.sites):
            lines.append(
                f"{j:<6}{s.role:<40}{s.element:<8}"
                f"{s.radius:>7.3f}{s.electronegativity:>6.2f}{s.mass:>8.2f}{s.cost:>8.1f}{s.toxicity:>6.2f}"
            )
        lines.append("-" * 90)
        lines.append("Inverse-designer targets (what gradient descent asked for at each site):")
        for j, s in enumerate(rec.sites):
            lines.append(
                f"  site {j} ({s.element:>2}): target r={s.target_radius:.3f} Å, "
                f"EN={s.target_en:.3f}, mass={s.target_mass:.2f}, cost={s.target_cost:.2f}"
            )
        lines.append("")
        lines.append("RESULTS — AI PREDICTION vs ORACLE (TRUE PHYSICS):")
        lines.append("-" * 90)
        lines.append(f"{'Metric':<30}{'AI predicted':>16}{'Oracle (true)':>16}{'|diff|':>12}")
        lines.append("-" * 90)
        lines.append(f"{'Bandgap (eV)':<30}{rec.ai_bandgap:>16.3f}{rec.true_bandgap:>16.3f}{abs(rec.ai_bandgap-rec.true_bandgap):>12.3f}")
        lines.append(f"{'Storage (mAh/g)':<30}{rec.ai_storage:>16.3f}{rec.true_storage:>16.3f}{abs(rec.ai_storage-rec.true_storage):>12.3f}")
        lines.append(f"{'PV sub-score':<30}{rec.ai_pv_score:>16.3f}{rec.true_pv_score:>16.3f}{rec.pv_disagreement:>12.3f}")
        lines.append(f"{'Battery sub-score':<30}{rec.ai_battery_score:>16.3f}{rec.true_battery_score:>16.3f}{rec.battery_disagreement:>12.3f}")
        lines.append(f"{'Stability (eV)':<30}{'--':>16}{rec.true_stability:>16.3f}{'--':>12}")
        lines.append(f"{'Stability sub-score':<30}{'--':>16}{rec.true_stability_score:>16.3f}{'--':>12}")
        lines.append(f"{'Cost penalty':<30}{'--':>16}{rec.true_cost_penalty:>16.3f}{'--':>12}")
        lines.append("-" * 90)
        lines.append(f"Frozen targets : bandgap={self.FIXED_TARGET_BANDGAP} eV (Shockley-Queisser) | "
                     f"storage={self.FIXED_TARGET_STORAGE} mAh/g | stability={self.FIXED_OPTIMAL_STABILITY} eV")
        lines.append("")
        lines.append("GATES:")
        lines.append(f"  Chemistry valid : {rec.chemistry_valid}" + (f"  ({rec.chemistry_note})" if rec.chemistry_note else ""))
        lines.append(f"  Coherent (AI≈oracle within 0.20): {rec.coherent}  "
                     f"(pv_disagree={rec.pv_disagreement:.3f}, bat_disagree={rec.battery_disagreement:.3f})")
        lines.append("")
        lines.append(f"STATUS: {rec.status_note}")
        lines.append("=" * 90)
        return "\n".join(lines)

    def format_ascii_structure(self, rec: MaterialRecord) -> str:
        """Compact ASCII-art representation of the ABX3 unit cell."""
        if '?' in rec.formula:
            return f"[{rec.formula}]  (chemistry invalid -- no ASCII art)"
        s = rec.sites
        A  = s[0].element if len(s) > 0 else '?'
        B  = s[1].element if len(s) > 1 else '?'
        X1 = s[2].element if len(s) > 2 else '?'
        X2 = s[3].element if len(s) > 3 else '?'
        # Build a small legend with the key metrics
        eg    = rec.true_bandgap
        cap   = rec.true_storage
        pv    = rec.true_pv_score
        bat   = rec.true_battery_score
        stab  = rec.true_stability_score
        cost  = rec.true_cost_penalty
        lines = []
        lines.append("")
        lines.append(f"  ABX3-style unit cell for {rec.formula_compact}  (iter {rec.iteration})")
        lines.append("")
        lines.append("        +-----+-----+")
        lines.append("        |     |     |")
        lines.append(f"        |  {A:>2} |  {X1:>2} |    <- A-site cation + X-site anion #1")
        lines.append("        |     |     |")
        lines.append("        +-----+-----+")
        lines.append("        |     |     |")
        lines.append(f"        |  {B:>2} |  {X2:>2} |    <- B-site metal  + X-site anion #2")
        lines.append("        |     |     |")
        lines.append("        +-----+-----+")
        lines.append("")
        lines.append(f"  Site details:")
        for j, site in enumerate(s):
            lines.append(
                f"    site {j}: {site.element:>2}  [{site.role.split('(')[0].strip()}]  "
                f"r={site.radius:.2f}Å  EN={site.electronegativity:.2f}  "
                f"mass={site.mass:.1f}  cost=${site.cost:.1f}/kg  tox={site.toxicity:.2f}"
            )
        lines.append("")
        lines.append(f"  Computed properties (oracle / true physics):")
        lines.append(f"    Bandgap        = {eg:.3f} eV   (target {self.FIXED_TARGET_BANDGAP} eV)")
        lines.append(f"    Storage        = {cap:.1f} mAh/g (target {self.FIXED_TARGET_STORAGE} mAh/g)")
        lines.append(f"    PV sub-score   = {pv:.3f}")
        lines.append(f"    Battery score  = {bat:.3f}")
        lines.append(f"    Stability      = {stab:.3f}")
        lines.append(f"    Cost penalty   = {cost:.3f}")
        valid_str = "VALID" if rec.chemistry_valid else "INVALID"
        coh_str   = "COHERENT" if rec.coherent else "INCOHERENT"
        best_str  = " *** NEW BEST ***" if rec.declared_new_best else ""
        lines.append(f"  Gates: chemistry={valid_str}  coherency={coh_str}{best_str}")
        return "\n".join(lines)

    def evaluate(self, genome: SystemGenome) -> Tuple[float, Dict]:
        n_tasks = 8
        total_learn, total_gen, total_think = 0.0, 0.0, 0.0
        
        for task_id in range(n_tasks):
            key = jax.random.PRNGKey(task_id * 100 + 7)
            idx = jax.random.choice(key, self.X.shape[0], shape=(15,), replace=False)
            task = {
                'X_support': self.X[idx[:5]],
                'y_support': self.y[idx[:5]],
                'X_query': self.X[idx[5:]],
                'y_query': self.y[idx[5:]],
            }
            try:
                system = EvolvedCognitiveSystem(genome, self.input_dim, self.output_dim)
                losses = []
                for i in range(len(task['X_support'])):
                    loss = system.learn(task['X_support'][i], task['y_support'][i])
                    losses.append(max(0.0, loss))
                
                if len(losses) > 1 and losses[0] > 1e-8:
                    total_learn += max(0.0, min(1.0, (losses[0] - losses[-1]) / (losses[0] + 1e-8)))
                
                query_losses = []
                for i in range(len(task['X_query'])):
                    pred = system.think(task['X_query'][i])
                    ql = float(jnp.mean((pred - task['y_query'][i]) ** 2))
                    if not np.isfinite(ql): ql = 100.0
                    query_losses.append(ql)
                total_gen += max(0.0, min(1.0, 1.0 / (1.0 + np.mean(query_losses))))
                
                genome_copy = genome.copy()
                genome_copy.n_think_steps = 1
                system_fast = EvolvedCognitiveSystem(genome_copy, self.input_dim, self.output_dim)
                for i in range(min(3, len(task['X_support']))):
                    system_fast.learn(task['X_support'][i], task['y_support'][i])
                
                pred_fast = system_fast.think(task['X_query'][0])
                loss_fast = float(jnp.mean((pred_fast - task['y_query'][0]) ** 2))
                if not np.isfinite(loss_fast): loss_fast = 100.0
                total_think += max(0.0, min(1.0, loss_fast - (query_losses[0] if query_losses else 0.0)))
            except Exception:
                pass
        
        base_fitness = 0.35 * (total_learn / n_tasks) + 0.45 * (total_gen / n_tasks) + 0.20 * (total_think / n_tasks)
        fitness = float(base_fitness - 0.001 * genome.complexity())
        if not np.isfinite(fitness):
            fitness = -10.0
        return fitness, {'learning': total_learn/n_tasks, 'generalization': total_gen/n_tasks, 'thinking': total_think/n_tasks, 'complexity': genome.complexity()}

# ============================================================================
# PART 5: META-EVOLUTIONARY ENGINE
# ============================================================================

class EvolutionaryEngine:
    def __init__(self, population_size=20, input_dim=16, output_dim=7):
        self.pop_size = population_size
        self.evaluator = UniversalEnergyEvaluator(input_dim, output_dim)
        
        delta_rule = PlasticityNode('*', [PlasticityNode('lr'), PlasticityNode('*', [PlasticityNode('pre'), PlasticityNode('err')])])
        steps = [CognitiveStep(op=0, input_src=0, weight_dim=64, plastic=True) for _ in range(3)]
        seed_genome = SystemGenome(
            plasticity_rule=delta_rule,
            cognitive_steps=steps,
            memory_size=32, n_think_steps=2, hidden_dim=64,
            learning_rate=0.05, plasticity_mask=[True, True, True]
        )
        self.population = [seed_genome] + [random_genome() for _ in range(population_size - 1)]
        self.fitnesses = [0.0] * population_size
        
        self.strategies = {
            'mutate_plasticity': 0.20, 'mutate_program': 0.20, 'mutate_config': 0.15,
            'crossover': 0.20, 'add_step': 0.10, 'remove_step': 0.05,
            'deepen_thinking': 0.05, 'simplify_thinking': 0.05,
        }
        self.strategy_rewards = {k: 0.01 for k in self.strategies}
        self.strategy_counts = {k: 1 for k in self.strategies}
        self.hall_of_fame: List[Tuple[SystemGenome, float]] = []
        self.generation = 0
    
    def _mutate(self, genome: SystemGenome, strategy: str) -> SystemGenome:
        g = genome.copy()
        if strategy == 'mutate_plasticity':
            if pyrandom.random() < 0.6: g.plasticity_rule.mutate()
            else: g.plasticity_rule = random_plasticity_tree(max_depth=3)
        elif strategy == 'mutate_program':
            if g.cognitive_steps:
                idx = pyrandom.randint(0, len(g.cognitive_steps) - 1)
                g.cognitive_steps[idx].op = pyrandom.randint(0, len(COGNITIVE_OPS) - 1)
        elif strategy == 'mutate_config':
            param = pyrandom.choice(['memory_size', 'hidden_dim', 'learning_rate'])
            if param == 'memory_size': g.memory_size = pyrandom.choice([16, 32, 64, 128])
            elif param == 'hidden_dim': g.hidden_dim = pyrandom.choice([32, 64, 128, 256])
            else: g.learning_rate = 10 ** pyrandom.uniform(-5, 0)
        elif strategy == 'crossover':
            if len(self.population) > 1:
                donor = pyrandom.choice(self.population)
                if donor.cognitive_steps and g.cognitive_steps:
                    cp = pyrandom.randint(1, min(len(donor.cognitive_steps), len(g.cognitive_steps)))
                    g.cognitive_steps = deepcopy(donor.cognitive_steps[:cp]) + deepcopy(g.cognitive_steps[cp:])
                    g.plasticity_mask = [s.plastic for s in g.cognitive_steps]
        elif strategy == 'add_step':
            if len(g.cognitive_steps) < 8:
                new_step = CognitiveStep(pyrandom.randint(0, len(COGNITIVE_OPS)-1), pyrandom.randint(0, 3), pyrandom.choice([32, 64, 128]), pyrandom.random() < 0.6)
                g.cognitive_steps.append(new_step)
                g.plasticity_mask.append(new_step.plastic)
        elif strategy == 'remove_step':
            if len(g.cognitive_steps) > 2:
                idx = pyrandom.randint(0, len(g.cognitive_steps) - 1)
                g.cognitive_steps.pop(idx)
                g.plasticity_mask.pop(idx)
        elif strategy == 'deepen_thinking': g.n_think_steps = min(8, g.n_think_steps + 1)
        elif strategy == 'simplify_thinking': g.n_think_steps = max(1, g.n_think_steps - 1)
        return g
    
    def _evaluate_population(self):
        for i, genome in enumerate(self.population):
            try:
                fitness, _ = self.evaluator.evaluate(genome)
                self.fitnesses[i] = fitness
            except Exception:
                self.fitnesses[i] = -10.0
    
    def _select_parents(self, n: int) -> List[int]:
        parents = []
        for _ in range(n):
            candidates = pyrandom.sample(range(self.pop_size), min(3, self.pop_size))
            parents.append(max(candidates, key=lambda i: self.fitnesses[i]))
        return parents
    
    def evolve_one_generation(self) -> Tuple[SystemGenome, float, Dict, float]:
        gen_start = time.time()
        self._evaluate_population()
        ranked = sorted(enumerate(self.population), key=lambda x: self.fitnesses[x[0]], reverse=True)
        best_idx, best_genome = ranked[0]
        best_fitness = self.fitnesses[best_idx]
        
        self.hall_of_fame.append((best_genome.copy(), best_fitness))
        elite_count = max(2, self.pop_size // 4)
        new_pop = [ranked[i][1].copy() for i in range(elite_count)]
        
        strategy_names = list(self.strategies.keys())
        strategy_probs = list(self.strategies.values())
        
        while len(new_pop) < self.pop_size:
            parent_idx = self._select_parents(1)[0]
            parent = self.population[parent_idx].copy()
            strategy = np.random.choice(strategy_names, p=strategy_probs)
            offspring = self._mutate(parent, strategy)
            new_pop.append(offspring)
            self.strategy_counts[strategy] += 1
        
        self.population = new_pop[:self.pop_size]
        for k in self.strategies:
            self.strategies[k] = max(0.01, self.strategy_rewards[k] / (self.strategy_counts[k] + 1))
        total = sum(self.strategies.values())
        for k in self.strategies: self.strategies[k] /= total
        for k in self.strategy_rewards: self.strategy_rewards[k] *= 0.95
        
        gen_time = time.time() - gen_start
        self.generation += 1
        return best_genome, best_fitness, {}, gen_time
    
    def run(self, n_generations=15, verbose=True):
        print("=" * 90)
        print("Project PROMETHEUS: Universal Energy Materials Discovery")
        print("=" * 90)
        print(f"Population: {self.pop_size} | Generations: {n_generations} | Device: {DEFAULT_DEVICE}")
        print(f"Database: {len(self.evaluator.periodic_table)} Elements | Continuous Multi-Objective Physics")
        print("-" * 90)
        
        for gen in range(n_generations):
            best_genome, best_fit, details, gen_time = self.evolve_one_generation()
            if verbose:
                print(f"Gen {gen:3d} | Fit: {best_fit:+.4f} | Cplx: {best_genome.complexity():4d} | {gen_time:.1f}s")
        
        print("\n" + "=" * 90)
        print("EVOLUTION COMPLETE — Best System Discovered:")
        print("=" * 90)
        best_genome, best_fitness = self.hall_of_fame[-1]
        print(f"  Learning Rule (Plasticity):\n    Δw = {best_genome.plasticity_rule.to_expr()}")
        print(f"  Think Steps: {best_genome.n_think_steps} | Ops: {len(best_genome.cognitive_steps)} | Hidden: {best_genome.hidden_dim}")
        return best_genome, best_fitness


# ============================================================================
# PART 6: CONTINUOUS MATERIAL DISCOVERY & SELF-IMPROVEMENT LOOP
# ============================================================================

def continuous_discovery_loop(best_genome: SystemGenome, evaluator: UniversalEnergyEvaluator, iterations=10):
    """
    HONEST version of the discovery loop.

    What changed vs. the dishonest version:

      1. COHERENCY GATE. A discovery is only declared "NEW BEST" if the
         AI's own prediction `think(x)` AGREES with the oracle's
         `physics_oracle(x)` on the same input -- within a tolerance.
         If the AI predicts pv_score=0 but the oracle says pv_score=0.55,
         that is NOT a discovery, it is the AI failing. We log it as
         "incoherent" and do not claim a new best.

      2. CHEMISTRY-AWARE MATCHING. The inverse-designed feature vector
         is mapped back to a real ABX3-style formula using
         `evaluator.match_features_to_real_material`, which enforces
         role constraints (A-site = large cation, B-site = transition /
         post-transition metal, X-sites = small high-EN anions) and a
         per-site diversity constraint. No more H-H-H-H.

      3. FROZEN TARGETS. The inverse-design target always points at
         the Shockley-Queisser optimum (1.34 eV) and the storage ceiling
         (450 mAh/g). These never move, so "new best" is measured
         against a fixed external goal, not against whatever the last
         discovery happened to be.

      4. SEPARATE SUB-SCORES. We report PV score and Battery score
         separately and never blend them into one composite number.
         A material can be good for solar, good for batteries, both,
         or neither -- the user sees all four numbers, not a single
         "energy score" that hides which sub-objective is being satisfied.

      5. WHAT THE LOOP ACTUALLY LEARNS. After each discovery the only
         thing that adapts is `training_weights` (how much the AI
         emphasises each sub-objective during retraining). This cannot
         move the goalposts because the targets and the scoring formula
         are frozen. It only changes which past examples the AI pays
         more attention to.
    """
    print("\n" + "=" * 90)
    print("PHASE 2: HONEST CONTINUOUS DISCOVERY")
    print("=" * 90)
    print(f"  Frozen PV target       : bandgap = {evaluator.FIXED_TARGET_BANDGAP} eV (Shockley-Queisser)")
    print(f"  Frozen Battery target  : storage = {evaluator.FIXED_TARGET_STORAGE} mAh/g")
    print(f"  Frozen Stability target: {evaluator.FIXED_OPTIMAL_STABILITY} eV (Goldschmidt-ideal)")
    print("  Sub-scores reported SEPARATELY (PV / Battery / Stability) -- never blended.")
    print("  Coherency gate: a discovery only counts if AI prediction tracks the oracle.")
    print("  Chemistry gate: A-site cation, B-site metal, X-site anions, no reuse.")
    print("-" * 90)

    final_system = EvolvedCognitiveSystem(best_genome, evaluator.input_dim, evaluator.output_dim)
    X_db, y_db = evaluator.X, evaluator.y

    print(f"[1] Initial Training on {X_db.shape[0]} known elemental combinations...")
    for epoch in range(10):
        for i in range(X_db.shape[0]):
            final_system.learn(X_db[i], y_db[i])

    # FROZEN inverse-design target.
    # Output is the 7-vector: (eg, storage, stability, pv_score,
    # battery_score, stability_score, cost_penalty).
    # The first three are physical quantities; the last four are
    # quality sub-scores in [0,1]. We push the AI toward the frozen
    # physical optima AND toward maximum sub-scores.
    target = to_device(jnp.array([
        evaluator.FIXED_TARGET_BANDGAP,        # eg -> 1.34 eV
        evaluator.FIXED_TARGET_STORAGE,        # storage -> 450 mAh/g
        evaluator.FIXED_OPTIMAL_STABILITY,     # stability -> -4.5 eV
        1.0, 1.0, 1.0, 0.0,                    # pv=1, battery=1, stab=1, cost_pen=0
    ]))

    def loss_fn(x_in):
        pred = final_system.think(x_in)
        pred_clamped = jnp.clip(pred,
                                jnp.array([0.1, 0.0, -10.0, 0.0, 0.0, 0.0, 0.0]),
                                jnp.array([6.0, 500.0, 5.0, 1.0, 1.0, 1.0, 1.0]))
        # Weight the four quality sub-scores more than the raw physical
        # quantities, because the sub-scores are what we actually optimise.
        weights = jnp.array([1.0, 1.0, 1.0, 5.0, 5.0, 3.0, 3.0])
        return jnp.mean(((pred_clamped - target) ** 2) * weights)

    grad_fn = jax.grad(loss_fn)

    # Trackers
    best_discovery = None
    best_pv_score = -1.0
    best_battery_score = -1.0
    best_combined = -1.0  # pv + battery (only valid if both > 0)
    n_incoherent = 0
    n_invalid_chem = 0
    initial_weights = dict(evaluator.training_weights)
    # Collect a full MaterialRecord for every iteration (for printing + saving)
    all_records: List[MaterialRecord] = []

    # Tolerance for the coherency gate: AI's predicted sub-scores must
    # be within this absolute distance of the oracle's sub-scores.
    COHERENCY_TOL = 0.20

    for loop in range(iterations):
        w = evaluator.training_weights
        print(f"\n[2.{loop}] AI is searching for a better material...")
        print(f"     Current training emphasis: pv={w['pv_bandgap']:.2f} "
              f"battery={w['battery_storage']:.2f} "
              f"stability={w['stability']:.2f} cost={w['cost']:.2f}")
        print(f"     Frozen targets: Eg*={evaluator.FIXED_TARGET_BANDGAP} eV, "
              f"Storage*={evaluator.FIXED_TARGET_STORAGE} mAh/g")

        key = jax.random.PRNGKey(777 + loop)
        x_discovery = jax.random.uniform(key, (16,), minval=0.5, maxval=3.0)
        x_discovery = to_device(x_discovery)

        # ---- Inverse design by gradient descent on the input features ----
        lr = 0.02
        for i in range(500):
            final_system.memory = to_device(jnp.zeros((final_system.genome.memory_size, final_system.genome.hidden_dim)))
            grads = grad_fn(x_discovery)
            grads = jnp.nan_to_num(grads, nan=0.0, posinf=0.0, neginf=0.0)
            x_discovery = x_discovery - lr * grads
            # Enforce physical bounds based on real elemental data
            for j in range(4):
                x_discovery = x_discovery.at[j*4 + 0].set(jnp.clip(x_discovery[j*4 + 0], 0.28, 2.50))  # Radius
                x_discovery = x_discovery.at[j*4 + 1].set(jnp.clip(x_discovery[j*4 + 1], 0.0, 4.0))    # EN
                x_discovery = x_discovery.at[j*4 + 2].set(jnp.clip(x_discovery[j*4 + 2], 1.0, 210.0))  # Mass
                x_discovery = x_discovery.at[j*4 + 3].set(jnp.clip(x_discovery[j*4 + 3], 1.0, 5000.0)) # Cost

        # ---- Evaluate the discovery ----
        final_system.memory = to_device(jnp.zeros((final_system.genome.memory_size, final_system.genome.hidden_dim)))
        ai_pred = final_system.think(x_discovery)  # 7-vector
        ai_eg   = float(ai_pred[0])
        ai_cap  = float(ai_pred[1])
        ai_pv   = float(ai_pred[3])
        ai_bat  = float(ai_pred[4])

        # TRUE physics from the oracle (frozen formula)
        true_metrics = evaluator._physics_oracle(x_discovery)
        true_eg, true_cap, true_stab, true_pv, true_bat, true_stab_s, true_cost_pen = (
            float(v) for v in true_metrics
        )

        # ---- CHEMISTRY GATE: match to a real ABX3-style formula ----
        formula, is_chem_valid, chem_note = evaluator.match_features_to_real_material(x_discovery)

        # ---- COHERENCY GATE: does the AI agree with the oracle? ----
        pv_disagree  = abs(ai_pv  - true_pv)
        bat_disagree = abs(ai_bat - true_bat)
        coherent = (pv_disagree <= COHERENCY_TOL) and (bat_disagree <= COHERENCY_TOL)

        print("-" * 90)
        print(f"CONTINUOUS LOOP ITERATION {loop+1} RESULTS:")
        print(f"  AI Predicted:   Eg={ai_eg:.2f} eV | Storage={ai_cap:.1f} mAh/g | "
              f"PV={ai_pv:.3f} | Battery={ai_bat:.3f}")
        print(f"  Oracle (true):  Eg={true_eg:.2f} eV | Storage={true_cap:.1f} mAh/g | "
              f"PV={true_pv:.3f} | Battery={true_bat:.3f} | Stab={true_stab_s:.3f} | CostPen={true_cost_pen:.3f}")
        print(f"  Closest real formula: {formula}")
        print(f"  Chemistry valid: {is_chem_valid}" + (f"  ({chem_note})" if chem_note else ""))
        print(f"  Coherency: pv_disagree={pv_disagree:.3f} bat_disagree={bat_disagree:.3f} "
              f"-> {'COHERENT' if coherent else 'INCOHERENT'}")

        # ---- HONEST DISCOVERY DECLARATION ----
        # A discovery only counts as "new best" if:
        #   (a) the AI's prediction is coherent with the oracle (the AI
        #       actually found this material, not the oracle's fallback)
        #   (b) the chemistry is valid (real ABX3-style formula)
        #   (c) the relevant sub-score exceeds the previous best
        declared_new_best = False
        status_note = ""
        if not coherent:
            n_incoherent += 1
            status_note = "INCOHERENT -- AI prediction does not match oracle. NOT a discovery."
            print(f"  Status: {status_note}")
        if not is_chem_valid:
            n_invalid_chem += 1
            status_note = "CHEMISTRY INVALID -- formula is degenerate. NOT a discovery."
            print(f"  Status: {status_note}")

        if coherent and is_chem_valid:
            if true_pv > best_pv_score:
                best_pv_score = true_pv
                declared_new_best = True
                status_note = f"NEW BEST PV MATERIAL (PV={true_pv:.3f})"
                print(f"  Status: {status_note}")
            if true_bat > best_battery_score:
                best_battery_score = true_bat
                declared_new_best = True
                status_note = f"NEW BEST BATTERY MATERIAL (Battery={true_bat:.3f})"
                print(f"  Status: {status_note}")
            combined = true_pv + true_bat
            if combined > best_combined:
                best_combined = combined
                best_discovery = (formula, true_eg, true_cap, true_stab, true_pv, true_bat, true_stab_s, true_cost_pen)
                declared_new_best = True
                status_note = f"NEW BEST COMBINED MATERIAL (PV+Battery={combined:.3f})"
                print(f"  Status: {status_note}")
            if not declared_new_best:
                status_note = "Coherent and chemically valid, but did not exceed any prior best."
                print(f"  Status: {status_note}")

        # ---- BUILD FULL MATERIAL RECORD + PRINT STRUCTURE ----
        record = evaluator.build_material_record(
            x_discovery=x_discovery,
            iteration=loop + 1,
            formula=formula,
            is_chem_valid=is_chem_valid,
            chem_note=chem_note,
            ai_pred=ai_pred,
            true_metrics=true_metrics,
            coherent=coherent,
            pv_disagree=pv_disagree,
            bat_disagree=bat_disagree,
            declared_new_best=declared_new_best,
            status_note=status_note,
        )
        all_records.append(record)

        # Print the ASCII-art structure (compact, always)
        print(evaluator.format_ascii_structure(record))
        # Print the FULL detailed structure + results (always, so the
        # user sees the complete picture per iteration)
        print(evaluator.format_full_structure(record))

        # ---- FEED THE TRUE PHYSICS BACK INTO THE AI'S DATABASE ----
        new_X = x_discovery.reshape(1, 16)
        new_y = jnp.array([[true_eg, true_cap, true_stab, true_pv, true_bat, true_stab_s, true_cost_pen]])
        X_db = jnp.concatenate([X_db, new_X])
        y_db = jnp.concatenate([y_db, new_y])

        # ---- ADAPT TRAINING EMPHASIS (the only thing that learns) ----
        # discovery tuple layout: (formula, eg, storage, stability,
        # pv_score, battery_score, stability_score, cost_penalty,
        # ai_pv_pred, ai_battery_pred, coherent)
        discovery_record = (formula, true_eg, true_cap, true_stab,
                            true_pv, true_bat, true_stab_s, true_cost_pen,
                            ai_pv, ai_bat, coherent)
        new_weights = evaluator.self_improve_method(discovery_record)
        print(f"  [TRAINING EMPHASIS UPDATED] "
              f"pv={new_weights['pv_bandgap']:.2f} "
              f"battery={new_weights['battery_storage']:.2f} "
              f"stability={new_weights['stability']:.2f} "
              f"cost={new_weights['cost']:.2f}")
        print("  (Note: targets and scoring formula are FROZEN -- only training emphasis adapts.)")

        # Re-score the database with the (frozen, deterministic) oracle so
        # the new row gets consistent labels.
        y_db = evaluator.rescore_database(X_db)

        # Retrain. The training_weights modulate how many extra epochs we
        # spend on each sub-objective (a simple proxy for sample weighting
        # that does not require restructuring EvolvedCognitiveSystem.learn).
        print(f"  [RETRAIN] {X_db.shape[0]} materials, "
              f"emphasis-weighted passes...")
        base_passes = 2
        for _ in range(base_passes):
            for i in range(X_db.shape[0]):
                final_system.learn(X_db[i], y_db[i])
        # Extra passes on PV- and Battery-relevant examples (the ones whose
        # oracle sub-scores are above 0.3). This is what the training_weights
        # actually do -- they steer WHERE the AI spends extra learning effort.
        extra_pv = max(0, int(round(new_weights['pv_bandgap'])))
        extra_bat = max(0, int(round(new_weights['battery_storage'])))
        for _ in range(extra_pv):
            for i in range(X_db.shape[0]):
                if float(y_db[i, 3]) > 0.3:  # PV-relevant
                    final_system.learn(X_db[i], y_db[i])
        for _ in range(extra_bat):
            for i in range(X_db.shape[0]):
                if float(y_db[i, 4]) > 0.3:  # Battery-relevant
                    final_system.learn(X_db[i], y_db[i])

    # ==================================================================
    # SAVE ALL DISCOVERIES (full structure + results) TO DISK
    # ------------------------------------------------------------------
    # Three files are written to /home/z/my-project/download/:
    #   prometheus_discoveries.json   -- machine-readable, full detail
    #   prometheus_discoveries.csv    -- tabular, one row per discovery
    #   prometheus_discoveries.txt    -- human-readable, full structure
    #                                    + ASCII art for every iteration
    # ==================================================================
    save_dir = "/home/z/my-project/download"
    try:
        import json, csv, os
        os.makedirs(save_dir, exist_ok=True)

        # --- JSON ---
        json_path = os.path.join(save_dir, "prometheus_discoveries.json")
        json_payload = {
            'metadata': {
                'total_iterations': iterations,
                'frozen_target_bandgap_eV': evaluator.FIXED_TARGET_BANDGAP,
                'frozen_target_storage_mAh_g': evaluator.FIXED_TARGET_STORAGE,
                'frozen_optimal_stability_eV': evaluator.FIXED_OPTIMAL_STABILITY,
                'coherency_tolerance': COHERENCY_TOL,
                'n_incoherent': n_incoherent,
                'n_chemistry_invalid': n_invalid_chem,
                'n_valid_discoveries': max(0, iterations - n_incoherent - n_invalid_chem),
                'best_pv_score': best_pv_score if best_pv_score >= 0 else None,
                'best_battery_score': best_battery_score if best_battery_score >= 0 else None,
                'best_combined_pv_plus_battery': best_combined if best_combined >= 0 else None,
                'final_training_weights': evaluator.training_weights,
            },
            'discoveries': [rec.to_dict() for rec in all_records],
        }
        with open(json_path, 'w') as f:
            json.dump(json_payload, f, indent=2)
        print(f"\n  [SAVED] Full discovery records (JSON)  -> {json_path}")

        # --- CSV ---
        csv_path = os.path.join(save_dir, "prometheus_discoveries.csv")
        csv_fields = [
            'iteration', 'formula', 'formula_compact',
            'site0_element', 'site0_role', 'site0_radius', 'site0_EN', 'site0_mass', 'site0_cost', 'site0_tox',
            'site1_element', 'site1_role', 'site1_radius', 'site1_EN', 'site1_mass', 'site1_cost', 'site1_tox',
            'site2_element', 'site2_role', 'site2_radius', 'site2_EN', 'site2_mass', 'site2_cost', 'site2_tox',
            'site3_element', 'site3_role', 'site3_radius', 'site3_EN', 'site3_mass', 'site3_cost', 'site3_tox',
            'ai_bandgap_eV', 'ai_storage_mAh_g', 'ai_pv_score', 'ai_battery_score',
            'true_bandgap_eV', 'true_storage_mAh_g', 'true_stability_eV',
            'true_pv_score', 'true_battery_score', 'true_stability_score', 'true_cost_penalty',
            'chemistry_valid', 'chemistry_note', 'coherent',
            'pv_disagreement', 'battery_disagreement',
            'declared_new_best', 'status_note',
        ]
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            for rec in all_records:
                row = {
                    'iteration': rec.iteration,
                    'formula': rec.formula,
                    'formula_compact': rec.formula_compact,
                }
                for j, s in enumerate(rec.sites):
                    row[f'site{j}_element'] = s.element
                    row[f'site{j}_role'] = s.role
                    row[f'site{j}_radius'] = round(s.radius, 4)
                    row[f'site{j}_EN'] = round(s.electronegativity, 4)
                    row[f'site{j}_mass'] = round(s.mass, 4)
                    row[f'site{j}_cost'] = round(s.cost, 4)
                    row[f'site{j}_tox'] = round(s.toxicity, 4)
                row['ai_bandgap_eV'] = round(rec.ai_bandgap, 4)
                row['ai_storage_mAh_g'] = round(rec.ai_storage, 4)
                row['ai_pv_score'] = round(rec.ai_pv_score, 4)
                row['ai_battery_score'] = round(rec.ai_battery_score, 4)
                row['true_bandgap_eV'] = round(rec.true_bandgap, 4)
                row['true_storage_mAh_g'] = round(rec.true_storage, 4)
                row['true_stability_eV'] = round(rec.true_stability, 4)
                row['true_pv_score'] = round(rec.true_pv_score, 4)
                row['true_battery_score'] = round(rec.true_battery_score, 4)
                row['true_stability_score'] = round(rec.true_stability_score, 4)
                row['true_cost_penalty'] = round(rec.true_cost_penalty, 4)
                row['chemistry_valid'] = rec.chemistry_valid
                row['chemistry_note'] = rec.chemistry_note
                row['coherent'] = rec.coherent
                row['pv_disagreement'] = round(rec.pv_disagreement, 4)
                row['battery_disagreement'] = round(rec.battery_disagreement, 4)
                row['declared_new_best'] = rec.declared_new_best
                row['status_note'] = rec.status_note
                writer.writerow(row)
        print(f"  [SAVED] Tabular discovery records (CSV) -> {csv_path}")

        # --- TXT (human-readable, full structure + ASCII art) ---
        txt_path = os.path.join(save_dir, "prometheus_discoveries.txt")
        with open(txt_path, 'w') as f:
            f.write("PROJECT PROMETHEUS -- FULL DISCOVERY LOG\n")
            f.write("=" * 90 + "\n")
            f.write(f"Frozen PV target       : bandgap = {evaluator.FIXED_TARGET_BANDGAP} eV (Shockley-Queisser)\n")
            f.write(f"Frozen Battery target  : storage = {evaluator.FIXED_TARGET_STORAGE} mAh/g\n")
            f.write(f"Frozen Stability target: {evaluator.FIXED_OPTIMAL_STABILITY} eV (Goldschmidt-ideal)\n")
            f.write(f"Coherency tolerance    : {COHERENCY_TOL}\n")
            f.write(f"Total iterations       : {iterations}\n")
            f.write(f"Incoherent (rejected)  : {n_incoherent}\n")
            f.write(f"Chemistry-invalid      : {n_invalid_chem}\n")
            f.write("=" * 90 + "\n\n")
            for rec in all_records:
                f.write(evaluator.format_ascii_structure(rec))
                f.write("\n")
                f.write(evaluator.format_full_structure(rec))
                f.write("\n\n")
        print(f"  [SAVED] Human-readable structure log -> {txt_path}")
    except Exception as e:
        print(f"  [WARNING] Could not save discovery files: {e}")

    # ---- FINAL HONEST SUMMARY ----
    print("-" * 90)
    print("CONTINUOUS DISCOVERY COMPLETE -- HONEST SUMMARY")
    print("-" * 90)
    print(f"  Total iterations:            {iterations}")
    print(f"  Incoherent (AI != oracle):   {n_incoherent}  (these were NOT counted as discoveries)")
    print(f"  Chemistry-invalid:           {n_invalid_chem}  (these were NOT counted as discoveries)")
    n_valid = iterations - n_incoherent - n_invalid_chem
    print(f"  Valid discoveries:           {max(0, n_valid)}  (coherent + chemically valid)")
    if best_pv_score >= 0:
        print(f"  Best PV sub-score:           {best_pv_score:.3f} / 1.000  (frozen target: bandgap={evaluator.FIXED_TARGET_BANDGAP} eV)")
    else:
        print(f"  Best PV sub-score:           NONE (no coherent + chemically valid discovery made)")
    if best_battery_score >= 0:
        print(f"  Best Battery sub-score:      {best_battery_score:.3f} / 1.000  (frozen target: storage={evaluator.FIXED_TARGET_STORAGE} mAh/g)")
    else:
        print(f"  Best Battery sub-score:      NONE (no coherent + chemically valid discovery made)")
    if best_combined >= 0:
        print(f"  Best combined PV+Battery:    {best_combined:.3f} / 2.000")
    else:
        print(f"  Best combined PV+Battery:    NONE")
    if best_discovery:
        formula, eg, cap, stab, pv, bat, stab_s, cost_pen = best_discovery
        print(f"  Best combined formula:       {formula}")
        print(f"    Bandgap  = {eg:.3f} eV   (target {evaluator.FIXED_TARGET_BANDGAP})")
        print(f"    Storage  = {cap:.1f} mAh/g (target {evaluator.FIXED_TARGET_STORAGE})")
        print(f"    Stability= {stab_s:.3f}  | CostPen={cost_pen:.3f}")
        print(f"    PV sub-score     = {pv:.3f}")
        print(f"    Battery sub-score= {bat:.3f}")
    else:
        print(f"  Best combined formula:       NONE -- the AI did not produce any coherent + chemically valid discovery.")
        print(f"    (This is an honest failure. Increase n_generations or iterations to give the AI more training time.)")
    print("-" * 90)
    print("TRAINING EMPHASIS HISTORY (only this adapts -- targets are frozen):")
    print(f"  initial: {initial_weights}")
    print(f"  final  : {evaluator.training_weights}")
    print(f"  revisions: {len(evaluator.training_weight_history) - 1}")
    print("=" * 90)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\nInitiating Project PROMETHEUS for Universal Energy Discovery...\n")
    
    engine = EvolutionaryEngine(population_size=20, input_dim=16, output_dim=7)
    
    # Phase 1: Evolve the learning algorithm
    best_genome, best_fitness = engine.run(n_generations=21, verbose=True)
    
    # Phase 2: Continuous self-improvement and material discovery
    continuous_discovery_loop(best_genome, engine.evaluator, iterations=15)