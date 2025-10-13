"""Genetic algorithm optimizer for PPO hyperparameters."""

from __future__ import annotations

import math
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from crypto_rl.config.loader import load_config
from crypto_rl.pipelines.train import run_training
from crypto_rl.pipelines.test import run_backtests


@dataclass
class GAResult:
    individual: Dict[str, float]
    fitness: float
    growth: float
    volatility: float
    run_id: str
    meta: Dict[str, Any]


class SearchSpace:
    """Continuous search space with optional log-scaled parameters."""

    def __init__(self, bounds: Dict[str, Sequence[float]], log_scaled: Iterable[str] | None = None):
        self.bounds: Dict[str, Tuple[float, float]] = {}
        for key, value in bounds.items():
            if len(value) != 2:
                raise ValueError(f"Bounds for {key} must have length 2, got {value}")
            low, high = float(value[0]), float(value[1])
            if low >= high:
                raise ValueError(f"Bounds for {key} must satisfy low < high (got {low}, {high})")
            self.bounds[key] = (low, high)
        self.log_scaled = set(log_scaled or [])

    def random(self) -> Dict[str, float]:
        sample: Dict[str, float] = {}
        for key, (low, high) in self.bounds.items():
            if key in self.log_scaled:
                value = math.exp(random.uniform(math.log(low), math.log(high)))
            else:
                value = random.uniform(low, high)
            sample[key] = float(value)
        return sample

    def crossover(self, parent_a: Dict[str, float], parent_b: Dict[str, float]) -> Dict[str, float]:
        child = {}
        for key in self.bounds:
            if random.random() < 0.5:
                child[key] = parent_a[key]
            else:
                child[key] = parent_b[key]
        return child

    def mutate(self, individual: Dict[str, float], mutation_rate: float, mutation_scale: float) -> Dict[str, float]:
        mutated = dict(individual)
        for key, (low, high) in self.bounds.items():
            if random.random() > mutation_rate:
                continue
            if key in self.log_scaled:
                log_low, log_high = math.log(low), math.log(high)
                log_val = math.log(mutated[key])
                noise = random.gauss(0.0, mutation_scale * (log_high - log_low))
                log_val = min(max(log_val + noise, log_low), log_high)
                mutated[key] = float(math.exp(log_val))
            else:
                span = high - low
                noise = random.gauss(0.0, mutation_scale * span)
                value = min(max(mutated[key] + noise, low), high)
                mutated[key] = float(value)
        return mutated


def _build_overrides(individual: Dict[str, float], extra: Dict[str, Any]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for key, value in individual.items():
        overrides[key] = float(value)
    overrides.update(extra)
    return overrides


def _evaluate_worker(payload: Dict[str, Any]) -> GAResult:
    config_path: Optional[str] = payload["config_path"]
    profile: str = payload["profile"]
    run_id: str = payload["run_id"]
    individual: Dict[str, float] = payload["individual"]
    overrides: Dict[str, Any] = payload["overrides"]
    volatility_weight: float = payload["vol_weight"]
    output_root = payload["output_root"]

    try:
        training_result = run_training(
            config_path=config_path,
            profile=profile,
            run_id=run_id,
            overrides=overrides,
            log_to_console=False,
            output_root=output_root,
        )

        summary = run_backtests(
            config_path=config_path,
            run_id=run_id,
            profiles=[profile],
            deterministic=True,
            processes=1,
            overrides=overrides,
            log_to_console=False,
            output_root=output_root,
        )

        profile_summary = summary.get(profile)
        if not profile_summary:
            raise RuntimeError(f"Backtest summary missing for profile '{profile}'")

        tests_summary = profile_summary.get("tests") or {}
        if not tests_summary:
            raise RuntimeError(f"No test scenarios produced results for profile '{profile}'")

        test_metrics: Dict[str, Dict[str, float]] = {}
        fitness_values: List[float] = []
        growth_values: List[float] = []
        volatility_values: List[float] = []

        for test_name, test_info in tests_summary.items():
            trades_path = Path(test_info["trades_log_path"])
            if not trades_path.exists():
                raise FileNotFoundError(f"Trades log not found: {trades_path}")

            trades_df = pd.read_csv(trades_path)
            if "portfolio_value" not in trades_df.columns:
                raise ValueError(f"'portfolio_value' column missing in {trades_path}")

            equity = trades_df["portfolio_value"].to_numpy(dtype=float)
            if equity.size < 2:
                raise ValueError(
                    f"Trades log '{trades_path}' must contain at least two rows for evaluation."
                )

            initial_value = float(equity[0])
            final_value = float(equity[-1])
            growth = final_value / initial_value

            returns = np.diff(equity) / equity[:-1]
            volatility = float(np.std(returns)) if returns.size else 0.0
            fitness = growth - volatility_weight * volatility

            fitness_values.append(fitness)
            growth_values.append(growth)
            volatility_values.append(volatility)
            test_metrics[test_name] = {
                "trades_path": str(trades_path),
                "range": test_info.get("range"),
                "growth": growth,
                "volatility": volatility,
                "fitness": fitness,
                "final_value": final_value,
            }

        avg_fitness = float(np.mean(fitness_values))
        avg_growth = float(np.mean(growth_values))
        avg_volatility = float(np.mean(volatility_values))

        meta = {
            "training": training_result,
            "tests": test_metrics,
            "average_growth": avg_growth,
            "average_volatility": avg_volatility,
        }

        return GAResult(
            individual=individual,
            fitness=avg_fitness,
            growth=avg_growth,
            volatility=avg_volatility,
            run_id=run_id,
            meta=meta,
        )
    except Exception as exc:  # pragma: no cover - defensive
        return GAResult(
            individual=individual,
            fitness=float("-inf"),
            growth=0.0,
            volatility=float("inf"),
            run_id=run_id,
            meta={"error": repr(exc)},
        )


class GAOptimizer:
    """Simple real-valued genetic algorithm with parallel evaluation."""

    def __init__(
        self,
        *,
        config_path: Optional[str],
        profile: str,
        ga_config: Dict[str, Any],
    ):
        self.config_path = config_path
        self.profile = profile
        self.population_size = int(ga_config.get("population_size", 6))
        self.generations = int(ga_config.get("generations", 4))
        self.elite_fraction = float(ga_config.get("elite_fraction", 0.2))
        self.crossover_rate = float(ga_config.get("crossover_rate", 0.6))
        self.mutation_rate = float(ga_config.get("mutation_rate", 0.3))
        self.mutation_scale = float(ga_config.get("mutation_scale", 0.15))
        self.max_workers = int(ga_config.get("max_workers", 2))
        self.evaluation_epochs = int(ga_config.get("evaluation_epochs", 40))
        self.eval_validate_every = int(ga_config.get("evaluation_validate_every", 10))
        self.volatility_weight = float(ga_config.get("volatility_weight", 0.5))

        search_space_cfg = ga_config.get("search_space")
        if not search_space_cfg:
            raise ValueError("GA configuration must define a 'search_space' section.")
        log_scaled = ga_config.get("log_scaled", [])
        self.search_space = SearchSpace(search_space_cfg, log_scaled)

        elite_count = max(1, int(round(self.elite_fraction * self.population_size)))
        self.elite_count = min(elite_count, self.population_size)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.run_prefix = f"ga_{timestamp}"
        self.output_root = Path("outputs") / self.run_prefix

    def _evaluate_population(
        self,
        population: List[Dict[str, float]],
        generation: int,
    ) -> List[GAResult]:
        tasks = []
        extra_overrides = {
            "training.total_epochs": self.evaluation_epochs,
            "training.validate_every": max(1, min(self.evaluation_epochs, self.eval_validate_every)),
            "training.save_best_model": True,
        }

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            for idx, individual in enumerate(population):

                overrides = _build_overrides(individual, extra_overrides)
                run_id = f"{self.run_prefix}_g{generation:02d}_i{idx:02d}"
                payload = {
                    "config_path": self.config_path,
                    "profile": self.profile,
                    "run_id": run_id,
                    "individual": individual,
                    "overrides": overrides,
                    "vol_weight": self.volatility_weight,
                    "output_root": str(self.output_root),
                }
                tasks.append(executor.submit(_evaluate_worker, payload))

            results: List[GAResult] = []
            for future in as_completed(tasks):
                results.append(future.result())
        return results

    def _select_parents(self, population: List[Dict[str, float]], fitnesses: List[float]) -> Tuple[Dict[str, float], Dict[str, float]]:
        def tournament() -> Dict[str, float]:
            contenders = random.sample(list(zip(population, fitnesses)), k=min(3, len(population)))
            contenders.sort(key=lambda x: x[1], reverse=True)
            return contenders[0][0]

        return tournament(), tournament()

    def run(self) -> GAResult:
        population = [self.search_space.random() for _ in range(self.population_size)]
        best_overall: Optional[GAResult] = None

        for generation in range(self.generations):
            evaluation_results = self._evaluate_population(population, generation)
            evaluation_results.sort(key=lambda r: r.fitness, reverse=True)

            if best_overall is None or evaluation_results[0].fitness > best_overall.fitness:
                best_overall = evaluation_results[0]

            elites = evaluation_results[: self.elite_count]
            best = elites[0]
            print(
                f"[GA] Generation {generation + 1}/{self.generations} "
                f"| Best fitness={best.fitness:.4f} growth={best.growth:.4f} "
                f"volatility={best.volatility:.4f} run_id={best.run_id}"
            )

            # Prepare next population
            next_population: List[Dict[str, float]] = [dict(result.individual) for result in elites]
            fitness_values = [result.fitness for result in evaluation_results]
            parent_pool = [dict(result.individual) for result in evaluation_results]

            while len(next_population) < self.population_size:
                parent_a, parent_b = self._select_parents(parent_pool, fitness_values)
                if random.random() < self.crossover_rate:
                    child = self.search_space.crossover(parent_a, parent_b)
                else:
                    child = dict(parent_a)
                child = self.search_space.mutate(child, self.mutation_rate, self.mutation_scale)
                next_population.append(child)

            population = next_population

        assert best_overall is not None  # for type checkers
        return best_overall


__all__ = ["GAOptimizer", "GAResult"]
