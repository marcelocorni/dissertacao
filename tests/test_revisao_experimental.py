"""Testes pequenos, artificiais e sem rede. unittest, sem dependência de pytest."""
from __future__ import annotations

import json
import io
import os
import socket
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

from revisao_experimental.core import (CONFIGURATIONS, Store, atomic_json, block_priority, file_hash,
                                      grids, partition_blocks, select_candidate, validate_destination)
from revisao_experimental.metricas import (classification, clustered_contrast, discrimination,
                                          orient_baseline, paired_values, select_threshold, top_metrics)
from revisao_experimental.modelos import Checkpoint, fit_model, score_model

TEST_ROOT = Path(os.environ.get("REVISAO_TEST_ROOT", r"E:\Mestrado\revisao-01\etapa-05\testes"))
os.environ["MPLCONFIGDIR"] = str(TEST_ROOT / "matplotlib")


class RevisionTests(unittest.TestCase):
    def setUp(self):
        TEST_ROOT.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=TEST_ROOT)
        self.root = Path(self.tmp.name)
        # Any accidental network access fails instead of silently fetching data.
        self.network = patch.object(socket.socket, "connect", side_effect=AssertionError("Rede proibida nos testes"))
        self.network.start()

    def tearDown(self):
        self.network.stop()
        self.tmp.cleanup()

    def test_grid_contains_history_and_twelve_distinct_candidates(self):
        for method, candidates in grids().items():
            self.assertEqual(len(candidates), 12)
            self.assertEqual(len({json.dumps(c["parameters"], sort_keys=True) for c in candidates}), 12)
        self.assertEqual(grids()["AE"][0]["parameters"]["latent_dim"], 4)
        self.assertEqual(grids()["IF"][0]["parameters"], {"n_estimators": 200, "max_samples": 8192})

    def test_cli_plan_does_not_create_output_and_execute_is_explicit(self):
        from revisao_experimental import cli
        from fixtures_revisao import synthetic_source
        source = self.root / "execucoes" / "source-05pct"
        synthetic_source(source)
        destination = self.root / "execucoes" / "new-05pct"
        args = ["--source-run", str(source), "--output", str(destination)]
        with patch.object(cli, "ROOT", self.root), redirect_stdout(io.StringIO()) as output:
            self.assertEqual(cli.main(args), 0)
            self.assertEqual(json.loads(output.getvalue())["stage"], "plan")
            self.assertFalse(destination.exists())
            with self.assertRaises(SystemExit), patch.object(sys, "stderr", io.StringIO()):
                cli.main([*args, "--stage", "prepare"])
            self.assertFalse(destination.exists())

    def test_incomplete_search_cannot_select(self):
        with self.assertRaises(ValueError):
            select_candidate("AE", {})

    def test_selection_means_five_and_resolves_exact_ties(self):
        candidates = grids()["IF"]
        values = {c["id"]: dict.fromkeys(CONFIGURATIONS, .7) for c in candidates}
        self.assertEqual(select_candidate("IF", values)["id"], candidates[0]["id"])
        values[candidates[1]["id"]][CONFIGURATIONS[0]] = .9
        self.assertEqual(select_candidate("IF", values)["id"], candidates[1]["id"])
        values[candidates[1]["id"]].pop(CONFIGURATIONS[0])
        with self.assertRaises(ValueError):
            select_candidate("IF", values)

    def test_blocks_disjoint_whole_deterministic_and_if_target(self):
        counts = {20 * i: i % 7 + 2 for i in range(1, 1001)}
        part = partition_blocks(counts, 100)
        self.assertFalse(set(part["fit"]) & set(part["internal"]))
        self.assertEqual(set(part["fit"]) | set(part["internal"]), set(counts))
        self.assertTrue(set(part["if_fit"]) <= set(part["fit"]))
        self.assertGreaterEqual(sum(counts[b] for b in part["if_fit"]), 100)
        self.assertEqual(part, partition_blocks(dict(reversed(list(counts.items()))), 100))
        self.assertNotEqual(block_priority(20, "internal"), block_priority(20, "ae-search"))

    def test_destinations_reject_inputs_and_ancestors(self):
        source = self.root / "source"
        for dest in (source, source / "new", self.root):
            with self.assertRaises(ValueError):
                validate_destination(dest, [source])
        self.assertEqual(validate_destination(self.root / "new", [source]), self.root / "new")

    def test_stage_failure_resume_and_hash_integrity(self):
        store = Store(self.root / "out", {"protocol": 1})
        with self.assertRaises(RuntimeError):
            with store.stage("x") as out:
                atomic_json(out / "data.json", {"partial": True})
                raise RuntimeError("interrupção simulada")
        self.assertFalse(store.complete("x"))
        with store.stage("x") as out:
            atomic_json(out / "data.json", {"partial": False})
        self.assertTrue(store.complete("x"))
        atomic_json(out / "data.json", {"corrupted": True})
        with self.assertRaises(ValueError):
            store.complete("x")
        with self.assertRaises(ValueError):
            Store(self.root / "out", {"protocol": 2})

    def test_path_escape_and_running_lock(self):
        store = Store(self.root / "out", {})
        with self.assertRaises(ValueError):
            store.destination("../outside")
        with store.stage("x") as out:
            with self.assertRaises(FileExistsError):
                with store.stage("x"):
                    pass
            atomic_json(out / "ok.json", {"ok": 1})

    def test_ap_threshold_ties_baseline_and_top(self):
        y = np.array([0, 0, 1, 1])
        score = np.array([.1, .2, .8, .9])
        self.assertEqual(discrimination(y, score)["average_precision"], 1.)
        self.assertEqual(orient_baseline(y, -score)[0], -1)
        self.assertEqual(orient_baseline(y, np.ones(4))[0], 1)
        chosen, _ = select_threshold(y, np.ones(4), np.ones(20))
        self.assertEqual(chosen["threshold"], "q990")
        self.assertEqual(chosen["flagged_share"], 1.)
        self.assertEqual(top_metrics(y, score, .5)["positives"], 2)

    def test_paired_estimands_and_multiblock_rejection(self):
        result = paired_values(["e", "e", "e"], ["attacker_front", "victim", "attacker_back"], [.9, .5, .2], [20, 20, 20])
        self.assertEqual(len(result["margins"]), 2)
        self.assertEqual((result["margins"] > 0).mean(), .5)
        self.assertFalse(result["event_success"][0])
        with self.assertRaises(ValueError):
            paired_values(["e", "e"], ["attacker", "victim"], [.9, .5], [20, 40])

    def test_bootstrap_preserves_repeated_blocks(self):
        seen = []
        def statistic(indices):
            seen.append(indices.tolist())
            return {"ae": float(len(indices)), "if": float(len(indices)), "if_minus_ae": 0.}
        result = clustered_contrast([20, 20, 40], statistic, 30)
        self.assertTrue(any(len(indices) == 4 for indices in seen))
        self.assertEqual(result["intervals"]["if_minus_ae"], [0., 0.])
        self.assertEqual(result, clustered_contrast([20, 20, 40], statistic, 30))
        self.assertEqual(clustered_contrast([20], lambda i: None, 10)["status"], "requires_review")

    def test_checkpoint_matches_accepted_epoch_not_raw_minimum(self):
        checkpoint = Checkpoint(.1, 2)
        self.assertTrue(checkpoint.update(1., 1))
        self.assertFalse(checkpoint.update(.95, 2))
        self.assertFalse(checkpoint.update(.94, 3))
        self.assertEqual(checkpoint.epoch, 1)
        self.assertEqual(checkpoint.stale, 2)

    def test_tiny_cpu_training_both_algorithms(self):
        x = np.random.default_rng(42).normal(size=(80, 4)).astype(np.float32)
        internal = np.arange(80) % 10 == 0
        ae, info = fit_model("AE", grids()["AE"][0]["parameters"], x, internal,
                             budget={"max_epochs": 2, "batch_size": 16, "threads": 1})
        self.assertEqual(info["epochs"], 2)
        self.assertEqual(info["fit_rows"], 72)
        self.assertEqual(len(score_model("AE", ae, x)), 80)
        forest, info = fit_model("IF", {"n_estimators": 3, "max_samples": 16}, x, internal,
                                 budget={"threads": 1})
        self.assertEqual(info["max_samples_effective"], 16)
        self.assertEqual(len(score_model("IF", forest, x)), 80)
        with self.assertRaises(ValueError):
            fit_model("IF", {"n_estimators": 2, "max_samples": 100}, x, internal)

    def test_artificial_pipeline_prepare_fit_score_export_and_complements(self):
        import pyarrow.parquet as pq
        from fixtures_revisao import synthetic_source
        from revisao_experimental import complementos, fluxo
        from revisao_experimental.dados import prepare, SEMANTIC, connect
        from revisao_experimental.exportacao import export
        from revisao_experimental.figuras import generate
        source = self.root / "source"
        splits = synthetic_source(source)
        input_digests = {str(p): file_hash(p) for p in source.rglob("*") if p.is_file()}
        store = Store(self.root / "new", {"mode": "synthetic_test_only"})
        prepare(source, store, if_target=20)
        self.assertTrue(store.complete("prepare"))
        from revisao_experimental.dados import parquet
        preparation = json.loads((store.destination("prepare") / "preprocessing.json").read_text(encoding="utf-8"))
        with connect(self.root / "scaler-check") as con:
            fit_median = con.execute(f"SELECT median(gas_used_log) FROM {parquet(store.destination('prepare') / 'fit_c.parquet')}").fetchone()[0]
        self.assertEqual(fit_median, preparation["parameters"]["gas_used_log"]["median"])
        with store.stage("selection") as out:
            atomic_json(out / "winners.json", {"AE": grids()["AE"][0],
                        "IF": {"id": "tiny_test", "parameters": {"n_estimators": 3, "max_samples": 16}}})
        original_fit = fluxo.fit_model
        def tiny_fit(*args, **kwargs):
            return original_fit(*args, **kwargs, budget={"max_epochs": 2, "batch_size": 32, "threads": 1})
        with patch.object(fluxo, "fit_model", side_effect=tiny_fit):
            fluxo.final_fit(source, store, "cpu")
        fluxo.score_all(source, store, "cpu")
        export(source, store)
        self.assertTrue(store.complete("exports"))
        raw_transaction_intervals = complementos.transaction_intervals
        raw_cluster = complementos.clustered_contrast
        with patch.object(complementos, "transaction_intervals", side_effect=lambda *a: raw_transaction_intervals(*a, repetitions=20)), \
             patch.object(complementos, "clustered_contrast", side_effect=lambda *a: raw_cluster(*a, repetitions=20)):
            complementos.evaluate(source, store)
        complementos.count_reuse_and_types(source, store)
        generate(source, store)
        self.assertTrue(store.complete("complements"))
        self.assertTrue(store.complete("descriptive"))
        self.assertTrue(store.complete("figures-semantic"))
        # Exercise the actual report mains in-process on tiny artificial inputs;
        # the socket blocker remains active and no scientific source is opened.
        from revisao_experimental import exportacao
        from revisao_experimental.dados import legacy, ROOT
        def local_report(command, **kwargs):
            relative = str(Path(command[1]).relative_to(ROOT))
            with patch.object(sys, "argv", command[1:]), redirect_stdout(io.StringIO()):
                self.assertEqual(legacy(relative).main(), 0)
        with patch.object(exportacao.subprocess, "run", side_effect=local_report):
            exportacao.legacy_reports(source, store)
        for name in ("figures-agreement", "external-labelcloud", "drift", "correlation-figures"):
            self.assertTrue(store.complete(name))
        self.assertEqual(input_digests, {str(p): file_hash(p) for p in source.rglob("*") if p.is_file()})
        # Conflict excludes the whole event; an unrelated missing score is an error.
        validation = next(s for s in splits if s["role"] == "validation")
        label_path = source / SEMANTIC / "03_rotulos_transacao.parquet"
        labels = pq.read_table(label_path).to_pylist()
        affected = next(r for r in labels if r["split"] == validation["key"])
        affected["sensitivity_conflict"] = True
        import pyarrow as pa
        pq.write_table(pa.Table.from_pylist(labels), label_path)
        with connect(self.root / "check-temp") as con:
            pop = complementos.get_scores(con, source, store, validation, "a_completa", "AE", "sensitivity")
            roles, audit = complementos.role_population(con, source, validation, "sensitivity", pop)
            self.assertEqual(audit["events_excluded_incomplete_or_conflicting"], 1)
            self.assertEqual(len(roles["event_key"]), 8)
            with self.assertRaises(ValueError):
                complementos.role_population(con, source, validation, "sensitivity", {k: v[:-1] for k,v in pop.items()})

    def test_search_driver_simulated_complete_grid_and_resume(self):
        from revisao_experimental import fluxo
        store = Store(self.root / "search-simulation", {"mode": "mock_no_scientific_fit"})
        with store.stage("prepare") as out:
            atomic_json(out / "preprocessing.json", {"configurations": [{"key": k, "features": ["x"]} for k in CONFIGURATIONS]})
            atomic_json(out / "blocks.json", {})
        splits = [{"role": "train"}, {"role": "validation"}]
        y = np.array([0, 0, 1, 1])
        x = np.array([[.1], [.2], [.8], [.9]], dtype=np.float32)
        with redirect_stdout(io.StringIO()), \
             patch.object(fluxo, "load_source", return_value=({}, splits)), \
             patch.object(fluxo, "training_arrays", return_value=(x, np.array([0, 0, 0, 1], dtype=bool))), \
             patch.object(fluxo, "validation_arrays", return_value=(y, x)), \
             patch.object(fluxo, "fit_model", return_value=(None, {"history": [], "fit_seconds": 0.})) as fit, \
             patch.object(fluxo, "score_model", return_value=x[:, 0]), \
             patch.object(fluxo.torch.cuda, "is_available", return_value=False):
            fluxo.search(self.root, store, "cpu")
            self.assertEqual(fit.call_count, 120)
            self.assertTrue(store.complete("selection"))
            fluxo.search(self.root, store, "cpu")
            self.assertEqual(fit.call_count, 120)

    def test_tiny_cuda_training_if_available(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA não disponível")
        x = np.random.default_rng(42).normal(size=(40, 4)).astype(np.float32)
        model, info = fit_model("AE", grids()["AE"][0]["parameters"], x, np.arange(40) % 5 == 0,
                               device="cuda", budget={"max_epochs": 2, "batch_size": 16, "threads": 1})
        self.assertEqual(info["epochs"], 2)
        self.assertGreater(info["peak_gpu_memory_bytes"], 0)
        self.assertTrue(np.isfinite(score_model("AE", model, x)).all())


if __name__ == "__main__":
    unittest.main()
