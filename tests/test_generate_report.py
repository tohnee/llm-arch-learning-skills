#!/usr/bin/env python3
"""Unit tests for LLM architecture report generator."""
import json
import os
import sys
import tempfile
import unittest
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
import generate_report as gr


class TestFormattingHelpers(unittest.TestCase):
    """Test formatting helper functions."""

    def test_fmt_num(self):
        self.assertEqual(gr.fmt_num(None), "—")
        self.assertEqual(gr.fmt_num("123"), "123")
        self.assertEqual(gr.fmt_num(1234), "1,234")
        self.assertEqual(gr.fmt_num(123), "123")
        self.assertEqual(gr.fmt_num(12.5), "12.5")
        self.assertEqual(gr.fmt_num(12.0), "12")

    def test_fmt_ctx(self):
        self.assertEqual(gr.fmt_ctx(None), "—")
        self.assertEqual(gr.fmt_ctx(1000000), "1M")
        self.assertEqual(gr.fmt_ctx(128000), "128K")
        self.assertEqual(gr.fmt_ctx(256000), "256K")
        self.assertEqual(gr.fmt_ctx(512), "512")

    def test_lab_class(self):
        self.assertEqual(gr.lab_class("DeepSeek"), "lab-deepseek")
        self.assertEqual(gr.lab_class("OpenAI"), "lab-openai")
        self.assertEqual(gr.lab_class("Google (DeepMind)"), "lab-google")
        self.assertEqual(gr.lab_class("UnknownLab"), "lab-deepseek")

    def test_lab_color(self):
        self.assertEqual(gr.lab_color("DeepSeek"), "#5b8cff")
        self.assertEqual(gr.lab_color("Meta"), "#0668e1")
        self.assertTrue(gr.lab_color("Unknown").startswith("#"))

    def test_attention_label_zh(self):
        self.assertEqual(gr.attention_label("MLA", "zh"), "MLA")
        self.assertEqual(gr.attention_label("GQA", "zh"), "GQA")
        self.assertIn("线性", gr.attention_label("linear", "zh"))
        self.assertIn("混合", gr.attention_label("mamba-hybrid", "zh"))

    def test_attention_label_en(self):
        self.assertEqual(gr.attention_label("linear", "en"), "Linear")
        self.assertEqual(gr.attention_label("mamba-hybrid", "en"), "Mamba+GQA")

    def test_moe_label_dense(self):
        dense_model = {"is_moe": False}
        self.assertEqual(gr.moe_label(dense_model, "zh"), "稠密")
        self.assertEqual(gr.moe_label(dense_model, "en"), "Dense")

    def test_moe_label_moe(self):
        moe_model = {"is_moe": True, "num_experts": 256, "num_active_experts": 8, "num_shared_experts": 1}
        label = gr.moe_label(moe_model, "en")
        self.assertIn("256E", label)
        self.assertIn("8A", label)
        self.assertIn("+1S", label)


class TestDataLoading(unittest.TestCase):
    """Test snapshot loading and path resolution."""

    def test_get_paths_default_date(self):
        paths = gr.get_paths()
        self.assertIn("snapshots", paths["snapshot"])
        self.assertIn("assets", paths["template"])
        self.assertTrue(paths["snapshot"].endswith(".json"))
        self.assertTrue(paths["output"].endswith(".html"))

    def test_get_paths_custom_date(self):
        paths = gr.get_paths("2026-06-26")
        self.assertIn("2026-06-26", paths["snapshot"])
        self.assertIn("2026-06-26", paths["output"])

    def test_load_snapshot_valid(self):
        test_data = {
            "snapshot_date": "2026-06-26",
            "models": [
                {
                    "id": "test-model",
                    "lab": "TestLab",
                    "display_name": "Test Model",
                    "total_params_b": 100,
                    "active_params_b": 20,
                    "attention_type": "GQA",
                    "is_moe": False,
                    "context_window": 128000,
                    "release_date": "2026-01-01"
                }
            ],
            "openrouter_rankings": []
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(test_data, f)
            tmp_path = f.name
        try:
            loaded = gr.load_snapshot(tmp_path)
            self.assertEqual(loaded["snapshot_date"], "2026-06-26")
            self.assertEqual(len(loaded["models"]), 1)
            self.assertEqual(loaded["models"][0]["id"], "test-model")
        finally:
            os.unlink(tmp_path)

    def test_kv_cache_computation(self):
        """Test KV cache size calculation for different attention types."""
        gqa_model = {
            "num_layers": 62,
            "num_attention_heads": 128,
            "num_kv_heads": 8,
            "head_dim": 128,
            "attention_type": "GQA"
        }
        mla_model = {
            "num_layers": 61,
            "mla_latent_dim": 512,
            "attention_type": "MLA"
        }
        kv_gqa = gr.kv_cache_gb(gqa_model, 128000)
        kv_mla = gr.kv_cache_gb(mla_model, 128000)
        self.assertGreater(kv_gqa, 0)
        self.assertGreater(kv_mla, 0)
        self.assertLess(kv_mla, kv_gqa)


class TestChartDataGeneration(unittest.TestCase):
    """Test chart data builder functions via output verification."""

    def setUp(self):
        self.test_models = [
            {
                "id": "dense-model",
                "lab": "TestLab",
                "display_name": "Dense Model",
                "total_params_b": 70,
                "active_params_b": 70,
                "attention_type": "GQA",
                "is_moe": False,
                "context_window": 128000,
                "release_date": "2026-01-01",
                "pricing_input_per_m": 0.15,
                "pricing_output_per_m": 0.60,
                "num_layers": 62,
                "num_attention_heads": 128,
                "num_kv_heads": 8,
                "head_dim": 128
            },
            {
                "id": "moe-model",
                "lab": "DeepSeek",
                "display_name": "MoE Model",
                "total_params_b": 600,
                "active_params_b": 40,
                "attention_type": "MLA",
                "is_moe": True,
                "num_experts": 128,
                "num_active_experts": 6,
                "num_shared_experts": 2,
                "context_window": 256000,
                "release_date": "2026-03-01",
                "pricing_input_per_m": 0.25,
                "pricing_output_per_m": 1.00,
                "num_layers": 61,
                "mla_latent_dim": 512
            }
        ]
        self.test_rankings = [
            {"display_name": "MoE Model", "usage_rank": 3, "weekly_traffic": "10B", "price_reason": "Good price/performance"}
        ]

    def test_build_chart_js_contains_all_charts(self):
        chart_js = gr.build_chart_js(self.test_models, self.test_rankings, "zh")
        self.assertIn("groupedBars", chart_js)
        self.assertIn("CH.scatter", chart_js)
        self.assertIn("CH.lines", chart_js)
        self.assertIn("CH.timeline", chart_js)

    def test_build_chart_js_moe_xlog_enabled(self):
        chart_js = gr.build_chart_js(self.test_models, self.test_rankings, "zh")
        self.assertIn("xlog:true", chart_js)
        self.assertIn("MoE稀疏度", chart_js)

    def test_build_chart_js_moe_points(self):
        chart_js = gr.build_chart_js(self.test_models, self.test_rankings, "zh")
        self.assertIn("MoE Model", chart_js)
        self.assertIn("600", chart_js)
        self.assertIn("40", chart_js)

    def test_build_chart_js_kv_series(self):
        chart_js = gr.build_chart_js(self.test_models, self.test_rankings, "zh")
        self.assertIn("KV缓存", chart_js)
        self.assertIn("GQA", chart_js)
        self.assertIn("MLA", chart_js)

    def test_build_chart_js_timeline(self):
        chart_js = gr.build_chart_js(self.test_models, self.test_rankings, "zh")
        self.assertIn("2026-01-01", chart_js)
        self.assertIn("2026-03-01", chart_js)


class TestTableBuilders(unittest.TestCase):
    """Test HTML table row construction."""

    def setUp(self):
        self.test_models = [
            {
                "id": "test-model",
                "lab": "DeepSeek",
                "display_name": "DeepSeek V4",
                "total_params_b": 800,
                "active_params_b": 45,
                "attention_type": "MLA",
                "is_moe": True,
                "num_experts": 256,
                "num_active_experts": 8,
                "context_window": 256000,
                "release_date": "2026-05-20",
                "pricing_input_per_m": 0.27,
                "pricing_output_per_m": 1.10,
                "reasoning": "MTP-3"
            }
        ]

    def test_build_roster_rows(self):
        rows = gr.build_roster_rows(self.test_models, "zh")
        self.assertIn("DeepSeek V4", rows)
        self.assertIn("lab-deepseek", rows)
        self.assertIn("256E8A", rows)
        self.assertIn("256K", rows)

    def test_build_model_cards(self):
        cards = gr.build_model_cards(self.test_models, "zh")
        self.assertIn("DeepSeek V4", cards)
        self.assertIn("card", cards)

    def test_build_capability_rows(self):
        rows = gr.build_capability_rows(self.test_models, "zh")
        self.assertIn("DeepSeek V4", rows)
        self.assertIn("—", rows)

    def test_build_capability_rows_real_model(self):
        """Test capability rows with actual snapshot data."""
        snapshot_path = gr.get_paths("2026-06-26")["snapshot"]
        if not os.path.exists(snapshot_path):
            self.skipTest("Snapshot file not found")
        data = gr.load_snapshot(snapshot_path)
        rows = gr.build_capability_rows(data["models"], "zh")
        self.assertIn("DeepSeek V4", rows)
        self.assertIn("95", rows)


class TestReportGeneration(unittest.TestCase):
    """Test full report generation end-to-end."""

    def test_report_generation_creates_file(self):
        """Test that report generation completes and creates a valid HTML file."""
        snapshot_path = gr.get_paths("2026-06-26")["snapshot"]
        if not os.path.exists(snapshot_path):
            self.skipTest("Snapshot file not found")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_report.html")
            original_paths = gr.get_paths
            
            def mock_paths(date_str):
                paths = original_paths(date_str)
                paths["output"] = output_path
                return paths
            
            gr.get_paths = mock_paths
            try:
                result = gr.main("2026-06-26", "zh")
                self.assertTrue(os.path.exists(output_path))
                self.assertGreater(os.path.getsize(output_path), 10000)
                with open(output_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.assertIn("<!DOCTYPE html>", content)
                self.assertIn("DeepSeek V4", content)
            finally:
                gr.get_paths = original_paths

    def test_report_contains_chart_helpers(self):
        """Verify generated report includes CH chart library."""
        snapshot_path = gr.get_paths("2026-06-26")["snapshot"]
        if not os.path.exists(snapshot_path):
            self.skipTest("Snapshot file not found")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_report_charts.html")
            original_paths = gr.get_paths
            
            def mock_paths(date_str):
                paths = original_paths(date_str)
                paths["output"] = output_path
                return paths
            
            gr.get_paths = mock_paths
            try:
                gr.main("2026-06-26", "zh")
                with open(output_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.assertIn("const CH", content)
                self.assertIn("groupedBars", content)
                self.assertIn("scatter", content)
                self.assertIn("lines", content)
                self.assertIn("timeline", content)
                self.assertIn("xlog:true", content)
            finally:
                gr.get_paths = original_paths

    def test_report_english_language(self):
        """Test English report generation."""
        snapshot_path = gr.get_paths("2026-06-26")["snapshot"]
        if not os.path.exists(snapshot_path):
            self.skipTest("Snapshot file not found")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_report_en.html")
            original_paths = gr.get_paths
            
            def mock_paths(date_str):
                paths = original_paths(date_str)
                paths["output"] = output_path
                return paths
            
            gr.get_paths = mock_paths
            try:
                gr.main("2026-06-26", "en")
                with open(output_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.assertIn("LLM Architecture", content)
                self.assertIn("MoE sparsity", content)
            finally:
                gr.get_paths = original_paths


class TestModelDataConsistency(unittest.TestCase):
    """Verify all models in snapshot are complete and consistent."""

    def test_snapshot_models_have_required_fields(self):
        """Check that every model has all required data fields."""
        snapshot_path = gr.get_paths("2026-06-26")["snapshot"]
        if not os.path.exists(snapshot_path):
            self.skipTest("Snapshot file not found")
        
        data = gr.load_snapshot(snapshot_path)
        required_fields = [
            "id", "lab", "display_name", "total_params_b", "active_params_b",
            "attention_type", "is_moe", "context_window", "release_date"
        ]
        for model in data["models"]:
            for field in required_fields:
                self.assertIn(field, model, f"Model {model.get('id', 'unknown')} missing field {field}")
            if model["is_moe"]:
                self.assertIn("num_experts", model)
                self.assertIn("num_active_experts", model)
            self.assertGreater(model["total_params_b"], 0)
            self.assertGreater(model["active_params_b"], 0)
            self.assertGreater(model["context_window"], 0)

    def test_snapshot_models_have_capability_scores(self):
        """Check that every model in the latest snapshot has capability scores."""
        snapshot_path = gr.get_paths("2026-06-26")["snapshot"]
        if not os.path.exists(snapshot_path):
            self.skipTest("Snapshot file not found")
        
        data = gr.load_snapshot(snapshot_path)
        cap_defaults = {
            "deepseek-v4", "deepseek-v3", "kimi-k2", "qwen3-235b-a22b",
            "glm-5.1", "mimo-v2-pro", "mistral-large-3", "minimax-m2",
            "llama-4-maverick", "nemotron-3-ultra", "gpt-oss-120b",
            "gemma-4-31b", "step-3.5-flash", "qwen3-32b"
        }
        model_ids = {m["id"] for m in data["models"]}
        self.assertEqual(model_ids, cap_defaults, 
            f"Missing capability scores for: {model_ids - cap_defaults}")

    def test_model_count(self):
        """Verify expected number of models in snapshot."""
        snapshot_path = gr.get_paths("2026-06-26")["snapshot"]
        if not os.path.exists(snapshot_path):
            self.skipTest("Snapshot file not found")
        
        data = gr.load_snapshot(snapshot_path)
        self.assertEqual(len(data["models"]), 14)
        moe_count = sum(1 for m in data["models"] if m["is_moe"])
        self.assertGreaterEqual(moe_count, 8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
