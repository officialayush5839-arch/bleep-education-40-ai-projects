"""
common_utils/tests/test_synthetic.py
Unit tests verifying reproducibility, shapes, data types, and hermetic offline compliance
of common_utils.base_synthetic across all 4 domains + agent traces.
"""

import os
import tempfile
import socket
import pytest
import numpy as np
import pandas as pd
import torch
from PIL import Image

from common_utils.base_synthetic import (
    get_rng,
    set_seed,
    generate_tabular_data,
    generate_image_dataset,
    save_synthetic_image_folder,
    generate_time_series_data,
    generate_sensor_windows,
    create_lagged_features,
    generate_rag_corpus,
    generate_rag_qa_pairs,
    save_rag_corpus_to_files,
    generate_agent_trace,
    generate_agent_scenarios,
    save_agent_trace_json,
    DOMAIN_PRESETS,
    AGENT_PROJECT_SPECS,
)


# ============================================================================
# 1. TABULAR GENERATOR TESTS
# ============================================================================

class TestTabularGenerator:
    """Verifies tabular synthetic data generation."""
    
    def test_reproducibility_identical_seeds(self):
        df1 = generate_tabular_data(n_samples=50, seed=42)
        df2 = generate_tabular_data(n_samples=50, seed=42)
        pd.testing.assert_frame_equal(df1, df2)
        
    def test_seed_variance_different_seeds(self):
        df1 = generate_tabular_data(n_samples=50, seed=42)
        df2 = generate_tabular_data(n_samples=50, seed=43)
        assert not df1.equals(df2)
        
    def test_shapes_and_column_names(self):
        df = generate_tabular_data(
            n_samples=80,
            n_continuous=3,
            n_categorical=2,
            target_name="label",
            seed=101
        )
        assert len(df) == 80
        assert df.shape[1] == 3 + 2 + 1
        assert "label" in df.columns
        assert all(f"cont_feat_{i}" in df.columns for i in range(3))
        assert all(f"cat_feat_{i}" in df.columns for i in range(2))
        
    def test_imbalance_ratio_binary(self):
        # ML-01 ~3.4% positive
        df = generate_tabular_data(n_samples=500, imbalance_ratio=0.034, seed=42)
        n_pos = df["target"].sum()
        assert 15 <= n_pos <= 20
        assert set(df["target"].unique()).issubset({0, 1})
        
    def test_multiclass_classification(self):
        # ML-05 22 classes
        df = generate_tabular_data(n_samples=220, n_classes=22, seed=42)
        assert df["target"].nunique() <= 22
        assert df["target"].min() >= 0
        assert df["target"].max() < 22
        
    def test_regression_continuous_target(self):
        df = generate_tabular_data(
            n_samples=60,
            problem_type="regression",
            regression_skew=True,
            seed=42
        )
        assert df["target"].dtype == np.float64
        assert (df["target"] > 0).all()  # Skewed property price is positive
        
    def test_null_injection(self):
        null_ratio = 0.2
        df = generate_tabular_data(
            n_samples=100,
            n_continuous=4,
            n_categorical=2,
            null_ratio=null_ratio,
            seed=42
        )
        # Target must never have nulls
        assert df["target"].isnull().sum() == 0
        # Feature columns must have approx 20% nulls
        feature_cols = [c for c in df.columns if c != "target"]
        total_feature_cells = 100 * len(feature_cols)
        null_cells = df[feature_cols].isnull().sum().sum()
        observed_ratio = null_cells / total_feature_cells
        assert abs(observed_ratio - null_ratio) < 0.05
        
    def test_n_samples_zero_classification_and_regression(self):
        """Verifies n_samples=0 returns empty DataFrame with expected column schema for both problem types."""
        # Classification with default names
        df_cls = generate_tabular_data(
            n_samples=0,
            problem_type="classification",
            n_continuous=3,
            n_categorical=2,
            target_name="label"
        )
        assert len(df_cls) == 0
        assert list(df_cls.columns) == ["cont_feat_0", "cont_feat_1", "cont_feat_2", "cat_feat_0", "cat_feat_1", "label"]

        # Regression with custom names
        df_reg = generate_tabular_data(
            n_samples=0,
            problem_type="regression",
            continuous_feature_names=["f1", "f2"],
            categorical_feature_names=["c1"],
            target_name="price"
        )
        assert len(df_reg) == 0
        assert list(df_reg.columns) == ["f1", "f2", "c1", "price"]

    def test_imbalance_ratio_boundary_zero_and_one(self):
        """Verifies imbalance_ratio=0.0 yields 0 positives and 1.0 yields all positives."""
        # 0.0 -> 0 positives
        df_zero = generate_tabular_data(n_samples=100, imbalance_ratio=0.0, seed=42)
        assert len(df_zero) == 100
        assert (df_zero["target"] == 0).all()
        assert df_zero["target"].sum() == 0

        # 1.0 -> 100% positives
        df_one = generate_tabular_data(n_samples=100, imbalance_ratio=1.0, seed=42)
        assert len(df_one) == 100
        assert (df_one["target"] == 1).all()
        assert df_one["target"].sum() == 100

    def test_n_continuous_zero(self):
        """Verifies n_continuous=0 produces array probs and 1D target array without error."""
        # Classification
        df = generate_tabular_data(n_samples=50, n_continuous=0, n_categorical=3, seed=42)
        assert len(df) == 50
        assert df.shape[1] == 4  # 3 categorical + 1 target
        assert not any("cont_feat" in c for c in df.columns)
        assert len(df["target"]) == 50
        assert set(df["target"].unique()).issubset({0, 1})
        assert isinstance(df["target"].values, np.ndarray)

        # Regression
        df_reg = generate_tabular_data(n_samples=50, problem_type="regression", n_continuous=0, n_categorical=2, seed=42)
        assert len(df_reg) == 50
        assert df_reg.shape[1] == 3  # 2 categorical + 1 target
        assert len(df_reg["target"]) == 50

    def test_edge_cases(self):
        # 1 sample
        df_one = generate_tabular_data(n_samples=1, seed=42)
        assert len(df_one) == 1
        # 0 continuous features
        df_no_cont = generate_tabular_data(n_samples=10, n_continuous=0, n_categorical=3, seed=42)
        assert df_no_cont.shape[1] == 4


# ============================================================================
# 2. IMAGE GENERATOR TESTS
# ============================================================================

class TestImageGenerator:
    """Verifies image synthetic data generation."""
    
    def test_reproducibility(self):
        t1, l1 = generate_image_dataset(n_samples=8, return_format="tensor", seed=42)
        t2, l2 = generate_image_dataset(n_samples=8, return_format="tensor", seed=42)
        assert torch.equal(t1, t2)
        assert torch.equal(l1, l2)
        
    def test_tensor_shapes_and_bounds(self):
        imgs, labels = generate_image_dataset(
            n_samples=12,
            image_size=(28, 28),
            channels=1,
            n_classes=10,
            return_format="tensor",
            seed=42
        )
        assert imgs.shape == (12, 1, 28, 28)
        assert labels.shape == (12,)
        assert imgs.dtype == torch.float32
        assert labels.dtype == torch.long
        assert imgs.min() >= 0.0
        assert imgs.max() <= 1.0
        
    def test_numpy_and_pil_formats(self):
        np_imgs, np_labels = generate_image_dataset(
            n_samples=4,
            image_size=(32, 32),
            channels=3,
            return_format="numpy",
            seed=42
        )
        assert np_imgs.shape == (4, 32, 32, 3)
        assert len(np_labels) == 4
        
        pil_imgs, pil_labels = generate_image_dataset(
            n_samples=4,
            image_size=(32, 32),
            channels=3,
            return_format="pil",
            seed=42
        )
        assert len(pil_imgs) == 4
        assert isinstance(pil_imgs[0], Image.Image)
        assert pil_imgs[0].size == (32, 32)
        assert pil_imgs[0].mode == "RGB"
        
    def test_save_image_folder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = save_synthetic_image_folder(
                output_dir=tmpdir,
                n_samples_per_class=3,
                class_names=["cat", "dog"],
                seed=42
            )
            cat_img_path = os.path.join(tmpdir, "cat", "img_0000.png")
            dog_img_path = os.path.join(tmpdir, "dog", "img_0002.png")
            assert os.path.exists(cat_img_path)
            assert os.path.exists(dog_img_path)
            with Image.open(cat_img_path) as img:
                assert img.size == (32, 32)


# ============================================================================
# 3. TIME-SERIES & SENSOR GENERATOR TESTS
# ============================================================================

class TestTimeSeriesGenerator:
    """Verifies time-series and sensor window generation."""
    
    def test_reproducibility(self):
        df1 = generate_time_series_data(n_timesteps=48, seed=42)
        df2 = generate_time_series_data(n_timesteps=48, seed=42)
        pd.testing.assert_frame_equal(df1, df2)
        
    def test_datetime_index_and_frequency(self):
        df = generate_time_series_data(n_timesteps=96, freq="15min", seed=42)
        assert len(df) == 96
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.is_monotonic_increasing
        assert df.index.freqstr == "15min"
        
    def test_solar_profile_night_zero(self):
        df = generate_time_series_data(
            n_timesteps=96,
            freq="15min",
            is_solar_profile=True,
            seed=42
        )
        # Nighttime at 02:00 must have exactly 0.0 radiation
        night_rows = df[df.index.hour == 2]
        assert (night_rows["channel_0"] == 0.0).all()
        # Daytime at 12:00 must be positive
        noon_rows = df[df.index.hour == 12]
        assert (noon_rows["channel_0"] > 100.0).all()
        
    def test_sensor_windows_shape_and_classes(self):
        windows, labels = generate_sensor_windows(
            n_windows=8,
            window_size=128,
            n_channels=9,
            n_classes=6,
            seed=42
        )
        assert windows.shape == (8, 128, 9)
        assert labels.shape == (8,)
        assert windows.dtype == np.float32
        assert set(labels).issubset(set(range(6)))
        
    def test_create_lagged_features(self):
        df = generate_time_series_data(n_timesteps=30, n_channels=1, channel_names=["power"], seed=42)
        lag_df = create_lagged_features(df, target_col="power", lags=[1, 2], rolling_windows=[3])
        assert "power_lag_1" in lag_df.columns
        assert "power_lag_2" in lag_df.columns
        assert "power_rolling_mean_3" in lag_df.columns
        assert not lag_df.isnull().any().any()


# ============================================================================
# 4. RAG DOCUMENT CORPUS GENERATOR TESTS
# ============================================================================

class TestDocumentCorpusGenerator:
    """Verifies RAG document corpus and Q&A generation across domains."""
    
    def test_reproducibility(self):
        c1 = generate_rag_corpus(domain="hr", n_documents=2, seed=42)
        c2 = generate_rag_corpus(domain="hr", n_documents=2, seed=42)
        assert c1 == c2
        
    def test_corpus_schema(self):
        corpus = generate_rag_corpus(domain="medical", n_documents=3, pages_per_doc=2, seed=42)
        assert len(corpus) == 6
        for doc in corpus:
            assert "doc_id" in doc
            assert "title" in doc
            assert "page" in doc
            assert "section" in doc
            assert "text" in doc
            assert doc["domain"] == "medical"
            assert len(doc["text"]) > 20
            
    def test_all_ten_domains_supported(self):
        for domain in DOMAIN_PRESETS.keys():
            corpus = generate_rag_corpus(domain=domain, n_documents=1, pages_per_doc=1, seed=42)
            assert len(corpus) >= 1
            assert corpus[0]["domain"] == domain
            
    def test_qa_pairs_answerable_and_abstention(self):
        corpus = generate_rag_corpus(domain="legal", n_documents=2, pages_per_doc=2, seed=42)
        qa_pairs = generate_rag_qa_pairs(corpus, n_answerable=2, n_unanswerable=2, seed=42)
        assert len(qa_pairs) == 4
        
        # Check answerable pair
        ans = qa_pairs[0]
        assert ans["is_abstention"] is False
        assert ans["citation"] is not None
        assert ans["citation"].startswith("[Doc: ")
        assert "Section: " in ans["citation"]
        
        # Check unanswerable pair
        unans = qa_pairs[2]
        assert unans["is_abstention"] is True
        assert unans["citation"] is None
        assert "I cannot answer this question" in unans["ground_truth_answer"]
        
    def test_save_corpus_to_files(self):
        corpus = generate_rag_corpus(domain="finance", n_documents=2, pages_per_doc=1, seed=42)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = save_rag_corpus_to_files(tmpdir, corpus)
            files = os.listdir(out_dir)
            assert len(files) == 2
            with open(os.path.join(out_dir, files[0]), "r", encoding="utf-8") as f:
                content = f.read()
                assert "# " in content
                assert "**Page**:" in content


# ============================================================================
# 5. AGENT SCENARIOS & TRACE GENERATOR TESTS
# ============================================================================

class TestAgentScenarioGenerator:
    """Verifies multi-agent trace and scenario generation."""
    
    def test_reproducibility(self):
        t1 = generate_agent_trace(project_id="AG-01", seed=42)
        t2 = generate_agent_trace(project_id="AG-01", seed=42)
        assert t1 == t2
        
    def test_trace_schema(self):
        trace = generate_agent_trace(project_id="AG-02", scenario_type="happy_path", n_turns=3, seed=42)
        assert trace["project_id"] == "AG-02"
        assert trace["status"] == "COMPLETED"
        assert len(trace["steps"]) == 3
        for step in trace["steps"]:
            assert "step_number" in step
            assert "agent_name" in step
            assert "tool_name" in step
            assert "tool_input" in step
            assert "tool_output" in step
            assert step["status"] == "SUCCESS"
            
    def test_hitl_approval_and_rejection(self):
        # Approved gate
        trace_app = generate_agent_trace(project_id="AG-02", scenario_type="hitl_approved", seed=42)
        assert trace_app["status"] == "COMPLETED"
        assert len(trace_app["hitl_interventions"]) == 1
        assert trace_app["hitl_interventions"][0]["decision"] == "APPROVED"
        
        # Rejected gate
        trace_rej = generate_agent_trace(project_id="AG-02", scenario_type="hitl_rejected", seed=42)
        assert trace_rej["status"] == "ABORTED"
        assert len(trace_rej["hitl_interventions"]) == 1
        assert trace_rej["hitl_interventions"][0]["decision"] == "REJECTED"
        assert "aborted" in trace_rej["final_output"].lower()
        
    def test_tool_error_scenario(self):
        trace_err = generate_agent_trace(project_id="AG-03", scenario_type="tool_error", seed=42)
        err_steps = [s for s in trace_err["steps"] if s.get("status") == "ERROR"]
        assert len(err_steps) >= 1
        assert "error_message" in err_steps[0]
        
    def test_all_ten_agent_projects(self):
        for pid in AGENT_PROJECT_SPECS.keys():
            trace = generate_agent_trace(project_id=pid, seed=42)
            assert trace["project_id"] == pid
            assert len(trace["steps"]) >= 2
            
    def test_save_agent_trace_json(self):
        trace = generate_agent_trace(project_id="AG-09", seed=42)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "trace.json")
            save_agent_trace_json(out_file, trace)
            assert os.path.exists(out_file)
            assert os.path.getsize(out_file) > 100


# ============================================================================
# 6. HERMETIC OFFLINE COMPLIANCE TEST
# ============================================================================

def test_hermetic_offline_guarantee(monkeypatch):
    """
    Enforces that zero network socket connections are opened across all generators.
    """
    def forbidden_socket(*args, **kwargs):
        raise RuntimeError("CRITICAL ERROR: Synthetic generator attempted network socket creation!")
        
    monkeypatch.setattr(socket, "socket", forbidden_socket)
    
    # 1. Tabular
    df = generate_tabular_data(n_samples=20, seed=42)
    assert len(df) == 20
    
    # 2. Images
    t, l = generate_image_dataset(n_samples=4, seed=42)
    assert len(l) == 4
    
    # 3. Time Series
    ts = generate_time_series_data(n_timesteps=24, seed=42)
    assert len(ts) == 24
    
    # 4. RAG
    rag = generate_rag_corpus(domain="cybersecurity", n_documents=2, seed=42)
    assert len(rag) >= 2
    
    # 5. Agent Trace
    agent = generate_agent_trace(project_id="AG-08", seed=42)
    assert agent["project_id"] == "AG-08"
