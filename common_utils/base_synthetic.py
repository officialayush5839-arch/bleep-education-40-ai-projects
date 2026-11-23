"""
common_utils/base_synthetic.py
Universal Deterministic Synthetic Data Generator for Bleep Education AI Projects.

Provides seed-based, 100% offline, hermetic synthetic data generators across 4 domains:
1. Tabular: Classification (imbalanced & multiclass), regression, mixed types, null injection.
2. Images: RGB/Grayscale tensors, PIL images, and PyTorch ImageFolder directory layout.
3. Time Series & Sensors: Datetime-indexed sequences, diurnal cycles, 3D sensor windows.
4. Text & Document Corpus: Domain-specific RAG knowledge bases with ground-truth Q&A and citations.
5. Agent Traces & Scenarios: Multi-turn agent execution traces with HITL approval gates.
"""

import os
import json
import math
from typing import List, Dict, Tuple, Optional, Union, Any
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw


# ============================================================================
# 1. DETERMINISTIC RNG UTILITIES
# ============================================================================

def get_rng(seed: Optional[int] = 42) -> np.random.Generator:
    """Return an isolated numpy random Generator instance."""
    return np.random.default_rng(seed)


def set_seed(seed: int = 42) -> None:
    """Set global seeds across python, numpy, and torch for safety."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================================
# 2. TABULAR DATA GENERATOR
# ============================================================================

def generate_tabular_data(
    n_samples: int = 100,
    problem_type: str = "classification",  # "classification" | "regression"
    n_continuous: int = 5,
    n_categorical: int = 2,
    n_classes: int = 2,
    imbalance_ratio: Optional[float] = None,  # e.g., 0.0339 for ML-01, 0.00172 for ML-03
    categorical_cardinalities: Optional[List[int]] = None,
    continuous_feature_names: Optional[List[str]] = None,
    categorical_feature_names: Optional[List[str]] = None,
    target_name: str = "target",
    null_ratio: float = 0.0,
    regression_skew: bool = False,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate deterministic synthetic tabular data supporting classification and regression.
    
    Parameters:
        n_samples: Total number of rows.
        problem_type: "classification" or "regression".
        n_continuous: Number of continuous numeric features.
        n_categorical: Number of categorical string features.
        n_classes: Number of distinct classes (for classification).
        imbalance_ratio: Positive class fraction for binary classification (e.g. 0.05).
        categorical_cardinalities: List of category counts per categorical column.
        continuous_feature_names: Optional custom names for continuous columns.
        categorical_feature_names: Optional custom names for categorical columns.
        target_name: Name of target column.
        null_ratio: Fraction of feature values to randomly set to NaN.
        regression_skew: If True, apply log-normal skew to regression target.
        seed: Random seed for deterministic reproducibility.
        
    Returns:
        pd.DataFrame containing features and target.
    """
    cont_names = continuous_feature_names or [f"cont_feat_{i}" for i in range(n_continuous)]
    if len(cont_names) != n_continuous:
        n_continuous = len(cont_names)
        
    cat_names = categorical_feature_names or [f"cat_feat_{i}" for i in range(n_categorical)]
    if len(cat_names) != n_categorical:
        n_categorical = len(cat_names)

    # Guard against n_samples == 0
    if n_samples == 0:
        cols = cont_names + cat_names + [target_name]
        return pd.DataFrame(columns=cols)
        
    rng = get_rng(seed)
    
    # 1. Continuous Features
    cont_data = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_continuous))
        
    if categorical_cardinalities is None:
        categorical_cardinalities = [3] * n_categorical
    elif len(categorical_cardinalities) != n_categorical:
        categorical_cardinalities = (categorical_cardinalities + [3] * n_categorical)[:n_categorical]
        
    cat_dict = {}
    for i, name in enumerate(cat_names):
        card = max(2, categorical_cardinalities[i])
        categories = [f"CAT_{chr(65+j)}" for j in range(card)]
        cat_dict[name] = rng.choice(categories, size=n_samples)
        
    # 3. Target Generation
    if problem_type == "classification":
        if n_classes == 2:
            if imbalance_ratio is not None and 0.0 <= imbalance_ratio <= 1.0:
                if imbalance_ratio == 0.0:
                    n_pos = 0
                elif imbalance_ratio == 1.0:
                    n_pos = n_samples
                else:
                    n_pos = max(1, min(n_samples - 1, int(round(n_samples * imbalance_ratio)))) if n_samples > 1 else int(round(n_samples * imbalance_ratio))
                n_neg = n_samples - n_pos
                y = np.array([1] * n_pos + [0] * n_neg, dtype=np.int64)
                rng.shuffle(y)
            else:
                # Balanced or uniform binary
                probs = 1.0 / (1.0 + np.exp(-cont_data[:, 0] if n_continuous > 0 else np.zeros(n_samples)))
                y = (probs >= 0.5).astype(np.int64)
        else:
            # Multiclass
            y = rng.integers(0, n_classes, size=n_samples, dtype=np.int64)
    else:
        # Regression
        if n_continuous > 0:
            weights = rng.uniform(0.5, 2.0, size=n_continuous)
            signal = np.dot(cont_data, weights)
        else:
            signal = np.zeros(n_samples)
        noise = rng.normal(0, 0.1, size=n_samples)
        y_val = signal + noise
        if regression_skew:
            # Shift to positive and apply exponential to emulate right-skewed prices
            y_val = np.exp(y_val - np.min(y_val) + 1.0) * 1000.0
        y = y_val.astype(np.float64)
        
    # 4. Construct DataFrame
    df_data = {}
    for i, name in enumerate(cont_names):
        df_data[name] = cont_data[:, i]
    for name, vals in cat_dict.items():
        df_data[name] = vals
    df_data[target_name] = y
    
    df = pd.DataFrame(df_data)
    
    # 5. Inject Nulls into features if requested
    if null_ratio > 0.0:
        feature_cols = cont_names + cat_names
        for col in feature_cols:
            mask = rng.uniform(0.0, 1.0, size=n_samples) < null_ratio
            df.loc[mask, col] = np.nan
            
    return df


# ============================================================================
# 3. IMAGE SYNTHETIC GENERATOR
# ============================================================================

def generate_image_dataset(
    n_samples: int = 16,
    image_size: Tuple[int, int] = (32, 32),
    channels: int = 3,  # 1 or 3
    n_classes: int = 2,
    return_format: str = "tensor",  # "tensor" | "numpy" | "pil"
    seed: int = 42
) -> Tuple[Union[torch.Tensor, np.ndarray, List[Image.Image]], Union[torch.Tensor, np.ndarray, List[int]]]:
    """
    Generate deterministic synthetic images with class-separable geometric patterns.
    
    Parameters:
        n_samples: Number of image samples.
        image_size: (Height, Width) tuple.
        channels: 1 (grayscale) or 3 (RGB).
        n_classes: Number of distinct classes.
        return_format: "tensor" (PyTorch), "numpy" (NumPy array), or "pil" (PIL Images).
        seed: Random seed for reproducibility.
        
    Returns:
        (images, labels) tuple in requested format.
    """
    rng = get_rng(seed)
    H, W = image_size
    labels = np.array([i % n_classes for i in range(n_samples)], dtype=np.int64)
    rng.shuffle(labels)
    
    # Base array (N, C, H, W) normalized to [0, 1]
    images = np.zeros((n_samples, channels, H, W), dtype=np.float32)
    
    for i in range(n_samples):
        # Base background noise
        images[i] = rng.uniform(0.05, 0.25, size=(channels, H, W)).astype(np.float32)
        c = labels[i]
        
        # Embed class-separable geometric pattern
        pattern_type = c % 4
        if pattern_type == 0:
            # Horizontal stripe
            images[i, :, H // 3 : 2 * H // 3, :] += 0.5
        elif pattern_type == 1:
            # Vertical stripe
            images[i, :, :, W // 3 : 2 * W // 3] += 0.5
        elif pattern_type == 2:
            # Center rectangle / square
            images[i, :, H // 4 : 3 * H // 4, W // 4 : 3 * W // 4] += 0.6
        elif pattern_type == 3:
            # Corner accents
            images[i, :, : H // 3, : W // 3] += 0.6
            images[i, :, 2 * H // 3 :, 2 * W // 3 :] += 0.6
            
    images = np.clip(images, 0.0, 1.0)
    
    if return_format == "tensor":
        tensor_imgs = torch.from_numpy(images).float()
        tensor_labels = torch.from_numpy(labels).long()
        return tensor_imgs, tensor_labels
        
    elif return_format == "numpy":
        # Return channels-last (N, H, W, C)
        np_imgs = np.transpose(images, (0, 2, 3, 1))
        return np_imgs, labels
        
    elif return_format == "pil":
        pil_images = []
        for i in range(n_samples):
            img_arr = (images[i] * 255.0).astype(np.uint8)
            if channels == 1:
                img_obj = Image.fromarray(img_arr[0], mode="L")
            else:
                img_obj = Image.fromarray(np.transpose(img_arr, (1, 2, 0)), mode="RGB")
            pil_images.append(img_obj)
        return pil_images, labels.tolist()
    else:
        raise ValueError(f"Unknown return_format: {return_format}")


def save_synthetic_image_folder(
    output_dir: str,
    n_samples_per_class: int = 10,
    image_size: Tuple[int, int] = (32, 32),
    channels: int = 3,
    class_names: Optional[List[str]] = None,
    seed: int = 42
) -> str:
    """
    Save synthetic images to disk in standard PyTorch ImageFolder hierarchy:
    <output_dir>/<class_name>/img_001.png
    """
    if class_names is None:
        class_names = ["class_0", "class_1"]
        
    os.makedirs(output_dir, exist_ok=True)
    rng = get_rng(seed)
    
    for c_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        imgs, _ = generate_image_dataset(
            n_samples=n_samples_per_class,
            image_size=image_size,
            channels=channels,
            n_classes=1,
            return_format="pil",
            seed=seed + c_idx * 100
        )
        for i, img in enumerate(imgs):
            file_path = os.path.join(class_dir, f"img_{i:04d}.png")
            img.save(file_path, format="PNG")
            
    return os.path.abspath(output_dir)


# ============================================================================
# 4. TIME-SERIES & SENSOR SYNTHETIC GENERATOR
# ============================================================================

def generate_time_series_data(
    n_timesteps: int = 100,
    n_channels: int = 1,
    freq: str = "15min",
    start_date: str = "2026-01-01 00:00:00",
    has_seasonality: bool = True,
    is_solar_profile: bool = False,
    noise_level: float = 0.1,
    channel_names: Optional[List[str]] = None,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate sequential multi-channel time-series data with timestamp indexing.
    Supports diurnal cycles and solar profile zeroing at night.
    """
    rng = get_rng(seed)
    dates = pd.date_range(start=start_date, periods=n_timesteps, freq=freq)
    
    cols = channel_names or [f"channel_{i}" for i in range(n_channels)]
    if len(cols) != n_channels:
        n_channels = len(cols)
        
    data = {}
    time_indices = np.arange(n_timesteps)
    
    for i, col in enumerate(cols):
        # Trend
        trend = 0.02 * time_indices
        # Seasonality
        if has_seasonality:
            # 24-step or periodic cycle
            season = np.sin(2 * np.pi * time_indices / 24.0 + i)
        else:
            season = np.zeros(n_timesteps)
        # Noise
        noise = rng.normal(0, noise_level, size=n_timesteps)
        series = 10.0 + trend + season + noise
        
        if is_solar_profile:
            # Diurnal solar cycle: 0 radiation outside 06:00 to 18:00
            hours = dates.hour + dates.minute / 60.0
            solar_mask = (hours >= 6.0) & (hours <= 18.0)
            solar_curve = np.maximum(0.0, np.sin(np.pi * (hours - 6.0) / 12.0))
            series = np.where(solar_mask, 500.0 * solar_curve + noise * 10.0, 0.0)
            series = np.maximum(0.0, series)
            
        data[col] = series
        
    df = pd.DataFrame(data, index=dates)
    df.index.name = "timestamp"
    return df


def generate_sensor_windows(
    n_windows: int = 16,
    window_size: int = 128,
    n_channels: int = 9,
    n_classes: int = 6,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate 3D sensor windows for Human Activity Recognition (HAR DL-07).
    Shape: (n_windows, window_size, n_channels) with labels in [0, n_classes-1].
    """
    rng = get_rng(seed)
    labels = np.array([i % n_classes for i in range(n_windows)], dtype=np.int64)
    rng.shuffle(labels)
    
    windows = np.zeros((n_windows, window_size, n_channels), dtype=np.float32)
    
    for i in range(n_windows):
        c = labels[i]
        freq = (c + 1) * 0.5  # Activity-specific oscillation frequency
        t = np.linspace(0, 4 * np.pi, window_size)
        base_signal = np.sin(freq * t)
        for ch in range(n_channels):
            noise = rng.normal(0, 0.1, size=window_size)
            windows[i, :, ch] = base_signal * (1.0 + 0.1 * ch) + noise
            
    return windows, labels


def create_lagged_features(
    df: pd.DataFrame,
    target_col: str,
    lags: List[int] = [1, 2, 3],
    rolling_windows: List[int] = [3, 5]
) -> pd.DataFrame:
    """
    Create time-lagged and rolling statistics features from a time series DataFrame.
    """
    out_df = df.copy()
    for lag in lags:
        out_df[f"{target_col}_lag_{lag}"] = out_df[target_col].shift(lag)
    for rw in rolling_windows:
        out_df[f"{target_col}_rolling_mean_{rw}"] = out_df[target_col].rolling(rw).mean()
        out_df[f"{target_col}_rolling_std_{rw}"] = out_df[target_col].rolling(rw).std()
    return out_df.dropna()


# ============================================================================
# 5. TEXT & DOCUMENT CORPUS GENERATOR (RAG)
# ============================================================================

DOMAIN_PRESETS = {
    "hr": {
        "title": "Global HR Enterprise Policies 2026",
        "sections": [
            ("Annual Leave Policy", "Full-time employees receive 25 days of paid annual leave per calendar year. Unused leave can be rolled over up to a maximum of 5 days into Q1 of the following year."),
            ("Sick and Medical Leave", "Employees are granted 10 paid sick days annually. A registered medical certificate is mandatory for consecutive absences exceeding 3 days."),
            ("Remote Work Guidelines", "Employees may work remotely up to 2 days per week upon prior written approval from their immediate supervisor and IT security clearance."),
            ("Code of Professional Conduct", "All employees must maintain the highest standards of professional ethics. Conflicts of interest must be formally disclosed to HR within 14 days.")
        ]
    },
    "medical": {
        "title": "Clinical Research Pharmacotherapy Guidelines",
        "sections": [
            ("Dosing Schedule", "The standard adult therapeutic dose is 50 mg administered orally twice daily with meals. Renal dose adjustments are indicated for creatinine clearance below 30 mL/min."),
            ("Adverse Event Reporting", "Adverse reactions including mild nausea, headache, and transient elevation in transaminases must be documented within 24 hours in the clinical study registry."),
            ("Contraindications", "Concurrent administration with strong CYP3A4 inhibitors is strictly contraindicated due to heightened risk of systemic toxicity."),
            ("Screening Protocol", "Prior to drug initiation, patients must undergo baseline liver function testing, serum electrolyte analysis, and an anterior-posterior chest radiograph.")
        ]
    },
    "legal": {
        "title": "Standard Commercial Service Agreement",
        "sections": [
            ("Limitation of Liability", "Neither party shall be liable for indirect, incidental, or consequential damages. Total aggregate liability under this agreement is capped at fees paid in the prior 12 months."),
            ("Confidentiality Obligations", "Proprietary information disclosed by either party must be protected for a period of 5 years following disclosure using reasonable industry care."),
            ("Termination Clause", "Either party may terminate this agreement with 30 days prior written notice in the event of an uncured material breach by the other party."),
            ("Governing Law and Jurisdiction", "This contract shall be construed and governed according to the laws of the State of Delaware, without regard to choice of law provisions.")
        ]
    },
    "finance": {
        "title": "Annual Financial Report and MD&A Disclosure",
        "sections": [
            ("Revenue and Operating Performance", "Total consolidated revenue for fiscal year 2025 reached $4.2 billion, representing an 8.5% year-over-year increase driven by cloud services expansion."),
            ("Operating Margin and EBITDA", "Adjusted EBITDA grew by 12.1% to $1.15 billion, with operating margin expanding 140 basis points to 27.4% due to operational efficiencies."),
            ("Liquidity and Capital Resources", "Cash and cash equivalents totaled $620 million as of December 31, 2025. Free cash flow generation remained robust at $480 million."),
            ("Market Risk Exposures", "The company manages foreign exchange and interest rate volatility through structured hedge accounting and derivative instruments.")
        ]
    },
    "agriculture": {
        "title": "Integrated Agronomic Extension Manual",
        "sections": [
            ("Soil Nutrient Management", "Optimal wheat production requires an N-P-K nutrient ratio of 120:60:40 kg/ha. Nitrogen should be split across basal and top-dressing applications."),
            ("Soil pH Amelioration", "For acidic soils with pH below 5.5, agricultural lime application at 2.5 tonnes/ha is recommended before the monsoon sowing window."),
            ("Pest and Disease Control", "Early blight in tomato crops should be managed using prophylactic bio-fungicide sprays followed by copper oxychloride if symptoms persist."),
            ("Irrigation Scheduling", "Drip irrigation delivers 35% higher water use efficiency compared to conventional flood methods and maintains root-zone moisture at 70% field capacity.")
        ]
    },
    "cybersecurity": {
        "title": "Enterprise Cybersecurity Operations Manual",
        "sections": [
            ("Incident Response Lifecycle", "All detected high-severity alerts must follow the 4-phase NIST framework: Preparation, Detection & Analysis, Containment, and Post-Incident Recovery."),
            ("Access Control and MFA", "Multi-Factor Authentication (MFA) with FIDO2 hardware tokens is mandatory for all administrative access and remote SSH sessions."),
            ("Phishing Triage Protocols", "Reported suspicious emails must be quarantined automatically, and extracted URLs detonated in an isolated malware analysis sandbox."),
            ("Zero-Trust Architecture", "Network perimeter security enforces continuous mutual TLS authentication and micro-segmentation across all workload clusters.")
        ]
    },
    "developer_docs": {
        "title": "REST API Integration Reference",
        "sections": [
            ("Authentication Tokens", "API requests must provide a valid Bearer token in the Authorization header. Expired tokens return HTTP 401 Unauthorized."),
            ("Rate Limiting Policies", "Standard API tiers are throttled to 100 requests per minute. Responses include headers X-RateLimit-Limit and X-RateLimit-Remaining."),
            ("Pagination Syntax", "Collection endpoints accept query parameters 'limit' (default 20, max 100) and 'cursor' for opaque pagination tokens."),
            ("Error Codes Schema", "Errors return structured JSON containing 'error_code', 'message', and a correlation 'request_id' for diagnostic tracing.")
        ]
    },
    "climate": {
        "title": "Climate Science Assessment Report",
        "sections": [
            ("Global Warming Trajectories", "Global surface temperature has risen by 1.1 degrees Celsius compared to pre-industrial baselines, predominantly driven by greenhouse gas emissions."),
            ("Sea Level Rise Projections", "Under SSP2-4.5 scenarios, global mean sea level is projected to rise between 0.44 and 0.76 meters by 2100 relative to 1995-2014 levels."),
            ("Renewable Energy Integration", "Decarbonization pathways require solar and wind electricity generation to comprise at least 65% of global power capacity by 2040."),
            ("Carbon Budget Constraints", "The remaining carbon budget for a 50% probability of limiting warming to 1.5 degrees Celsius is estimated at 380 Gt CO2.")
        ]
    },
    "edtech": {
        "title": "Computer Science Curriculum Syllabus",
        "sections": [
            ("Asymptotic Complexity", "Analysis of divide-and-conquer algorithms utilizes the Master Theorem. Merge sort guarantees O(N log N) worst-case time complexity."),
            ("Tree Data Structures", "Self-balancing AVL and Red-Black trees ensure O(log N) lookup, insertion, and deletion by maintaining tree height invariants."),
            ("Graph Traversal Algorithms", "Breadth-First Search (BFS) computes shortest paths in unweighted graphs, while Dijkstra's algorithm solves weighted non-negative edge graphs."),
            ("Bloom's Taxonomy Assessment", "Assessments are structured across cognitive levels: Remember, Understand, Apply, Analyze, Evaluate, and Create.")
        ]
    },
    "academic": {
        "title": "University Academic Regulations and Degree Requirements",
        "sections": [
            ("Credit Requirements", "Undergraduate degree completion mandates a minimum of 120 semester credit hours with a cumulative GPA of at least 2.0 on a 4.0 scale."),
            ("Course Withdrawal Deadline", "Students may drop courses with a 'W' grade until the end of the 10th week of instruction without affecting grade point average."),
            ("Academic Integrity Policy", "Plagiarism and unauthorized collaboration result in immediate course failure and formal referral to the University Disciplinary Committee."),
            ("Honors Eligibility", "Graduation with Cum Laude honors requires a GPA of 3.50; Magna Cum Laude requires 3.75; Summa Cum Laude requires 3.90.")
        ]
    }
}


def generate_rag_corpus(
    domain: str = "hr",
    n_documents: int = 4,
    pages_per_doc: int = 2,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Generate deterministic document corpus for RAG ingestion.
    Each item contains: [doc_id, title, page, section, text, domain].
    """
    domain_key = domain.lower()
    if domain_key not in DOMAIN_PRESETS:
        domain_key = "hr"
        
    preset = DOMAIN_PRESETS[domain_key]
    title = preset["title"]
    sections = preset["sections"]
    
    corpus = []
    sec_idx = 0
    
    for doc_num in range(1, n_documents + 1):
        doc_id = f"DOC_{domain_key.upper()}_{doc_num:03d}"
        for page_num in range(1, pages_per_doc + 1):
            sec_name, sec_text = sections[sec_idx % len(sections)]
            sec_idx += 1
            corpus.append({
                "doc_id": doc_id,
                "title": f"{title} (Vol {doc_num})",
                "page": page_num,
                "section": sec_name,
                "text": sec_text,
                "domain": domain_key
            })
            
    return corpus


def generate_rag_qa_pairs(
    corpus: List[Dict[str, Any]],
    n_answerable: int = 4,
    n_unanswerable: int = 2,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Generate ground truth Q&A pairs with citations and unanswerable abstention queries.
    """
    qa_pairs = []
    
    # 1. Answerable queries based on corpus content
    for i in range(min(n_answerable, len(corpus))):
        doc = corpus[i]
        sec_title = doc["section"]
        words = doc["text"].split()[:12]
        query = f"What is stated in section '{sec_title}' regarding {' '.join(words[:4])}?"
        citation = f"[Doc: {doc['title']}, Page: {doc['page']}, Section: {doc['section']}]"
        qa_pairs.append({
            "query": query,
            "ground_truth_answer": doc["text"],
            "citation": citation,
            "is_abstention": False
        })
        
    # 2. Out-of-scope / Unanswerable queries (to test deterministic abstention)
    out_of_scope_topics = [
        "quantum teleportation protocol for interstellar spacecraft",
        "sub-atomic particle acceleration under zero-gravity conditions",
        "alchemical transmutation of lead into pure gold"
    ]
    for i in range(n_unanswerable):
        topic = out_of_scope_topics[i % len(out_of_scope_topics)]
        qa_pairs.append({
            "query": f"What are the mandatory guidelines for {topic}?",
            "ground_truth_answer": "I cannot answer this question based on the provided documentation.",
            "citation": None,
            "is_abstention": True
        })
        
    return qa_pairs


def save_rag_corpus_to_files(
    output_dir: str,
    corpus: List[Dict[str, Any]],
    format: str = "markdown"
) -> str:
    """Save corpus items as individual markdown or text files for directory loading."""
    os.makedirs(output_dir, exist_ok=True)
    for i, item in enumerate(corpus):
        filename = f"{item['doc_id']}_p{item['page']}.md"
        filepath = os.path.join(output_dir, filename)
        content = f"# {item['title']}\n\n**Page**: {item['page']} | **Section**: {item['section']}\n\n{item['text']}\n"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
    return os.path.abspath(output_dir)


# ============================================================================
# 6. AGENT TRACE & SCENARIO SYNTHETIC GENERATOR
# ============================================================================

AGENT_PROJECT_SPECS = {
    "AG-01": {"agents": ["SoilAnalystAgent", "WeatherForecastAgent", "AgronomySynthesizer"], "tools": ["SoilAnalyzerTool", "WeatherLookupTool"]},
    "AG-02": {"agents": ["TicketClassifierAgent", "PolicyRetrievalAgent", "ResponseDrafterAgent"], "tools": ["TicketClassifierTool", "PolicySearchTool"]},
    "AG-03": {"agents": ["FundamentalAnalystAgent", "NewsSentimentAgent", "ComplianceAuditor"], "tools": ["FinancialRatioTool", "NewsSentimentTool"]},
    "AG-04": {"agents": ["ResumeParserAgent", "JDMatcherAgent", "InterviewQuestionDrafter"], "tools": ["ResumeParserTool", "SkillGapTool"]},
    "AG-05": {"agents": ["DestinationResearchAgent", "RoutePlannerAgent", "BudgetAuditor"], "tools": ["TransitEstimatorTool", "AccommodationBudgetTool"]},
    "AG-06": {"agents": ["PaperDiscoveryAgent", "MethodologyExtractor", "SynthesisReviewer"], "tools": ["PaperMetadataTool", "MethodologyMatrixTool"]},
    "AG-07": {"agents": ["TelemetryIngestionAgent", "AnomalyDetector", "HVACOptimizationAgent"], "tools": ["LoadAnomalyTool", "ConservationSavingsTool"]},
    "AG-08": {"agents": ["SIEMLogAnalystAgent", "ThreatIntelMapper", "ResponsePlaybookDrafter"], "tools": ["LogParserTool", "MitreAttackLookupTool"]},
    "AG-09": {"agents": ["StockLevelMonitorAgent", "DemandForecaster", "PurchaseDrafter"], "tools": ["StockLevelCheckTool", "ReorderQuantityTool"]},
    "AG-10": {"agents": ["CurriculumArchitectAgent", "ContentWriterAgent", "PedagogicalReviewerAgent"], "tools": ["SyllabusAlignerTool", "QuizGeneratorTool"]}
}


def generate_agent_trace(
    project_id: str = "AG-01",
    scenario_type: str = "happy_path",  # "happy_path" | "tool_error" | "hitl_approved" | "hitl_rejected"
    n_turns: int = 3,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Generate deterministic agent execution trace compliant with results/agent_trace.json.
    """
    proj_spec = AGENT_PROJECT_SPECS.get(project_id, AGENT_PROJECT_SPECS["AG-01"])
    agents = proj_spec["agents"]
    tools = proj_spec["tools"]
    
    trace_id = f"TR_{project_id}_{seed:04d}"
    session_id = f"SESS_{seed:04d}"
    
    steps = []
    hitl_interventions = []
    
    for turn in range(1, n_turns + 1):
        agent_name = agents[(turn - 1) % len(agents)]
        tool_name = tools[(turn - 1) % len(tools)]
        
        is_hitl = (turn == n_turns and scenario_type.startswith("hitl"))
        is_error = (turn == 2 and scenario_type == "tool_error")
        
        step_dict = {
            "step_number": turn,
            "timestamp": f"2026-11-21T12:00:{turn:02d}Z",
            "agent_name": agent_name,
            "action": "call_tool",
            "tool_name": tool_name,
            "tool_input": {"param_key": f"test_val_{turn}", "seed": seed},
            "duration_ms": 120 + turn * 10
        }
        
        if is_error:
            step_dict["status"] = "ERROR"
            step_dict["error_message"] = "SimulatedToolTimeoutError: Connection timed out"
            step_dict["tool_output"] = None
        else:
            step_dict["status"] = "SUCCESS"
            step_dict["tool_output"] = {"status": "ok", "result": f"Execution result for step {turn}"}
            
        if is_hitl:
            step_dict["requires_human_approval"] = True
            decision = "APPROVED" if scenario_type == "hitl_approved" else "REJECTED"
            step_dict["approval_decision"] = decision
            hitl_interventions.append({
                "step_number": turn,
                "agent_name": agent_name,
                "action_type": "CONSEQUENTIAL_ACTION",
                "decision": decision,
                "approver": "supervisor@enterprise.org",
                "timestamp": f"2026-11-21T12:00:{turn:02d}.500Z"
            })
            
        steps.append(step_dict)
        
    final_status = "ABORTED" if scenario_type == "hitl_rejected" else "COMPLETED"
    final_output = (
        "Execution aborted by user rejection at safety gate."
        if scenario_type == "hitl_rejected"
        else f"Successfully completed multi-agent workflow for {project_id}."
    )
    
    return {
        "trace_id": trace_id,
        "project_id": project_id,
        "session_id": session_id,
        "scenario_type": scenario_type,
        "start_time": "2026-11-21T12:00:00Z",
        "end_time": f"2026-11-21T12:00:{n_turns:02d}Z",
        "status": final_status,
        "steps": steps,
        "hitl_interventions": hitl_interventions,
        "final_output": final_output
    }


def generate_agent_scenarios(
    project_id: str = "AG-01",
    n_scenarios: int = 10,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Generate a full suite of diverse scenarios for agent testing:
    happy paths, tool failures, approved gates, and rejected gates.
    """
    types = ["happy_path", "happy_path", "tool_error", "hitl_approved", "hitl_rejected"]
    scenarios = []
    for i in range(n_scenarios):
        sc_type = types[i % len(types)]
        sc = generate_agent_trace(
            project_id=project_id,
            scenario_type=sc_type,
            n_turns=3,
            seed=seed + i
        )
        scenarios.append(sc)
    return scenarios


def save_agent_trace_json(output_path: str, trace_data: Dict[str, Any]) -> str:
    """Save agent trace dictionary to JSON file."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(trace_data, f, indent=2)
    return os.path.abspath(output_path)
