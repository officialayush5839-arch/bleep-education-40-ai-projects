"""
================================================================================
Bleep Education 40 Industry AI Projects - Common Deliverables Toolkit
================================================================================
Universal shared utilities across all 4 domains (ML, DL, GenAI, Agentic AI):
1. Deterministic synthetic data generation (base_synthetic.py)
2. Production-grade ReportLab 12-section PDF report generation (report_generator.py)
================================================================================
"""

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

from common_utils.report_generator import (
    generate_pdf_report,
    STANDARD_SECTIONS,
    NumberedCanvas,
)

__all__ = [
    "get_rng",
    "set_seed",
    "generate_tabular_data",
    "generate_image_dataset",
    "save_synthetic_image_folder",
    "generate_time_series_data",
    "generate_sensor_windows",
    "create_lagged_features",
    "generate_rag_corpus",
    "generate_rag_qa_pairs",
    "save_rag_corpus_to_files",
    "generate_agent_trace",
    "generate_agent_scenarios",
    "save_agent_trace_json",
    "DOMAIN_PRESETS",
    "AGENT_PROJECT_SPECS",
    "generate_pdf_report",
    "STANDARD_SECTIONS",
    "NumberedCanvas",
]
