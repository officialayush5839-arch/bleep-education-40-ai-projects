"""
================================================================================
Bleep Education 40 Industry AI Projects - Master E2E Test Runner
================================================================================
Scans, discovers, and executes automated pytest test suites across all 40
projects and shared common utilities with isolated subprocess environments,
per-project duration metrics, live progress tracking, and exit code validation.

Usage:
    python run_all_tests.py                     # Run all discovered tests
    python run_all_tests.py --domain ml         # Run only Part A (ML) tests
    python run_all_tests.py --domain dl         # Run only Part B (DL) tests
    python run_all_tests.py --domain genai      # Run only Part C (RAG) tests
    python run_all_tests.py --domain agentic    # Run only Part D (Agentic) tests
    python run_all_tests.py --domain common     # Run common_utils unit tests
    python run_all_tests.py --project ml_01     # Run specific project test
    python run_all_tests.py --fast              # Run only fast unit tests
    python run_all_tests.py --verbose           # Verbose pytest output
    python run_all_tests.py --dry-run           # List discovery without running
    python run_all_tests.py --report summary.json # Save JSON execution report
================================================================================
"""

import argparse
import datetime
import json
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional


class SuiteStatus(Enum):
    READY = "READY"
    PASSED = "PASSED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    ERROR = "ERROR"
    NO_TESTS = "NO_TESTS"
    NOT_SCAFFOLDED = "NOT_SCAFFOLDED"


@dataclass
class ProjectMetadata:
    project_id: str
    folder_name: str
    domain_key: str
    domain_title: str
    title: str


@dataclass
class TestResult:
    project: ProjectMetadata
    status: SuiteStatus
    test_dir: Optional[Path] = None
    passed_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    error_count: int = 0
    duration_seconds: float = 0.0
    stdout: str = ""
    stderr: str = ""
    error_summary: str = ""


# ==============================================================================
# Canonical 40 Project Registry
# ==============================================================================
CANONICAL_PROJECTS: List[ProjectMetadata] = [
    # --- Common Infrastructure ---
    ProjectMetadata("COMMON", "common_utils", "common", "Infrastructure", "Common Deliverables Toolkit"),

    # --- Part A: Machine Learning (ML-01 to ML-10) ---
    ProjectMetadata("ML-01", "ml_01_predictive_maintenance", "part_a_machine_learning", "Part A: Machine Learning", "Predictive Maintenance & Failure"),
    ProjectMetadata("ML-02", "ml_02_patient_readmission", "part_a_machine_learning", "Part A: Machine Learning", "Patient Readmission Risk"),
    ProjectMetadata("ML-03", "ml_03_credit_card_fraud", "part_a_machine_learning", "Part A: Machine Learning", "Credit Card Fraud Detection"),
    ProjectMetadata("ML-04", "ml_04_telecom_churn", "part_a_machine_learning", "Part A: Machine Learning", "Telecom Customer Churn"),
    ProjectMetadata("ML-05", "ml_05_crop_recommendation", "part_a_machine_learning", "Part A: Machine Learning", "Crop Recommendation System"),
    ProjectMetadata("ML-06", "ml_06_property_price_prediction", "part_a_machine_learning", "Part A: Machine Learning", "Real Estate Property Price"),
    ProjectMetadata("ML-07", "ml_07_hotel_cancellation", "part_a_machine_learning", "Part A: Machine Learning", "Hotel Booking Cancellation"),
    ProjectMetadata("ML-08", "ml_08_solar_power_forecasting", "part_a_machine_learning", "Part A: Machine Learning", "Solar Power Generation Forecasting"),
    ProjectMetadata("ML-09", "ml_09_customer_segmentation", "part_a_machine_learning", "Part A: Machine Learning", "Retail Customer Segmentation"),
    ProjectMetadata("ML-10", "ml_10_employee_attrition", "part_a_machine_learning", "Part A: Machine Learning", "Employee Attrition Analytics"),

    # --- Part B: Deep Learning (DL-01 to DL-10) ---
    ProjectMetadata("DL-01", "dl_01_waste_classification", "part_b_deep_learning", "Part B: Deep Learning", "Automated Waste Classification"),
    ProjectMetadata("DL-02", "dl_02_plant_disease_classification", "part_b_deep_learning", "Part B: Deep Learning", "Plant Leaf Disease Classification"),
    ProjectMetadata("DL-03", "dl_03_chest_xray_pneumonia", "part_b_deep_learning", "Part B: Deep Learning", "Chest X-Ray Pneumonia Screening"),
    ProjectMetadata("DL-04", "dl_04_traffic_sign_recognition", "part_b_deep_learning", "Part B: Deep Learning", "Road Traffic Sign Recognition"),
    ProjectMetadata("DL-05", "dl_05_surface_defect_detection", "part_b_deep_learning", "Part B: Deep Learning", "Industrial Surface Defect Detection"),
    ProjectMetadata("DL-06", "dl_06_sentiment_analysis", "part_b_deep_learning", "Part B: Deep Learning", "Movie Review Sentiment Classification"),
    ProjectMetadata("DL-07", "dl_07_human_activity_recognition", "part_b_deep_learning", "Part B: Deep Learning", "Smartphone Human Activity Recognition"),
    ProjectMetadata("DL-08", "dl_08_electricity_forecasting", "part_b_deep_learning", "Part B: Deep Learning", "Household Electricity Forecasting"),
    ProjectMetadata("DL-09", "dl_09_fashion_mnist", "part_b_deep_learning", "Part B: Deep Learning", "Fashion Item Classification"),
    ProjectMetadata("DL-10", "dl_10_urban_sound_classification", "part_b_deep_learning", "Part B: Deep Learning", "Urban Sound Classification"),

    # --- Part C: Generative AI / RAG (GEN-01 to GEN-10) ---
    ProjectMetadata("GEN-01", "gen_01_hr_policy_assistant", "part_c_generative_ai_rag", "Part C: Generative AI / RAG", "Enterprise HR Policy Assistant"),
    ProjectMetadata("GEN-02", "gen_02_medical_research_assistant", "part_c_generative_ai_rag", "Part C: Generative AI / RAG", "Medical Research Literature Assistant"),
    ProjectMetadata("GEN-03", "gen_03_academic_info_assistant", "part_c_generative_ai_rag", "Part C: Generative AI / RAG", "University Academic Info Assistant"),
    ProjectMetadata("GEN-04", "gen_04_legal_research_assistant", "part_c_generative_ai_rag", "Part C: Generative AI / RAG", "Legal Document Research Assistant"),
    ProjectMetadata("GEN-05", "gen_05_financial_report_assistant", "part_c_generative_ai_rag", "Part C: Generative AI / RAG", "Financial Report Analysis Assistant"),
    ProjectMetadata("GEN-06", "gen_06_agri_extension_assistant", "part_c_generative_ai_rag", "Part C: Generative AI / RAG", "Agricultural Extension Assistant"),
    ProjectMetadata("GEN-07", "gen_07_cybersecurity_assistant", "part_c_generative_ai_rag", "Part C: Generative AI / RAG", "Cybersecurity Awareness Assistant"),
    ProjectMetadata("GEN-08", "gen_08_dev_doc_assistant", "part_c_generative_ai_rag", "Part C: Generative AI / RAG", "Software Dev Documentation Assistant"),
    ProjectMetadata("GEN-09", "gen_09_climate_research_assistant", "part_c_generative_ai_rag", "Part C: Generative AI / RAG", "Climate Science Research Assistant"),
    ProjectMetadata("GEN-10", "gen_10_tech_training_assistant", "part_c_generative_ai_rag", "Part C: Generative AI / RAG", "Technical Training & Revision Assistant"),

    # --- Part D: Agentic AI (AG-01 to AG-10) ---
    ProjectMetadata("AG-01", "ag_01_agri_advisory_multiagent", "part_d_agentic_ai", "Part D: Agentic AI", "Agricultural Advisory Multi-Agent"),
    ProjectMetadata("AG-02", "ag_02_support_ticket_triage", "part_d_agentic_ai", "Part D: Agentic AI", "Customer Support Ticket Triage"),
    ProjectMetadata("AG-03", "ag_03_financial_market_analyst", "part_d_agentic_ai", "Part D: Agentic AI", "Financial Market Research Multi-Agent"),
    ProjectMetadata("AG-04", "ag_04_recruitment_screening", "part_d_agentic_ai", "Part D: Agentic AI", "Recruitment Admin & Screening"),
    ProjectMetadata("AG-05", "ag_05_travel_itinerary_planner", "part_d_agentic_ai", "Part D: Agentic AI", "Travel Itinerary & Budget Multi-Agent"),
    ProjectMetadata("AG-06", "ag_06_research_paper_matrix", "part_d_agentic_ai", "Part D: Agentic AI", "Research Paper Analysis Multi-Agent"),
    ProjectMetadata("AG-07", "ag_07_facility_energy_optimization", "part_d_agentic_ai", "Part D: Agentic AI", "Facility Energy Optimization Agent"),
    ProjectMetadata("AG-08", "ag_08_security_incident_triage", "part_d_agentic_ai", "Part D: Agentic AI", "Security Incident Response Multi-Agent"),
    ProjectMetadata("AG-09", "ag_09_inventory_reordering_assistant", "part_d_agentic_ai", "Part D: Agentic AI", "Inventory Reordering Multi-Agent"),
    ProjectMetadata("AG-10", "ag_10_educational_content_multiagent", "part_d_agentic_ai", "Part D: Agentic AI", "Educational Content Development Agent"),
]


# ==============================================================================
# ANSI Color Helpers
# ==============================================================================
USE_COLOR = sys.stdout.isatty() or platform.system() == "Windows"

def colored(text: str, color_code: str) -> str:
    if not USE_COLOR:
        return text
    return f"\033[{color_code}m{text}\033[0m"

def green(text: str) -> str: return colored(text, "92")
def red(text: str) -> str: return colored(text, "91")
def yellow(text: str) -> str: return colored(text, "93")
def blue(text: str) -> str: return colored(text, "94")
def bold(text: str) -> str: return colored(text, "1")
def dim(text: str) -> str: return colored(text, "2")


# ==============================================================================
# Discovery Logic
# ==============================================================================
def discover_projects(
    workspace_root: Path,
    domain_filter: Optional[str] = None,
    project_filter: Optional[str] = None
) -> List[TestResult]:
    """
    Scans workspace for all 40 canonical projects + common utilities.
    Matches filesystem reality with canonical metadata.
    """
    results: List[TestResult] = []

    domain_alias_map = {
        "ml": "part_a_machine_learning",
        "part_a": "part_a_machine_learning",
        "part_a_machine_learning": "part_a_machine_learning",
        "dl": "part_b_deep_learning",
        "part_b": "part_b_deep_learning",
        "part_b_deep_learning": "part_b_deep_learning",
        "genai": "part_c_generative_ai_rag",
        "rag": "part_c_generative_ai_rag",
        "part_c": "part_c_generative_ai_rag",
        "part_c_generative_ai_rag": "part_c_generative_ai_rag",
        "agentic": "part_d_agentic_ai",
        "ag": "part_d_agentic_ai",
        "part_d": "part_d_agentic_ai",
        "part_d_agentic_ai": "part_d_agentic_ai",
        "common": "common",
        "common_utils": "common",
        "infra": "common",
    }

    normalized_domain = domain_alias_map.get(domain_filter.lower(), domain_filter.lower()) if domain_filter else None

    for meta in CANONICAL_PROJECTS:
        # Check domain filter
        if normalized_domain and meta.domain_key.lower() != normalized_domain:
            continue

        # Check project filter
        if project_filter:
            needle = project_filter.lower()
            if needle not in meta.project_id.lower() and needle not in meta.folder_name.lower():
                continue

        # Determine target directory
        if meta.domain_key == "common":
            project_dir = workspace_root / meta.folder_name
        else:
            project_dir = workspace_root / meta.domain_key / meta.folder_name

        test_dir = project_dir / "tests"

        # Check directory existence
        if not project_dir.exists():
            status = SuiteStatus.NOT_SCAFFOLDED
            resolved_test_dir = None
        elif not test_dir.exists():
            status = SuiteStatus.NO_TESTS
            resolved_test_dir = None
        else:
            # Check for test files
            test_files = list(test_dir.glob("test_*.py")) + list(test_dir.glob("*_test.py"))
            if not test_files:
                status = SuiteStatus.NO_TESTS
                resolved_test_dir = None
            else:
                status = SuiteStatus.READY
                resolved_test_dir = test_dir

        results.append(TestResult(
            project=meta,
            status=status,
            test_dir=resolved_test_dir
        ))

    return results


# ==============================================================================
# Test Execution Logic
# ==============================================================================
def execute_test_suite(
    result: TestResult,
    workspace_root: Path,
    fast_mode: bool = False,
    verbose: bool = False,
    timeout_sec: int = 180
) -> TestResult:
    """
    Executes pytest for a single project test suite inside an isolated subprocess.
    """
    if result.status != SuiteStatus.READY or not result.test_dir:
        return result

    project_dir = result.test_dir.parent

    # Build pytest CLI arguments
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(result.test_dir.name)
    ]

    if verbose:
        cmd.extend(["-v", "--tb=short"])
    else:
        cmd.extend(["-q", "--tb=line"])

    if fast_mode:
        cmd.extend(["-m", "not slow"])

    # Prepare isolated environment with PYTHONPATH forwarding
    env = os.environ.copy()
    py_path_parts = [str(project_dir), str(workspace_root)]
    existing_py_path = env.get("PYTHONPATH", "")
    if existing_py_path:
        py_path_parts.append(existing_py_path)
    env["PYTHONPATH"] = os.pathsep.join(py_path_parts)
    env["PYTHONUNBUFFERED"] = "1"
    env["ENVIRONMENT"] = "test"

    start_time = time.perf_counter()

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(project_dir),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec
        )
        duration = time.perf_counter() - start_time
        result.duration_seconds = round(duration, 2)
        result.stdout = proc.stdout or ""
        result.stderr = proc.stderr or ""

        # Parse test metrics from stdout
        full_output = result.stdout + "\n" + result.stderr

        m_pass = re.search(r"(\d+)\s+passed", full_output)
        m_fail = re.search(r"(\d+)\s+failed", full_output)
        m_skip = re.search(r"(\d+)\s+skipped", full_output)
        m_err = re.search(r"(\d+)\s+error", full_output)

        result.passed_count = int(m_pass.group(1)) if m_pass else 0
        result.failed_count = int(m_fail.group(1)) if m_fail else 0
        result.skipped_count = int(m_skip.group(1)) if m_skip else 0
        result.error_count = int(m_err.group(1)) if m_err else 0

        # Pytest return codes: 0 = all pass, 1 = tests failed, 5 = no tests collected
        if proc.returncode == 0:
            result.status = SuiteStatus.PASSED
        elif proc.returncode == 5:
            result.status = SuiteStatus.NO_TESTS
        else:
            result.status = SuiteStatus.FAILED
            # Extract first 20 lines of failure summary
            lines = [l for l in full_output.splitlines() if l.strip()]
            result.error_summary = "\n".join(lines[-25:])

    except subprocess.TimeoutExpired:
        result.duration_seconds = float(timeout_sec)
        result.status = SuiteStatus.TIMEOUT
        result.error_summary = f"Pytest execution timed out after {timeout_sec} seconds."
    except Exception as exc:
        result.duration_seconds = round(time.perf_counter() - start_time, 2)
        result.status = SuiteStatus.ERROR
        result.error_summary = str(exc)

    return result


# ==============================================================================
# Console Reporting & Formatting
# ==============================================================================
def print_banner(workspace_root: Path):
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(bold("=" * 80))
    print(bold(" Bleep Education 40 Industry AI Projects - Master E2E Test Runner"))
    print(bold("=" * 80))
    print(f" Workspace Root : {workspace_root}")
    print(f" Python Version : {platform.python_version()} ({sys.executable})")
    print(f" Platform       : {platform.system()} {platform.release()} ({platform.machine()})")
    print(f" Timestamp      : {now_str}")
    print(bold("-" * 80))


def print_summary_table(results: List[TestResult]):
    print("\n" + bold("=" * 80))
    print(bold(" E2E EXECUTION SUMMARY TABLE"))
    print(bold("=" * 80))
    header = f"{'Domain':<22} | {'ID':<7} | {'Project Directory':<30} | {'Status':<10} | {'Pass/Fail':<10} | {'Time':<6}"
    print(header)
    print("-" * len(header))

    for r in results:
        status_str = r.status.value
        if r.status == SuiteStatus.PASSED:
            colored_status = green(f"{status_str:<10}")
        elif r.status in (SuiteStatus.FAILED, SuiteStatus.ERROR, SuiteStatus.TIMEOUT):
            colored_status = red(f"{status_str:<10}")
        elif r.status == SuiteStatus.NO_TESTS:
            colored_status = yellow(f"{status_str:<10}")
        else:
            colored_status = dim(f"{status_str:<10}")

        pass_fail = f"{r.passed_count}P / {r.failed_count}F" if r.status in (SuiteStatus.PASSED, SuiteStatus.FAILED) else "-"
        time_str = f"{r.duration_seconds:.1f}s" if r.duration_seconds > 0 else "-"

        # Short domain name
        domain_display = r.project.domain_key.replace("part_", "").replace("_", " ").title()[:22]

        print(f"{domain_display:<22} | {r.project.project_id:<7} | {r.project.folder_name:<30} | {colored_status} | {pass_fail:<10} | {time_str:<6}")

    print(bold("=" * 80))

    # Aggregates
    total_suites = len(results)
    ready_suites = sum(1 for r in results if r.status in (SuiteStatus.PASSED, SuiteStatus.FAILED, SuiteStatus.TIMEOUT, SuiteStatus.ERROR))
    passed_suites = sum(1 for r in results if r.status == SuiteStatus.PASSED)
    failed_suites = sum(1 for r in results if r.status in (SuiteStatus.FAILED, SuiteStatus.ERROR, SuiteStatus.TIMEOUT))
    unscaffolded = sum(1 for r in results if r.status == SuiteStatus.NOT_SCAFFOLDED)
    no_tests = sum(1 for r in results if r.status == SuiteStatus.NO_TESTS)

    total_passed_tests = sum(r.passed_count for r in results)
    total_failed_tests = sum(r.failed_count for r in results)
    total_duration = sum(r.duration_seconds for r in results)

    print(f" Total Canonical Suites : {total_suites}")
    print(f" Executed Suites        : {ready_suites}")
    print(f" Passed Suites          : {green(str(passed_suites))}")
    print(f" Failed Suites          : {red(str(failed_suites)) if failed_suites > 0 else '0'}")
    print(f" Unscaffolded / Pending : {dim(str(unscaffolded))}")
    print(f" Empty / No Tests       : {yellow(str(no_tests))}")
    print(f" Total Assertions/Tests : {total_passed_tests} Passed, {total_failed_tests} Failed")
    print(f" Total Run Duration     : {total_duration:.2f} seconds")
    print(bold("=" * 80))


def save_json_report(results: List[TestResult], report_path: Path):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "summary": {
            "total_canonical": len(results),
            "executed": sum(1 for r in results if r.status in (SuiteStatus.PASSED, SuiteStatus.FAILED, SuiteStatus.TIMEOUT, SuiteStatus.ERROR)),
            "passed": sum(1 for r in results if r.status == SuiteStatus.PASSED),
            "failed": sum(1 for r in results if r.status in (SuiteStatus.FAILED, SuiteStatus.ERROR, SuiteStatus.TIMEOUT)),
            "unscaffolded": sum(1 for r in results if r.status == SuiteStatus.NOT_SCAFFOLDED),
            "no_tests": sum(1 for r in results if r.status == SuiteStatus.NO_TESTS),
            "total_passed_tests": sum(r.passed_count for r in results),
            "total_failed_tests": sum(r.failed_count for r in results),
            "total_duration_sec": sum(r.duration_seconds for r in results)
        },
        "results": [
            {
                "project_id": r.project.project_id,
                "folder_name": r.project.folder_name,
                "domain": r.project.domain_key,
                "title": r.project.title,
                "status": r.status.value,
                "passed_count": r.passed_count,
                "failed_count": r.failed_count,
                "skipped_count": r.skipped_count,
                "duration_seconds": r.duration_seconds,
                "error_summary": r.error_summary if r.status != SuiteStatus.PASSED else ""
            }
            for r in results
        ]
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f" [i] Detailed JSON report written to: {report_path}")


# ==============================================================================
# Main Orchestration Routine
# ==============================================================================
def main() -> int:
    parser = argparse.ArgumentParser(description="Master E2E Test Runner for Bleep Education 40 AI Projects")
    parser.add_argument("--domain", "-d", type=str, default=None, help="Filter by domain (ml, dl, genai, agentic, common)")
    parser.add_argument("--project", "-p", type=str, default=None, help="Filter by project ID or name (e.g. ml_01, waste)")
    parser.add_argument("--fast", "-f", action="store_true", help="Fast mode: skip slow integration/training tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose pytest output")
    parser.add_argument("--fail-fast", "-x", action="store_true", help="Stop execution immediately on first suite failure")
    parser.add_argument("--dry-run", action="store_true", help="Discover test suites without executing")
    parser.add_argument("--report", "-r", type=str, default=None, help="Path to write JSON summary report")
    parser.add_argument("--timeout", type=int, default=180, help="Per-suite timeout in seconds (default: 180)")

    args = parser.parse_args()

    workspace_root = Path(__file__).resolve().parent

    print_banner(workspace_root)

    # 1. Discover project test suites
    discovered = discover_projects(
        workspace_root=workspace_root,
        domain_filter=args.domain,
        project_filter=args.project
    )

    if not discovered:
        print(yellow(" [!] No project suites matched the specified filters."))
        return 0

    ready_to_run = [r for r in discovered if r.status == SuiteStatus.READY]
    unscaffolded = [r for r in discovered if r.status == SuiteStatus.NOT_SCAFFOLDED]
    no_tests = [r for r in discovered if r.status == SuiteStatus.NO_TESTS]

    print(f" Discovered: {len(discovered)} total suites | {len(ready_to_run)} ready to execute | {len(unscaffolded)} pending scaffold | {len(no_tests)} empty/no tests\n")

    if args.dry_run:
        print(blue(" [*] DRY RUN requested. Skipping test execution."))
        print_summary_table(discovered)
        return 0

    # 2. Execute ready test suites
    has_failure = False

    for idx, result in enumerate(discovered, 1):
        if result.status != SuiteStatus.READY:
            continue

        suite_header = f"[{idx}/{len(discovered)}] {result.project.project_id} ({result.project.folder_name})"
        print(f" Running {bold(suite_header)} ...", end="", flush=True)

        res = execute_test_suite(
            result=result,
            workspace_root=workspace_root,
            fast_mode=args.fast,
            verbose=args.verbose,
            timeout_sec=args.timeout
        )

        if res.status == SuiteStatus.PASSED:
            print(f" {green('PASSED')} ({res.passed_count} passed, {res.duration_seconds:.2f}s)")
        else:
            print(f" {red(res.status.value)} ({res.failed_count} failed, {res.error_count} err, {res.duration_seconds:.2f}s)")
            if res.error_summary:
                print(dim("    --- Failure Diagnostic Snippet ---"))
                for line in res.error_summary.splitlines():
                    print(dim(f"    | {line}"))
                print(dim("    ----------------------------------"))
            has_failure = True
            if args.fail_fast:
                print(red("\n [!] --fail-fast triggered. Aborting remaining tests."))
                break

    # 3. Print Final Summary Table
    print_summary_table(discovered)

    # 4. Save JSON Report if requested
    if args.report:
        save_json_report(discovered, Path(args.report))

    # 5. Return Exit Code
    # In intermediate milestones, if no suites are ready to execute yet, exit 0 cleanly.
    # If test suites ran and any failed, exit 1. Otherwise exit 0.
    if has_failure:
        print(red("\n [X] TEST EXECUTION FAILED: One or more test suites did not pass."))
        return 1
    else:
        print(green("\n [V] TEST EXECUTION SUCCESSFUL: All executed test suites passed 100%."))
        return 0


if __name__ == "__main__":
    sys.exit(main())
