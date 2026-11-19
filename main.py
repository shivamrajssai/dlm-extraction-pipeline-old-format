#!/usr/bin/env python3
"""
DLM Extraction Pipeline - Main Execution Entrypoint.
Modular Python application ported from Databricks DLM Extraction Pipeline notebook.
Execution target: AWS EC2 / Linux environment.

Usage:
    python3 main.py --zone East --phase 1 --cycle 1 --upc 20240900000000001
"""

import os
import sys
import argparse
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import PipelineSettings
from config.table_schema import TableSchemaManager
from src.common.logger import setup_logger, get_logger, log_stage_event
from src.common.run_context import init_run_context, get_current_context, stage_scope
from src.common.issue_reporter import IssueReporter
from src.common.batch_monitor import BatchMonitor
from src.common.exceptions import DLMPipelineError, ConfigurationError, DataIngestionError
from src.common.constants import (
    CHILD_10M,
    CHILD_100M,
    PARENT_LENGTH_M,
    TABLE_SPEC,
    CA_CRACK_PCT,
    CA_BI_THRESHOLD,
    CA_RUT_THRESHOLD,
    RAVELLING_RATING_THRESHOLD,
    IRC_TV,
    SRC_10M,
    SRC_100,
    SRC_1KM,
)

# Ingestion
from src.ingestion.tracker_loader import load_tracker_context
from src.ingestion.d2_loader import parse_d2_thresholds, parse_irc_specific_thresholds
from src.ingestion.backup_loader import load_backup_lane_parquets
from src.ingestion.d1_loader import load_d1_context
from src.ingestion.d3_loader import load_d3_conditions
from src.ingestion.d4_loader import load_d4_potholes
from src.ingestion.bronze_loader import load_all_raw_defects

# Processing
from src.processing.pothole import process_potholes
from src.processing.cracking import (
    calibrate_cracking_with_d2,
    build_cracking_pdf,
    build_cracking_agg_pdf,
    apply_cracking_severity_and_condition,
)
from src.processing.roughness import build_roughness_linear_set
from src.processing.rutting import build_rutting_linear_set
from src.processing.ravelling import build_ravelling_linear_set
from src.processing.pci import load_patching_data, calculate_pci, backfill_pci_to_outputs
from src.processing.road_condition import apply_road_condition
from src.processing.ratings import standardize_and_rate

# Hierarchy
from src.hierarchy.lhs_rhs_builder import build_all_silver_hierarchy
from src.hierarchy.chainage_builder import build_master_chainage

# Output
from src.output.table_writer import TableWriter
from src.output.image_uploader import ImageUploader
from src.output.excel_exporter import export_to_excel
from src.output.csv_exporter import export_to_csv

# Validation
from src.validation.validator import Validator
from src.validation.validation_report import generate_validation_summary, save_validation_report

logger = get_logger("dlm_pipeline")


def run_pipeline(
    settings: PipelineSettings,
    batch_id: Optional[str] = None,
    issue_reporter: Optional[IssueReporter] = None,
) -> Dict[str, Any]:
    """
    Execute the DLM extraction pipeline for a single UPC with given settings.
    """
    run_ctx = init_run_context(
        batch_id=batch_id,
        upc=settings.upc,
        zone=settings.zone,
        phase=str(settings.phase),
        cycle=int(settings.cycle),
    )
    reporter = issue_reporter or IssueReporter(
        tracebacks_dir=os.path.join(settings.validation_report_dir, "tracebacks")
    )

    logger.info("=" * 80)
    logger.info("STARTING DLM EXTRACTION PIPELINE [Run ID: %s]", run_ctx.run_id)
    logger.info("UPC: %s | Zone: %s | Phase: %s | Cycle: %s", settings.upc, settings.zone, settings.phase, settings.cycle)
    logger.info("=" * 80)

    t0 = time.time()
    try:
        # 1. Tracker Context
        logger.info("[Step 1/10] Loading Tracker Context...")
        with stage_scope("INGEST_TRACKER", "Loading Tracker Context") as stg:
            tracker_ctx = load_tracker_context(
                tracker_path=settings.tracker_excel_path,
                d2_base=settings.d2_dir,
                zone=settings.zone,
                phase=str(settings.phase),
                cycle=str(settings.cycle),
                upc_slash=settings.upc.replace('_', '/'),
            )
            survey_date = tracker_ctx.get('survey_start_date') or datetime.now().date()
            if isinstance(survey_date, datetime):
                run_date = survey_date.date()
            else:
                run_date = survey_date
            display_date_str = run_date.strftime('%d%m%y')
            ca_intervals = {
                d: (tracker_ctx.get('defect_standards', {}).get(d, {}).get('interval_m') or 100)
                for d in ['ravelling', 'rutting', 'cracking', 'roughness', 'pothole']
            }
            logger.info("Display defect date prefix: %s | Survey intervals (from NSV Tracker): %s", display_date_str, ca_intervals)

        # 2. D2 Thresholds & Standards
        logger.info("[Step 2/10] Parsing D2 Thresholds...")
        with stage_scope("INGEST_D2", "Parsing D2 Thresholds") as stg:
            d2_thresholds = parse_d2_thresholds(settings.d2_dir)
            irc_thresholds = parse_irc_specific_thresholds(settings.d2_dir, d2_thresholds)

            ca_crack_pct = float(d2_thresholds.get('cracking_pct', CA_CRACK_PCT))
            ca_bi_threshold = float(d2_thresholds.get('roughness_bi', CA_BI_THRESHOLD))
            ca_rut_threshold = float(d2_thresholds.get('rutting_mm', CA_RUT_THRESHOLD))
            ca_ravel_threshold = float(d2_thresholds.get('ravelling', RAVELLING_RATING_THRESHOLD))

            ca_threshold_map = {
                'cracking': ca_crack_pct,
                'roughness': ca_bi_threshold,
                'rutting': ca_rut_threshold,
                'ravelling': ca_ravel_threshold,
            }

            irc_crack_pct = float(irc_thresholds.get('cracking_pct', IRC_TV.get('cracking', 30.0)))
            irc_iri_threshold = float(irc_thresholds.get('roughness_iri', IRC_TV.get('roughness', 2400.0)))
            irc_rut_threshold = float(irc_thresholds.get('rutting_mm', IRC_TV.get('rutting', 10.0)))
            irc_ravel_threshold = float(irc_thresholds.get('ravelling', IRC_TV.get('ravelling', 10.0)))

        # 3. D1 Context
        logger.info("[Step 3/10] Loading D1 Context...")
        with stage_scope("INGEST_D1", "Loading D1 Road Context") as stg:
            d1_ctx = load_d1_context(settings.d1_dir, upc=settings.upc)

        # 4. Backup & Raw Defect Data
        logger.info("[Step 4/10] Ingesting NSV Backup & Raw Defects...")
        with stage_scope("INGEST_RAW", "Ingesting NSV Backup & Raw Defects") as stg:
            raw_defects = load_all_raw_defects(
                backup_base_dir=settings.backup_base_dir,
                d4_dir=settings.d4_dir,
                upc=settings.upc,
                zone=settings.zone,
                phase=str(settings.phase),
                cycle=str(settings.cycle),
            )

        # 5. D3 Conditions & D4 Potholes
        logger.info("[Step 5/10] Ingesting D3 Conditions and D4 Potholes...")
        with stage_scope("INGEST_D3_D4", "Ingesting D3 Conditions and D4 Potholes") as stg:
            d3_conditions = load_d3_conditions(settings.d3_dir)
            d4_potholes = raw_defects.get('pothole', pd.DataFrame())

        # 6. Processing Defect Tables
        logger.info("[Step 6/10] Processing Defect Tables (Pothole, Cracking, Roughness, Rutting, Ravelling)...")
        with stage_scope("PROCESS_DEFECTS", "Processing Defect Tables") as stg:
            output_dfs = {}

            # Potholes
            pothole_df = process_potholes(
                df_d4_raw=d4_potholes,
                d2_dir=settings.d2_dir,
                d1_ctx=d1_ctx,
                d3_conditions=d3_conditions,
                upc=settings.upc,
                zone=settings.zone,
                phase=str(settings.phase),
                cycle=str(settings.cycle),
                run_date=run_date,
            )
            output_dfs['master_pothole'] = pothole_df

            # Cracking
            raw_crack = raw_defects.get('cracking', pd.DataFrame())
            calib_crack_df = calibrate_cracking_with_d2(raw_crack, settings.d2_dir)
            crack_10m = build_cracking_pdf(
                calib_crack_df,
                upc=settings.upc,
                zone=settings.zone,
                phase=str(settings.phase),
                cycle=str(settings.cycle),
                d1_ctx=d1_ctx,
                tracker_ctx=tracker_ctx,
                run_date=run_date,
                ca_crack_pct=ca_crack_pct,
                irc_crack_pct=irc_crack_pct,
            )
            crack_100m = build_cracking_agg_pdf(
                crack_10m,
                bin_m=ca_intervals['cracking'],
                level='child_100m',
                ca_crack_pct=ca_crack_pct,
                upc=settings.upc,
            )
            crack_1km = build_cracking_agg_pdf(
                crack_10m,
                bin_m=PARENT_LENGTH_M,
                level='parent',
                ca_crack_pct=ca_crack_pct,
                upc=settings.upc,
            )

            output_dfs['master_cracking'] = crack_10m
            output_dfs['master_cracking_child_100m'] = crack_100m
            output_dfs['master_cracking_parent'] = crack_1km

            # Roughness
            rough_sets = build_roughness_linear_set(
                raw_defects.get('roughness', pd.DataFrame()),
                upc=settings.upc,
                zone=settings.zone,
                phase=str(settings.phase),
                cycle=str(settings.cycle),
                d1_ctx=d1_ctx,
                tracker_ctx=tracker_ctx,
                d3_conditions=d3_conditions,
                run_date=run_date,
                ca_bi_threshold=ca_bi_threshold,
                irc_iri_threshold=irc_iri_threshold,
                ca_interval_m=ca_intervals['roughness'],
            )
            output_dfs.update(rough_sets)

            # Rutting
            rut_sets = build_rutting_linear_set(
                raw_defects.get('rutting', pd.DataFrame()),
                upc=settings.upc,
                zone=settings.zone,
                phase=str(settings.phase),
                cycle=str(settings.cycle),
                d1_ctx=d1_ctx,
                tracker_ctx=tracker_ctx,
                d3_conditions=d3_conditions,
                run_date=run_date,
                ca_rut_threshold=ca_rut_threshold,
                irc_rut_threshold=irc_rut_threshold,
                ca_interval_m=ca_intervals['rutting'],
            )
            output_dfs.update(rut_sets)

            # Ravelling
            ravel_sets = build_ravelling_linear_set(
                raw_defects.get('ravelling', pd.DataFrame()),
                upc=settings.upc,
                zone=settings.zone,
                phase=str(settings.phase),
                cycle=str(settings.cycle),
                d1_ctx=d1_ctx,
                tracker_ctx=tracker_ctx,
                d3_conditions=d3_conditions,
                run_date=run_date,
                ca_ravelling_threshold=ca_ravel_threshold,
                irc_ravelling_threshold=irc_ravel_threshold,
                ca_interval_m=ca_intervals['ravelling'],
            )
            output_dfs.update(ravel_sets)

        # 7. PCI & Road Condition Ratings
        logger.info("[Step 7/10] Computing PCI, Road Conditions, and Ratings...")
        with stage_scope("PROCESS_PCI_RATINGS", "Computing PCI, Road Conditions, and Ratings") as stg:
            df_patch = load_patching_data(settings.backup_base_dir)
            pci_results = calculate_pci(
                output_dfs=output_dfs,
                df_patch=df_patch,
                tracker_ctx=tracker_ctx,
            )
            output_dfs = backfill_pci_to_outputs(output_dfs, pci_results)

            for tbl in list(output_dfs.keys()):
                output_dfs[tbl] = apply_road_condition(
                    df=output_dfs[tbl],
                    table_name=tbl,
                    d3_conditions=d3_conditions,
                    d2_dir=settings.d2_dir,
                )
                output_dfs[tbl] = standardize_and_rate(
                    df=output_dfs[tbl],
                    table_name=tbl,
                    ca_thresholds=ca_threshold_map,
                    irc_thresholds=IRC_TV,
                )

        # 8. LHS/RHS Silver Hierarchy & Master Chainage
        logger.info("[Step 8/10] Building Silver Hierarchy & Master Chainage...")
        with stage_scope("HIERARCHY_SILVER", "Building Silver Hierarchy & Master Chainage") as stg:
            silver_outputs = build_all_silver_hierarchy(
                output_dfs=output_dfs,
                upc=settings.upc,
                date_str=display_date_str,
                ca_intervals=ca_intervals,
                ca_thresholds=ca_threshold_map,
                irc_thresholds=IRC_TV,
            )

            upc_slash = settings.upc.replace('_', '/')
            master_chainage_df = build_master_chainage(
                output_dfs=output_dfs,
                upc=settings.upc,
                upc_slash=upc_slash,
                zone=settings.zone,
                phase=str(settings.phase),
                cycle=int(settings.cycle),
            )
            silver_outputs['master_chainage'] = master_chainage_df

            # Build geometry tables
            from src.hierarchy.geometry_builder import build_geometry_tables
            geom_p, geom_c = build_geometry_tables(
                master_chainage_df=master_chainage_df,
                upc=settings.upc,
                project_id=tracker_ctx.get('project_id'),
                nh_number=tracker_ctx.get('nh_number'),
                section_code=tracker_ctx.get('section_code'),
                survey_start_date=tracker_ctx.get('survey_start_date', run_date),
                survey_end_date=tracker_ctx.get('survey_end_date', run_date),
            )
            silver_outputs['geometry_parent'] = geom_p
            silver_outputs['geometry_child'] = geom_c

            # Enforce exact target schemas on all output tables
            from config.table_schema import enforce_table_schema
            for tbl in list(silver_outputs.keys()):
                silver_outputs[tbl] = enforce_table_schema(silver_outputs[tbl], tbl)

        # 9. Defect Images & CDN Upload
        logger.info("[Step 9/10] Processing Defect Image Uploads...")
        with stage_scope("UPLOAD_IMAGES", "Processing Defect Image Uploads") as stg:
            uploader = ImageUploader(
                deliverable_dir=settings.deliverable_dir,
                upload_url=settings.image_upload_base_url,
            )
            for t_img, kind in [('master_pothole', 'pothole'), ('master_cracking_10m', 'cracking')]:
                if t_img in silver_outputs and not silver_outputs[t_img].empty:
                    silver_outputs[t_img], img_summary = uploader.process_table_images(
                        table_name=t_img,
                        df=silver_outputs[t_img],
                        defect_type=kind,
                    )
                    logger.info("Image upload summary for %s: %s", t_img, img_summary)

        # Write tables to Parquet
        writer = TableWriter(output_dir=settings.silver_output_dir)
        write_counts = writer.write_all(silver_outputs, upc=settings.upc, phase=str(settings.phase), cycle=int(settings.cycle))
        total_rows_written = sum(write_counts.values())
        logger.info("Parquet persistence complete: %d tables, %d rows written", len(write_counts), total_rows_written)

        # 10. Validation & Exports
        logger.info("[Step 10/10] Executing Validation Checks and Generating Exports...")
        with stage_scope("VALIDATE_EXPORT", "Executing Validation Checks and Generating Exports") as stg:
            validator = Validator(
                new_outputs=silver_outputs,
                source_dfs=output_dfs,
                upc=settings.upc,
                zone=settings.zone,
                phase=str(settings.phase),
                cycle=int(settings.cycle),
                ca_intervals=ca_intervals,
                ca_thresholds=ca_threshold_map,
            )
            val_results = validator.validate_all()
            summary_text = generate_validation_summary(val_results)
            logger.info(summary_text)

            # Record validation failures in IssueReporter
            if not val_results.empty and 'status' in val_results.columns:
                failed_checks = val_results[val_results['status'] == 'FAIL']
                for _, r in failed_checks.iterrows():
                    reporter.record_validation_discrepancy(
                        check_name=str(r.get('check', '')),
                        scope=str(r.get('scope', '')),
                        failed_count=int(r.get('failed', 0)),
                        checked_count=int(r.get('checked', 0)),
                        detail=str(r.get('detail', '')),
                        section=str(r.get('section', 'Validation')),
                        ctx=run_ctx,
                    )

            # Save validation report CSV
            val_out_path = os.path.join(
                settings.validation_report_dir,
                f"validation_{settings.upc}_P{settings.phase}_C{settings.cycle}.csv"
            )
            save_validation_report(val_results, val_out_path)

            # Save issue report CSV
            issues_out_path = os.path.join(
                settings.validation_report_dir,
                f"issues_{settings.upc}_P{settings.phase}_C{settings.cycle}.csv"
            )
            reporter.save_issue_report(issues_out_path)

            # Excel export
            excel_path = os.path.join(
                settings.validation_report_dir,
                f"DLM_KPI_NEW_{settings.upc}_P{settings.phase}_C{settings.cycle}_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
            )
            export_to_excel(silver_outputs, excel_path, validation_df=val_results)

            # CSV export
            csv_dir = os.path.join(
                settings.exports_dir,
                f"{settings.upc}_P{settings.phase}_C{settings.cycle}"
            )
            export_to_csv(silver_outputs, csv_dir)

        duration_total_ms = (time.time() - t0) * 1000.0
        log_stage_event(
            logger,
            logging.INFO,
            f"PIPELINE COMPLETED SUCCESSFULLY FOR UPC: {settings.upc}",
            execution_status="SUCCESS",
            duration_ms=duration_total_ms,
            records_written=total_rows_written,
        )

        logger.info("=" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY FOR UPC: %s (duration: %.1fs)", settings.upc, duration_total_ms / 1000.0)
        logger.info("=" * 80)

        return {
            'status': 'SUCCESS',
            'run_id': run_ctx.run_id,
            'upc': settings.upc,
            'duration_ms': duration_total_ms,
            'tables': list(silver_outputs.keys()),
            'tables_written': len(write_counts),
            'records_written': total_rows_written,
            'excel_export': excel_path,
            'csv_export_dir': csv_dir,
            'issues_report': issues_out_path,
            'issues_count': len(reporter.issues),
            'validation_summary': val_results['status'].value_counts().to_dict() if not val_results.empty else {},
        }

    except Exception as e:
        duration_err_ms = (time.time() - t0) * 1000.0
        reporter.record_technical_exception(e, ctx=run_ctx)
        # Attempt to save issue report even on failure
        try:
            issues_err_path = os.path.join(
                settings.validation_report_dir,
                f"issues_{settings.upc}_P{settings.phase}_C{settings.cycle}.csv"
            )
            reporter.save_issue_report(issues_err_path)
        except Exception:
            pass

        log_stage_event(
            logger,
            logging.ERROR,
            f"Pipeline failed for UPC {settings.upc}: {e}",
            execution_status="FAILED",
            duration_ms=duration_err_ms,
            error={"exception_type": type(e).__name__, "message": str(e)},
        )
        raise e


def main():
    parser = argparse.ArgumentParser(description="DLM Extraction Pipeline - EC2 Modular Application")
    parser.add_argument("--zone", type=str, help="Zone identifier (e.g. East, North)")
    parser.add_argument("--phase", type=str, help="Phase identifier (e.g. 1)")
    parser.add_argument("--cycle", type=int, help="Cycle number (e.g. 1)")
    parser.add_argument("--upc", type=str, help="UPC code (e.g. 20240900000000001)")
    parser.add_argument("--batch-file", type=str, help="Path to text file containing list of UPCs (one per line)")
    parser.add_argument("--data-root", type=str, help="Root directory containing input and output datasets")
    parser.add_argument("--env-file", type=str, default=".env", help="Path to .env configuration file")

    args = parser.parse_args()

    # Load settings from CLI + .env
    base_settings = PipelineSettings.from_cli(args, env_file=args.env_file)
    setup_logger(
        level=base_settings.log_level,
        log_file=base_settings.log_file,
        json_log_file=base_settings.json_log_file,
    )

    if args.batch_file and os.path.exists(args.batch_file):
        with open(args.batch_file, 'r') as f:
            upcs = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        batch_monitor = BatchMonitor(batch_file=args.batch_file)
        batch_monitor.set_total_upcs(len(upcs))

        logger.info("Starting batch execution [Batch ID: %s] for %d UPCs from %s", batch_monitor.batch_id, len(upcs), args.batch_file)

        for entry in upcs:
            parts = entry.split()
            if len(parts) >= 4:
                if parts[0].startswith('N_') or '_' in parts[0]:
                    upc_val, z_val, p_val, c_val = parts[0], parts[1], parts[2], parts[3]
                else:
                    z_val, p_val, c_val, upc_val = parts[0], parts[1], parts[2], parts[3]
            else:
                upc_val, z_val, p_val, c_val = entry, base_settings.zone, base_settings.phase, base_settings.cycle

            batch_monitor.start_upc(upc_val, z_val, str(p_val), int(c_val))
            upc_t0 = time.time()

            try:
                upc_settings = PipelineSettings.from_cli(
                    argparse.Namespace(
                        zone=z_val,
                        phase=p_val,
                        cycle=c_val,
                        upc=upc_val,
                        data_root=base_settings.data_root,
                    ),
                    env_file=args.env_file,
                )
                res = run_pipeline(upc_settings, batch_id=batch_monitor.batch_id)
                upc_dur = (time.time() - upc_t0) * 1000.0
                batch_monitor.complete_upc(
                    upc=upc_val,
                    duration_ms=upc_dur,
                    tables_written=res.get('tables_written', 0),
                    records_written=res.get('records_written', 0),
                    validation_summary=res.get('validation_summary', {}),
                    excel_export=res.get('excel_export'),
                    csv_export_dir=res.get('csv_export_dir'),
                )
            except Exception as e:
                upc_dur = (time.time() - upc_t0) * 1000.0
                batch_monitor.fail_upc(upc_val, upc_dur, str(e))
                logger.error("Failed processing survey %s: %s", entry, e, exc_info=True)

        # Save batch summary to BATCH_LOG_S3 (s3://dhms-new-stgae/DHMS/NSV/logs/) or local validation directory
        batch_dest = base_settings.batch_log_s3 if (base_settings.batch_log_s3 and not base_settings.data_root) else base_settings.validation_report_dir
        batch_summary_csv = f"{batch_dest.rstrip('/')}/batch_summary_{batch_monitor.batch_id}.csv"
        batch_summary_json = f"{batch_dest.rstrip('/')}/run_summary_{batch_monitor.batch_id}.json"
        batch_monitor.save_batch_summary(batch_summary_csv, json_path=batch_summary_json)

        # Upload JSONL execution telemetry log to BATCH_LOG_S3 if available
        if base_settings.batch_log_s3 and not base_settings.data_root and base_settings.json_log_file and os.path.exists(base_settings.json_log_file):
            try:
                from src.common.fs_utils import write_file_bytes
                with open(base_settings.json_log_file, "rb") as jf:
                    s3_jsonl_path = f"{base_settings.batch_log_s3.rstrip('/')}/execution_{batch_monitor.batch_id}.jsonl"
                    write_file_bytes(s3_jsonl_path, jf.read())
                logger.info("Uploaded telemetry log to %s", s3_jsonl_path)
            except Exception as e:
                logger.warning("Could not upload telemetry log to S3: %s", e)

        summary = batch_monitor.finish_batch()
        logger.info(
            "Batch execution finished. Completed: %d | Failed: %d | Skipped: %d | Total: %d",
            summary["upcs_completed"],
            summary["upcs_failed"],
            summary["upcs_skipped"],
            summary["total_upcs_received"],
        )
    else:
        if not base_settings.upc:
            logger.error("UPC parameter is required. Specify via --upc or in .env")
            sys.exit(1)
        run_pipeline(base_settings)


if __name__ == "__main__":
    main()
