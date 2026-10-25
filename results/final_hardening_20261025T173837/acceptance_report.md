# I03 独立解包验收：PASS

- zip: `待填_初赛_AI-MECH-2026-002_result.zip` (31.1 MB, 138 files)
- 解包目录（仓库外临时目录，验收后删除）：`D:\Code\AIC_frog_audit_backup\stage8_20261025T173837\i03_accept_dtmtf59p`

| # | 检查 | 结果 | 说明 |
|---|---|---|---|
| 1 | zip_crc | PASS | 138 entries |
| 2 | zip_root_layout | PASS | README and manifest at the archive root |
| 3 | zip_relative_posix_names | PASS | unsafe=[] |
| 4 | required_deliverables_present | PASS | missing=[] |
| 5 | manifest_present | PASS | 137 files |
| 6 | manifest_hashes_match | PASS | mismatched=[] |
| 7 | zip_files_all_listed | PASS | unlisted=[] |
| 8 | every_file_has_reason_and_source | PASS |  |
| 9 | no_absolute_local_paths | PASS | offenders=[] |
| 10 | no_secret_material | PASS | flagged=[] |
| 11 | no_task4_material | PASS |  |
| 12 | packaged_code_imports | PASS | turing_rd loaded from D:\Code\AIC_frog_audit_backup\stage8_20261025T173837\i03_accept_dtmtf59p\package\code\turing_rd |
| 13 | strict_manifest_validation | PASS | {'manifest_files': 137, 'payload_files': 137} |
| 14 | json_strict_and_legacy_compatibility | PASS | {"protocol": "delivery-json-acceptance-v1", "strict_json_files": ["PACKAGE_MANIFEST.json", "evidence/active_learning/EFFICIENCY_BREAKDOWN.json", "evidence/activ |
| 15 | packaged_formal_model_inference | PASS | {"task1_sphere": "spots", "task2_frog": "spots"} |
| 16 | four_mat_acceptance | PASS | [] |
| 17 | inversion_parameters_present | PASS | F=0.027712 k=0.060040 |
| 18 | inversion_mat_is_steady | PASS | residual=9.061e-14, pattern=spots/21 |
| 19 | inversion_best_match_is_steady | PASS | {"available": true, "pde_residual_recomputed": true, "full_residual": 2.030535561466331e-14, "mat_sha256": "4e06d56e575eda2c5c2671d8e249d52460a5a50c9f6771ee30fa |
| 20 | inversion_best_match_saved_vector_spectrum | PASS | {"available": true, "pde_residual_recomputed": true, "full_residual": 2.030535561466331e-14, "mat_sha256": "4e06d56e575eda2c5c2671d8e249d52460a5a50c9f6771ee30fa |
| 21 | inversion_report_has_errors | PASS |  |
| 22 | inversion_visuals_present | PASS |  |
| 23 | report_page_limit | PASS | 2 pages |
| 24 | report_numbers_match_package | PASS | missing=[] |
| 25 | zip_name_matches_manifest | PASS | 待填_初赛_AI-MECH-2026-002_result.zip |
| 26 | delivery_status_recorded | PASS | draft: team name pending; do not upload |

所有检查均在解包后的文件上执行；MAT 残差用包内 `code/turing_rd` 与包内 `frog.mat` 重新计算，未使用仓库中的结果缓存。
