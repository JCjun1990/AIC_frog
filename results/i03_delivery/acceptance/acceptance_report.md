# I03 独立解包验收：FAIL

- zip: `待填_初赛_AI-MECH-2026-002_result.zip` (11.1 MB, 72 files)
- 解包目录（仓库外临时目录，验收后删除）：`C:\Users\ZHOUSH~1\AppData\Local\Temp\i03_accept_iumeh7ad`

| # | 检查 | 结果 | 说明 |
|---|---|---|---|
| 1 | zip_crc | PASS | 72 entries |
| 2 | zip_root_layout | PASS | README and manifest at the archive root |
| 3 | zip_relative_posix_names | PASS | unsafe=[] |
| 4 | required_deliverables_present | PASS | missing=[] |
| 5 | manifest_present | PASS | 70 files |
| 6 | manifest_hashes_match | PASS | mismatched=[] |
| 7 | zip_files_all_listed | FAIL | unlisted=['README_提交.md'] |
| 8 | every_file_has_reason_and_source | PASS |  |
| 9 | no_absolute_local_paths | FAIL | offenders=['code/requirements.txt', 'task1_sphere/spots/stability_certificate.json', 'task1_sphere/spots/summary.json'] |
| 10 | no_secret_material | PASS | flagged=[] |
| 11 | no_task4_material | PASS |  |
| 12 | packaged_code_imports | PASS | turing_rd loaded from C:\Users\ZHOUSH~1\AppData\Local\Temp\i03_accept_iumeh7ad\package\code\turing_rd |
| 13 | four_mat_acceptance | PASS | [] |
| 14 | inversion_parameters_present | PASS | F=0.027712 k=0.060040 |
| 15 | inversion_mat_is_steady | PASS | residual=9.061e-14, pattern=spots/21 |
| 16 | inversion_report_has_errors | PASS |  |
| 17 | inversion_visuals_present | PASS |  |
| 18 | report_page_limit | PASS | 2 pages |
| 19 | report_numbers_match_package | PASS | missing=[] |
| 20 | zip_name_matches_manifest | PASS | 待填_初赛_AI-MECH-2026-002_result.zip |
| 21 | draft_status_recorded | PASS | draft: team name pending; regenerate before upload |

所有检查均在解包后的文件上执行；MAT 残差用包内 `code/turing_rd` 与包内 `frog.mat` 重新计算，未使用仓库中的结果缓存。
