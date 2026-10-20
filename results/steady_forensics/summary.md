# 离散格式取证（稳态假设）

每种离散在原始数据上拟合两条稳态方程；若正好是组委会的离散且为收敛稳态，R² 应趋近 1。「题给参数残差」= 用 Du = 2.78e-4、Dv = 1.39e-4、F = 0.028、k = 0.062 时的相对残差。

| 离散族 | 变体 | R²(u) | R²(v) | Du | F | Dv | k | Dv/Du | 题给参数残差 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Q1 bilinear (2x2 Gauss) | consistent, nodal nonlinear | 0.7531 | 0.9266 | 6.4e-05 | 0.0345 | 9.95e-06 | 0.0602 | 0.156 | 0.737 |
| Q1 bilinear (3x3 Gauss) | consistent, nodal nonlinear | 0.7531 | 0.9266 | 6.4e-05 | 0.0345 | 9.95e-06 | 0.0602 | 0.156 | 0.737 |
| Q1 bilinear (2x2 Gauss) | consistent, exact nonlinear | 0.7505 | 0.9264 | 6.49e-05 | 0.0345 | 1.08e-05 | 0.0604 | 0.166 | 0.732 |
| Q1 bilinear (3x3 Gauss) | consistent, exact nonlinear | 0.7505 | 0.9264 | 6.49e-05 | 0.0345 | 1.08e-05 | 0.0604 | 0.166 | 0.733 |
| P1 cotan | short diag / lumped, exact nonlinear | 0.7448 | 0.9266 | 5.71e-05 | 0.0347 | 5.43e-06 | 0.0601 | 0.095 | 0.793 |
| P1 cotan | short diag / consistent, nodal nonlinear | 0.7453 | 0.9258 | 5.79e-05 | 0.0347 | 9.03e-06 | 0.0602 | 0.156 | 0.789 |
| P1 cotan | short diag / consistent, exact nonlinear | 0.7429 | 0.9257 | 5.86e-05 | 0.0347 | 9.72e-06 | 0.0603 | 0.166 | 0.785 |
| Q1 bilinear (2x2 Gauss) | lumped | 0.7490 | 0.9196 | 6.82e-05 | 0.0346 | 1.09e-05 | 0.0603 | 0.159 | 0.721 |
| Q1 bilinear (3x3 Gauss) | lumped | 0.7490 | 0.9196 | 6.82e-05 | 0.0346 | 1.09e-05 | 0.0603 | 0.159 | 0.721 |
| P1 cotan | diag 1-3 / lumped, exact nonlinear | 0.7387 | 0.9273 | 5.09e-05 | 0.0349 | 4.13e-06 | 0.0600 | 0.081 | 0.852 |
| P1 cotan | diag 1-3 / consistent, nodal nonlinear | 0.7390 | 0.9262 | 5.18e-05 | 0.0349 | 8.02e-06 | 0.0601 | 0.155 | 0.847 |
| P1 robust | intrinsic Delaunay / lumped | 0.7430 | 0.9195 | 6.28e-05 | 0.0347 | 9.92e-06 | 0.0602 | 0.158 | 0.762 |
| P1 cotan | diag 1-3 / consistent, exact nonlinear | 0.7362 | 0.9260 | 5.26e-05 | 0.0349 | 8.73e-06 | 0.0603 | 0.166 | 0.843 |
| P1 cotan | short diag / Voronoi | 0.7414 | 0.9195 | 6.15e-05 | 0.0348 | 9.72e-06 | 0.0602 | 0.158 | 0.773 |
| P1 cotan | short diag / lumped | 0.7413 | 0.9194 | 6.13e-05 | 0.0348 | 9.68e-06 | 0.0602 | 0.158 | 0.774 |
| P1 cotan | diag 2-4 / lumped, exact nonlinear | 0.7333 | 0.9269 | 4.71e-05 | 0.0350 | 3.82e-06 | 0.0600 | 0.081 | 0.893 |
| P1 cotan | diag 2-4 / consistent, nodal nonlinear | 0.7335 | 0.9258 | 4.79e-05 | 0.0350 | 7.17e-06 | 0.0601 | 0.150 | 0.889 |
| graph | unweighted, vertex area | 0.7389 | 0.9200 | 4.57e-05 | 0.0349 | 8.07e-06 | 0.0600 | 0.177 | 1.035 |
| P1 cotan | diag 2-4 / consistent, exact nonlinear | 0.7308 | 0.9257 | 4.86e-05 | 0.0350 | 7.78e-06 | 0.0603 | 0.160 | 0.885 |
| P1 cotan | diag 1-3 / Voronoi | 0.7353 | 0.9194 | 5.58e-05 | 0.0349 | 8.97e-06 | 0.0601 | 0.161 | 0.827 |
| P1 cotan | diag 1-3 / lumped | 0.7350 | 0.9193 | 5.57e-05 | 0.0349 | 8.95e-06 | 0.0601 | 0.161 | 0.828 |
| P1 cotan | diag 2-4 / lumped | 0.7291 | 0.9191 | 5.13e-05 | 0.0351 | 7.88e-06 | 0.0602 | 0.154 | 0.871 |
| P1 cotan | diag 2-4 / Voronoi | 0.7291 | 0.9190 | 5.14e-05 | 0.0351 | 7.88e-06 | 0.0602 | 0.153 | 0.870 |
| graph | 1/|e| weights, vertex area | 0.7065 | 0.9187 | 5.13e-07 | 0.0356 | 8.96e-08 | 0.0599 | 0.175 | 77.316 |
| graph | unweighted, unit mass | 0.6905 | 0.9155 | 5.09e-05 | 0.0352 | 7.91e-06 | 0.0612 | 0.155 | 0.415 |
| graph | 1/|e|^2 weights, unit mass | 0.6555 | 0.9144 | 2.58e-05 | 0.0360 | 4.12e-06 | 0.0610 | 0.160 | 1.312 |
