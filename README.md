Software Defect Prediction
Experimental Environment
Python	Python 3.10/3.11
IDE	PyCharm Professional (or Community Edition)
NumPy	2.2.6
pandas	2.3.3
Scikit-learn	1.7.2
XGBoost	2.1.4
LightGBM	4.7.0
CatBoost	1.2.10
SHAP	0.49.1
LIME	0.2.0.1
BorutaPy	0.4.3

Experimental Procedures
1、When run the main.py, Set USE_BORUTA in config.py as False to get the no_FS results, the final_result.csv is in the File Folder named metrics
2、Set USE_BORUTA in config.py as True to get the FS results, after finished, we renamed the "results" as "results-FS"; we also rename the final_result.csv as final_result_FS.csv in the folder of metrics

3、The original inputs of csv file for each RQ are generated from step 1 and step 2;
In RQ1, RQ1_statistical_analysis.py is for Friedman, P-value, Wilcoxon Statistic cliffs_delta
In RQ2, overall_summary.csv and repository_summary.csv are from results-FS/feature_selection;
In RQ3, run collect_all_shap_csv.py first cuz we need to get Repository_SHAP_Importance.csv;
run the cross_repo_shap_scheme1.py to get spearman_rho.csv,spearman_pvalue.csv,fig_family_total.png and fig_spearman_heatmap.png;
run the cross_repo_shap_scheme1.py to get repo_topk_family_count.csv,fig_topk_stacked_bar.png


