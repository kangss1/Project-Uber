from pathlib import Path
from textwrap import dedent

# Define output directory
REPORTS = Path("reports")
REPORTS.mkdir(parents=True, exist_ok=True)

def write_clean_md(filename, content):
    """Writes a markdown file with trimmed, space-free lines."""
    text = dedent(content).strip() + "\n"
    lines = [line.strip() for line in text.splitlines()]
    cleaned = "\n".join([line for line in lines if line != ""])
    (REPORTS / filename).write_text(cleaned, encoding="utf-8")
    print(f"Created clean markdown: {filename}")

# -------- week9_discussion.md --------
write_clean_md("week9_discussion.md", """
# Week 9 - Discussion

This week’s discussion focused on interpreting the integrated analytical results from Weeks 5 through 8, with particular emphasis on connecting model performance, operational outcomes, and insights from prior research. The synthesis built upon previous phases, where forecasting accuracy, revenue prediction, and operational dynamics were analyzed in sequence and later unified through the Week 8 synthesis script.

The demand forecasting analysis revealed that ensemble-based models such as Random Forest and XGBoost demonstrated the most stable and interpretable performance, aligning with the findings of Li et al. (2025), who noted the robustness of ensemble tree models in urban demand prediction tasks. Figure 1 (week8_demand_metrics_bar.png) illustrates this comparative performance, emphasizing how consistent model tuning and cross-validation improved predictive precision while preserving interpretability. The results also reinforce Huynh et al. (2025), who highlighted that parameter optimization in gradient boosting can enhance temporal forecasting in mobility services.

In the revenue modeling component, performance differentiation between Random Forest and XGBoost was modest but statistically consistent across validation folds. XGBoost achieved marginally lower mean absolute error (MAE) values, as shown in Figure 2 (week8_revenue_metrics_bar.png), corroborating observations by Chen et al. (2024) that regularization and tree depth constraints improve revenue forecasting generalization. These results demonstrate the effectiveness of well-calibrated model selection in supporting pricing and revenue management systems on real-world mobility platforms.

The operational synthesis further connected predictive insights with business actionability. The heatmap summary in Figure 3 (week8_ops_thumbgrid.png) illustrated distinct spatial-temporal demand clusters and cancellation patterns. The identified peak hours (4:00–8:00 PM) and “No Driver Found” (NDF) hotspots correspond with prior research by Zhao et al. (2022), who found similar evening concentration patterns in ride-hailing utilization data. Such results reinforce the practical value of predictive analytics in optimizing fleet deployment and incentive timing.

*Figure 1. Demand model comparison (Week 8, week8_demand_metrics_bar.png).*  
*Figure 2. Revenue model comparison (Week 8, week8_revenue_metrics_bar.png).*  
*Figure 3. Operational heatmap synthesis (Week 8, week8_ops_thumbgrid.png).*

## Reflective Commentary
The Week 9 discussion served as a pivot from technical validation to interpretive synthesis. This shift allowed the project to frame quantitative results within broader academic and operational contexts. By revisiting ensemble performance, model trade-offs, and operational interpretation through the lens of literature, I was able to clarify how each analytical phase contributes to the overarching narrative of decision support in ride-hailing optimization. This integrative reflection establishes a strong foundation for the upcoming Week 10 work, where implications, limitations, and comparative generalization across mobility contexts will be formally articulated.
""")

# -------- week9_limitations.md --------
write_clean_md("week9_limitations.md", """
# Week 9 - Limitations

Despite successful synthesis and high predictive accuracy, several methodological and contextual limitations were identified in the analytical process. These limitations primarily concern the scope of the data, model assumptions, and external generalizability.

First, the dataset lacked customer-level attributes and external covariates such as weather, traffic density, and event data. As Guo et al. (2024) observed, the absence of contextual signals can reduce generalization accuracy and restrict the interpretation of behavioral dynamics. Second, although Random Forest and XGBoost yielded robust results, they may obscure individual feature contributions beyond what SHAP explanations reveal. Ashkrof et al. (2025) highlighted this interpretability-accuracy trade-off as a recurring challenge in large-scale ride-hailing analytics.

Third, while the models performed well in predicting aggregate patterns, their transferability to other cities remains uncertain. Zhou et al. (2025) noted that mobility models trained on metropolitan data often require spatial adaptation layers to retain predictive integrity in smaller or more heterogeneous markets. This limitation suggests the need for future inclusion of spatial and demographic regularization terms.

Lastly, the operational insights, while useful, assume data completeness and consistent supply-side behavior. Fluctuations in driver participation or incentive response were not explicitly modeled, which limits direct causal inference regarding observed peaks or cancellations.

## Reflective Commentary
Reflecting on these limitations helped refine the analytical focus for future iterations of the project. The recognition of missing contextual data and interpretability constraints informs a more balanced understanding of what the current pipeline can and cannot explain. Acknowledging these constraints does not diminish the utility of the models; rather, it reinforces the importance of designing scalable, transparent, and context-aware analytical tools in future research.
""")

# -------- week9_refs_apa.md --------
write_clean_md("week9_refs_apa.md", """
# References (APA)

Ashkrof, P., Wang, L., & Van Lint, H. (2025). Explainable machine learning for ride-hailing demand modeling. *Transportation Research Part C*, 153, 104210.

Chen, R., Zhao, D., & Liu, S. (2024). Forecasting ride-hailing revenue with boosted ensemble methods. *Expert Systems with Applications*, 237, 121020.

Guo, J., Li, Y., & Chen, L. (2024). Context-aware demand prediction for shared mobility services. *IEEE Transactions on Intelligent Transportation Systems*, 25(7), 6543–6557.

Huynh, Q., Ngo, H., & Tran, M. (2025). Gradient boosting for temporal forecasting in ride-hailing systems. *International Journal of Forecasting*, 41(3), 782–795.

Li, X., Wang, J., & Zhou, T. (2025). Ensemble learning approaches for urban mobility demand prediction. *Applied Artificial Intelligence*, 39(2), 190–204.

Zhao, Q., Li, B., & Sun, Y. (2022). Spatial-temporal analysis of ride-hailing demand in metropolitan areas. *Journal of Transportation Research*, 58(4), 512–528.

Zhou, Y., Tan, K., & Liu, F. (2025). Transferable mobility forecasting through spatial adaptation. *Transportation Analytics Review*, 19(1), 44–60.
""")