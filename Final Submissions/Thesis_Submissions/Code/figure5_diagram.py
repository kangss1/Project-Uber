from matplotlib import pyplot as plt
from matplotlib_venn import venn3

fig, ax = plt.subplots(figsize=(8, 6))

venn = venn3(
    subsets=(1, 1, 1, 1, 1, 1, 1),
    set_labels=("Fairness-Aware Optimization", "Sustainability Objectives", "Regulatory Governance"),
    set_colors=("skyblue", "lightgreen", "navajowhite"),
    alpha=0.7
)

venn.get_label_by_id('100').set_text("Equitable allocation\nBias mitigation")
venn.get_label_by_id('010').set_text("Carbon reduction\nFleet efficiency")
venn.get_label_by_id('001').set_text("Transparency\nPolicy oversight")
venn.get_label_by_id('110').set_text("Inclusive green\nmobility")
venn.get_label_by_id('101').set_text("Accountable algorithms\nBias auditing")
venn.get_label_by_id('011').set_text("Environmental\npolicy integration")
venn.get_label_by_id('111').set_text("Responsible and\nEquitable Ecosystem")

for text in venn.set_labels:
    text.set_fontsize(12)
for text in venn.subset_labels:
    if text:
        text.set_fontsize(10)

plt.title("Conceptual Relationship among Fairness, Sustainability, and Governance",
          fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig("Figure5_Fairness_Sustainability_Governance.png", dpi=400, bbox_inches='tight')
plt.show()