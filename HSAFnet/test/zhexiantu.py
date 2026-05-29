import matplotlib.pyplot as plt
import pandas as pd

# 准备数据
data = {
    'Threshold': ['0.2-0.65', '0.3-0.75', '0.4-0.85', '0.5-0.95'],
    'linear': [48.52, 48.81, 48.22, 47.92],
    'knn': [36.98, 37.03, 36.95, 35.96]
}
df = pd.DataFrame(data)

# 创建画布，一行两列 (figsize 调整为横向比例)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 设置全局字体 (如需 Times New Roman，请确保本地已安装)
plt.rcParams['font.family'] = 'serif'

# --- 子图 (a) Linear ---
ax1.plot(df['Threshold'], df['linear'], marker='o', color='blue', label='Linear', linewidth=2)
max_idx_lin = df['linear'].idxmax()
for i, val in enumerate(df['linear']):
    ax1.text(i, val + 0.1, f'{val:.2f}', ha='center', va='bottom', fontsize=12)
ax1.axvline(x=max_idx_lin, color='red', linestyle=':', alpha=0.6)
ax1.set_xlabel('Threshold', fontsize=12)
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.set_ylim(47.5, 49.5)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(loc='upper right')
# 设置图注在图的下方
ax1.text(0.5, -0.2, '(a) Linear Classification', transform=ax1.transAxes, 
         ha='center', fontsize=14)

# --- 子图 (b) KNN ---
ax2.plot(df['Threshold'], df['knn'], marker='s', color='green', label='K-NN', linewidth=2)
max_idx_knn = df['knn'].idxmax()
for i, val in enumerate(df['knn']):
    ax2.text(i, val + 0.05, f'{val:.2f}', ha='center', va='bottom', fontsize=12)
ax2.axvline(x=max_idx_knn, color='red', linestyle=':', alpha=0.6)
ax2.set_xlabel('Threshold', fontsize=12)
ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.set_ylim(35.5, 37.5)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(loc='upper right')
# 设置图注在图的下方
ax2.text(0.5, -0.2, '(b) K-NN Evaluation', transform=ax2.transAxes, 
         ha='center', fontsize=14)

plt.tight_layout(pad=4.0)
plt.savefig('combined_plot_horizontal.pdf', bbox_inches='tight')