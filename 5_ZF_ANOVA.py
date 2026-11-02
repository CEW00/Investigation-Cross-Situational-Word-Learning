"""
Zipfian analysis 
Runs statistics and generates plots
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import f_oneway, linregress
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Set up paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
RESULTS_DIR = os.path.join(BASE_DIR, "results")
ANALYSIS_DIR = os.path.join(BASE_DIR, "analysis", "ZF")
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# Load the data
df_uni = pd.read_csv(os.path.join(RESULTS_DIR, 'UNI_hp_exp_two', 'scene_caption.csv'))
df_zf = pd.read_csv(os.path.join(RESULTS_DIR, 'ZF_hp_exp_two', 'scene_caption.csv'))

# Clean up
df_uni = df_uni[df_uni['matched_accuracy'].notna()]
df_zf = df_zf[df_zf['matched_accuracy'].notna()]

# Add distribution labels
df_uni['dist'] = 'uniform'
df_zf['dist'] = 'zipfian'

# Combine
df_zipfian = pd.concat([df_uni, df_zf], ignore_index=True)

# Figure out model types
def get_model_type(row):
    if row['loss_fn'] == 'triplet_loss' and not row['normalized']:
        return 'VL-Baseline'
    elif row['loss_fn'] == 'triplet_loss' and row['normalized']:
        return 'VL-Norm'
    elif row['loss_fn'] == 'SGCL' and row['normalized']:
        return 'SGCL-Norm'
    else:
        return 'Other'

df_zipfian['model_type'] = df_zipfian.apply(get_model_type, axis=1)

# Keep only the three main models
df_zipfian = df_zipfian[df_zipfian['model_type'].isin(['VL-Baseline', 'VL-Norm', 'SGCL-Norm'])]

print("=== Data loaded ===")
print(f"Total runs: {len(df_zipfian)}")
print(df_zipfian.groupby(['model_type', 'dist']).size())

# Main 2x3 ANOVA (Distribution × Model)
print("\n=== 2×3 ANOVA (Matched Frequency) ===")
anova_model = ols('matched_accuracy ~ C(model_type) * C(dist)', data=df_zipfian).fit()
anova_table = sm.stats.anova_lm(anova_model, typ=2)
print(anova_table)

# Effect sizes
print("\n=== Effect sizes ===")
def partial_eta_squared(anova_table):
    eta_sq = {}
    for effect in anova_table.index:
        ss_effect = anova_table.loc[effect, 'sum_sq']
        ss_error = anova_table.loc['Residual', 'sum_sq']
        eta_sq[effect] = ss_effect / (ss_effect + ss_error)
    return eta_sq

eta_squared = partial_eta_squared(anova_table)
for effect, eta in eta_squared.items():
    print(f"{effect}: ηp² = {eta:.3f}")

# Simple effects: Distribution effect for each model
print("\n=== Simple effects (t-tests) ===")
simple_effects_results = []

for model in df_zipfian['model_type'].unique():
    model_data = df_zipfian[df_zipfian['model_type'] == model]
    
    uniform_acc = model_data[model_data['dist'] == 'uniform']['matched_accuracy']
    zipfian_acc = model_data[model_data['dist'] == 'zipfian']['matched_accuracy']
    
    uniform_mean = uniform_acc.mean()
    zipfian_mean = zipfian_acc.mean()
    benefit = zipfian_mean - uniform_mean
    
    t_stat, p_value = stats.ttest_ind(uniform_acc, zipfian_acc)
    
    n1, n2 = len(uniform_acc), len(zipfian_acc)
    pooled_sd = np.sqrt(((n1-1)*uniform_acc.std()**2 + (n2-1)*zipfian_acc.std()**2) / (n1 + n2 - 2))
    cohens_d = benefit / pooled_sd
    
    simple_effects_results.append({
        'Model': model,
        'Uniform Mean': uniform_mean,
        'Zipfian Mean': zipfian_mean,
        'Benefit': benefit,
        't': t_stat,
        'p': p_value,
        "Cohen's d": cohens_d
    })
    
    print(f"{model}: t = {t_stat:.3f}, p = {p_value:.4f}, d = {cohens_d:.3f}")
    print(f"  Uniform: {uniform_mean:.3f}, Zipfian: {zipfian_mean:.3f}")

simple_effects_df = pd.DataFrame(simple_effects_results)

# ========================================
# Frequency gradient analysis (Zipfian only)
# ========================================
print("\n" + "="*60)
print("Frequency gradient analysis (Zipfian only)")
print("="*60)

df_zipfian_only = df_zipfian[df_zipfian['dist'] == 'zipfian'].copy()

# Reshape to long format
frequency_data = []
for _, row in df_zipfian_only.iterrows():
    for freq_type, acc in [('Low', row['low_accuracy']), 
                          ('Matched', row['matched_accuracy']), 
                          ('High', row['high_accuracy'])]:
        if pd.notna(acc):
            frequency_data.append({
                'model_type': row['model_type'],
                'frequency': freq_type,
                'accuracy': acc
            })

df_frequency_long = pd.DataFrame(frequency_data)
df_frequency_long['frequency'] = pd.Categorical(
    df_frequency_long['frequency'], 
    categories=['Low', 'Matched', 'High'], 
    ordered=True
)

# 3×3 ANOVA (Model × Frequency)
print("\n=== 3×3 ANOVA (Frequency Gradient) ===")
anova_freq_model = ols('accuracy ~ C(model_type) * C(frequency)', data=df_frequency_long).fit()
anova_freq_table = sm.stats.anova_lm(anova_freq_model, typ=2)
print(anova_freq_table)

# Check assumptions
residuals = anova_freq_model.resid
shapiro_stat, shapiro_p = stats.shapiro(residuals)
print(f"Normality (Shapiro-Wilk): W = {shapiro_stat:.3f}, p = {shapiro_p:.4f}")

# Simple effects for each model
print("\n=== Frequency effect for each model ===")
for model in df_frequency_long['model_type'].unique():
    model_data = df_frequency_long[df_frequency_long['model_type'] == model]
    model_aov = ols('accuracy ~ C(frequency)', data=model_data).fit()
    model_aov_table = sm.stats.anova_lm(model_aov, typ=2)
    
    f_value = model_aov_table.loc['C(frequency)', 'F']
    p_value = model_aov_table.loc['C(frequency)', 'PR(>F)']
    
    print(f"{model}: F = {f_value:.3f}, p = {p_value:.4f}")
    
    # Means
    means = model_data.groupby('frequency')['accuracy'].mean()
    print(f"  Low: {means['Low']:.3f}, Matched: {means['Matched']:.3f}, High: {means['High']:.3f}")

# Tukey HSD for each model
print("\n=== Tukey HSD post-hocs ===")
for model in df_frequency_long['model_type'].unique():
    model_data = df_frequency_long[df_frequency_long['model_type'] == model]
    print(f"\n{model}:")
    tukey = pairwise_tukeyhsd(
        endog=model_data['accuracy'],
        groups=model_data['frequency'],
        alpha=0.05
    )
    print(tukey.summary())

# ========================================
# Make the figures
# ========================================
print("\n=== Making figures ===")

# Prep data for plotting
plot_data = []
for _, row in df_zipfian.iterrows():
    for freq_type in ['high_accuracy', 'matched_accuracy', 'low_accuracy']:
        if pd.notna(row[freq_type]):
            plot_data.append({
                'model_type': row['model_type'],
                'dist': row['dist'],
                'frequency': freq_type,
                'accuracy': row[freq_type]
            })

df_plot = pd.DataFrame(plot_data)

# Clean up labels
frequency_map = {
    'high_accuracy': 'High',
    'matched_accuracy': 'Matched', 
    'low_accuracy': 'Low'
}
df_plot['frequency'] = df_plot['frequency'].map(frequency_map)
df_plot['frequency'] = pd.Categorical(df_plot['frequency'], categories=['Low', 'Matched', 'High'])

# Summary stats
plot_summary = df_plot.groupby(['model_type', 'dist', 'frequency']).agg({
    'accuracy': ['mean', 'std', 'count']
}).round(3)
plot_summary.columns = ['mean', 'std', 'count']
plot_summary = plot_summary.reset_index()
plot_summary['se'] = plot_summary['std'] / np.sqrt(plot_summary['count'])

# Add human data from Hendrickson & Perfors (2019)
human_data = pd.DataFrame({
    'model_type': ['Human'] * 6,
    'dist': ['uniform', 'zipfian', 'uniform', 'zipfian', 'uniform', 'zipfian'],
    'frequency': ['Low', 'Low', 'Matched', 'Matched', 'High', 'High'],
    'mean': [np.nan, 0.26, 0.343, 0.53, np.nan, 0.77],
    'se': [np.nan, 0.03, 0.03, 0.03, np.nan, 0.04]
})

# Combine
all_plot_data = pd.concat([plot_summary, human_data], ignore_index=True)
dist_map = {'uniform': 'Uniform', 'zipfian': 'Zipfian'}
all_plot_data['dist'] = all_plot_data['dist'].map(dist_map)

# Order for plotting
model_order = ['Human', 'VL-Baseline', 'VL-Norm', 'SGCL-Norm']
all_plot_data['model_type'] = pd.Categorical(all_plot_data['model_type'], categories=model_order)

# Figure 1: 4-panel bar plot
fig, axes = plt.subplots(1, 4, figsize=(20, 6), sharey=True)

for i, model in enumerate(model_order):
    ax = axes[i]
    model_data = all_plot_data[all_plot_data['model_type'] == model]
    
    x_pos = np.arange(3)  # Three frequency levels
    width = 0.35
    
    # Split by distribution
    uniform_data = model_data[model_data['dist'] == 'Uniform']
    zipfian_data = model_data[model_data['dist'] == 'Zipfian']
    
    # Get means in correct order
    uniform_means = []
    zipfian_means = []
    uniform_se = []
    zipfian_se = []
    
    for freq in ['Low', 'Matched', 'High']:
        u = uniform_data[uniform_data['frequency'] == freq]
        z = zipfian_data[zipfian_data['frequency'] == freq]
        
        uniform_means.append(u['mean'].values[0] if len(u) > 0 else np.nan)
        zipfian_means.append(z['mean'].values[0] if len(z) > 0 else np.nan)
        uniform_se.append(u['se'].values[0] if len(u) > 0 else np.nan)
        zipfian_se.append(z['se'].values[0] if len(z) > 0 else np.nan)
    
    # Plot
    ax.bar(x_pos - width/2, uniform_means, width, label='Uniform', color='#F8766D', alpha=0.8)
    ax.bar(x_pos + width/2, zipfian_means, width, label='Zipfian', color='#00BA38', alpha=0.8)
    
    # Error bars
    for j in range(3):
        if not np.isnan(uniform_means[j]) and not np.isnan(uniform_se[j]):
            ax.errorbar(x_pos[j] - width/2, uniform_means[j], yerr=uniform_se[j], 
                       color='black', capsize=3, linewidth=1)
        if not np.isnan(zipfian_means[j]) and not np.isnan(zipfian_se[j]):
            ax.errorbar(x_pos[j] + width/2, zipfian_means[j], yerr=zipfian_se[j], 
                       color='black', capsize=3, linewidth=1)
    
    # Chance line
    ax.axhline(y=1/28, color='black', linestyle=':', alpha=0.7)
    
    # Title and labels
    ax.set_title(model, fontsize=16)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Low', 'Matched', 'High'], fontsize=12)
    
    if i == 0:
        ax.set_ylabel('Proportion Correct', fontsize=14)
        ax.legend(loc='upper left', fontsize=10)
    
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.text(0.5, 0.04, 'Frequency', ha='center', fontsize=16)
fig.suptitle('Zipfian Distribution Effect by Model Type', fontsize=18, y=0.95)
plt.tight_layout(rect=[0, 0.05, 1, 0.93])
plt.savefig(os.path.join(ANALYSIS_DIR, 'zipfian_effect_comparison.png'), dpi=300, bbox_inches='tight')
plt.show()

# Figure 2: Matched frequency comparison with significance
plt.figure(figsize=(12, 6))

model_types = ['VL-Baseline', 'VL-Norm', 'SGCL-Norm']
x_pos = np.arange(len(model_types))
width = 0.35

# Get matched frequency data
matched_summary = df_zipfian.groupby(['model_type', 'dist'])['matched_accuracy'].agg(['mean', 'std', 'count'])

uniform_means = [matched_summary.loc[(model, 'uniform'), 'mean'] for model in model_types]
zipfian_means = [matched_summary.loc[(model, 'zipfian'), 'mean'] for model in model_types]
uniform_se = [matched_summary.loc[(model, 'uniform'), 'std'] / np.sqrt(matched_summary.loc[(model, 'uniform'), 'count']) 
              for model in model_types]
zipfian_se = [matched_summary.loc[(model, 'zipfian'), 'std'] / np.sqrt(matched_summary.loc[(model, 'zipfian'), 'count']) 
              for model in model_types]

# Bars
plt.bar(x_pos - width/2, uniform_means, width, label='Uniform', 
        color='#F8766D', alpha=0.8, yerr=uniform_se, capsize=5)
plt.bar(x_pos + width/2, zipfian_means, width, label='Zipfian', 
        color='#00BA38', alpha=0.8, yerr=zipfian_se, capsize=5)

# Significance markers
for i, model in enumerate(model_types):
    model_effect = next((item for item in simple_effects_results if item['Model'] == model), None)
    if model_effect and model_effect['p'] < 0.05:
        y_max = max(uniform_means[i], zipfian_means[i]) + max(uniform_se[i], zipfian_se[i]) + 0.02
        plt.plot([x_pos[i] - width/2, x_pos[i] + width/2], [y_max, y_max], 'k-', linewidth=1)
        plt.text(x_pos[i], y_max + 0.01, '*', ha='center', fontsize=14, fontweight='bold')

# Human comparison lines
plt.axhline(y=0.343, color='#7A3C38', linestyle='--', alpha=0.7, linewidth=2, label='Human (Uniform)')
plt.axhline(y=0.53, color='#157D35', linestyle='--', alpha=0.7, linewidth=2, label='Human (Zipfian)')

# Chance line
plt.axhline(y=1/28, color='gray', linestyle=':', alpha=0.5)

# Format
plt.xlabel('Model Type', fontsize=14)
plt.ylabel('Matched Frequency Accuracy', fontsize=14)
plt.title('Zipfian Benefit in Matched Frequency Words', fontsize=16)
plt.xticks(x_pos, model_types)
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
plt.ylim(0, 0.8)

plt.tight_layout()
plt.savefig(os.path.join(ANALYSIS_DIR, 'zipfian_benefit_matched_frequency.png'), dpi=300, bbox_inches='tight')
plt.show()

# Figure 3: Frequency gradients (Zipfian only)
plt.figure(figsize=(10, 6))

colors = {'VL-Baseline': '#1f77b4', 'VL-Norm': '#ff7f0e', 'SGCL-Norm': '#2ca02c'}

for model in df_frequency_long['model_type'].unique():
    model_data = df_frequency_long[df_frequency_long['model_type'] == model]
    means = model_data.groupby('frequency')['accuracy'].mean()
    stds = model_data.groupby('frequency')['accuracy'].std()
    counts = model_data.groupby('frequency')['accuracy'].count()
    ses = stds / np.sqrt(counts)
    
    x_pos = np.arange(len(means))
    plt.errorbar(x_pos, means, yerr=ses, 
                 label=model, color=colors.get(model, 'gray'),
                 marker='o', markersize=8, linewidth=2, capsize=5)

# Human data
human_means = {'Low': 0.26, 'Matched': 0.53, 'High': 0.77}
human_ses = {'Low': 0.03, 'Matched': 0.03, 'High': 0.04}

human_x = np.arange(3)
human_y = [human_means[freq] for freq in ['Low', 'Matched', 'High']]
human_yerr = [human_ses[freq] for freq in ['Low', 'Matched', 'High']]
plt.errorbar(human_x, human_y, yerr=human_yerr, 
             label='Human', color='black', 
             marker='s', markersize=8, linewidth=2, capsize=5, linestyle='--')

# Format
plt.xlabel('Frequency Group', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.title('Frequency-Accuracy Gradient by Model (Zipfian Only)', fontsize=16)
plt.xticks(np.arange(3), ['Low', 'Matched', 'High'])
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(ANALYSIS_DIR, 'frequency_gradients.png'), dpi=300, bbox_inches='tight')
plt.show()

# Save results
simple_effects_df.to_csv(os.path.join(ANALYSIS_DIR, 'simple_effects_results.csv'), index=False)
anova_table.to_csv(os.path.join(ANALYSIS_DIR, 'anova_results.csv'))
df_frequency_long.to_csv(os.path.join(ANALYSIS_DIR, 'frequency_gradient_data.csv'), index=False)

print("\n=== All done ===")