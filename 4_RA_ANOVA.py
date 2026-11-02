"""
RA analysis
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
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
ANALYSIS_DIR = os.path.join(BASE_DIR, "analysis", "RA")


os.makedirs(ANALYSIS_DIR, exist_ok=True)

# Load 
FOLDER_TO_LOAD = 'RA_yu_smith' 
CSV_DIR = os.path.join(RESULTS_DIR, FOLDER_TO_LOAD, 'scene_caption.csv')
df_ra = pd.read_csv(CSV_DIR)
df_ra = df_ra[df_ra['test_accuracy'].notna()]  # Drop empty rows

# Figure out model types from the config
def get_model_type(row):
    if row['loss_fn'] == 'triplet_loss' and not row['normalized']:
        return 'VL-Baseline'
    elif row['loss_fn'] == 'triplet_loss' and row['normalized']:
        return 'VL-Norm'
    elif row['loss_fn'] == 'SGCL' and row['normalized']:
        return 'SGCL-Norm'
    else:
        return 'Other'

df_ra['model_type'] = df_ra.apply(get_model_type, axis=1)

# Get referent count from filename
df_ra['n_objs'] = df_ra['experiment_to_load'].str.extract(r'RA_(\d)Ref').astype(int)
df_ra['n_objs_cat'] = pd.Categorical(df_ra['n_objs'])  # For ANOVA

# Keep only the three main models
df_ra = df_ra[df_ra['model_type'].isin(['VL-Baseline', 'VL-Norm', 'SGCL-Norm'])]

print("=== Basic stats ===")
desc_stats = df_ra.groupby(['model_type', 'n_objs'])['test_accuracy'].agg(['mean', 'std', 'count'])
print(desc_stats)

# Main 3x3 ANOVA
print("\n=== 3×3 ANOVA (Model × Referent Count) ===")
anova_model = ols('test_accuracy ~ C(model_type) * C(n_objs_cat)', data=df_ra).fit()
anova_table = sm.stats.anova_lm(anova_model, typ=2)
print(anova_table)

# Get partial eta squared
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

# Check assumptions
print("\############ Assumption checks #######")
residuals = anova_model.resid
w_stat, p_norm = stats.shapiro(residuals)
print(f"Normality (Shapiro-Wilk): W = {w_stat:.3f}, p = {p_norm:.4f}")

groups = [group['test_accuracy'].values for name, group in df_ra.groupby(['model_type', 'n_objs'])]
levene_stat, p_levene = stats.levene(*groups)
print(f"Homogeneity (Levene): F = {levene_stat:.3f}, p = {p_levene:.4f}")

# ##################################
# Simple effects and post-hocs

print("\n" + "="*60)
print("Simple effects and post-hocs")
print("="*60)

# One-way ANOVAs for each model
print("\n=== One-way ANOVAs for each model ===")
simple_effects_results = []
for model in df_ra['model_type'].unique():
    model_data = df_ra[df_ra['model_type'] == model]
    model_aov = ols('test_accuracy ~ C(n_objs_cat)', data=model_data).fit()
    model_aov_table = sm.stats.anova_lm(model_aov, typ=2)
    
    f_value = model_aov_table.loc['C(n_objs_cat)', 'F']
    p_value = model_aov_table.loc['C(n_objs_cat)', 'PR(>F)']
    ss_effect = model_aov_table.loc['C(n_objs_cat)', 'sum_sq']
    ss_total = ss_effect + model_aov_table.loc['Residual', 'sum_sq']
    eta_sq = ss_effect / ss_total
    
    simple_effects_results.append({
        'Model': model,
        'F': f_value,
        'p': p_value,
        'ηp²': eta_sq
    })
    
    print(f"{model}: F = {f_value:.3f}, p = {p_value:.4f}, ηp² = {eta_sq:.3f}")

# Tukey HSD for each model
print("\n=== Tukey HSD post-hocs ===")
for model in df_ra['model_type'].unique():
    model_data = df_ra[df_ra['model_type'] == model]
    print(f"\n{model}:")
    tukey = pairwise_tukeyhsd(
        endog=model_data['test_accuracy'],
        groups=model_data['n_objs'].astype(str),
        alpha=0.05
    )
    print(tukey.summary())

# Compare models at each referent count
print("\n=== Model comparisons at each ambiguity level ===")
for n_obj in [2, 3, 4]:
    n_obj_data = df_ra[df_ra['n_objs'] == n_obj]
    if len(n_obj_data) > 0:
        n_obj_aov = ols('test_accuracy ~ C(model_type)', data=n_obj_data).fit()
        n_obj_aov_table = sm.stats.anova_lm(n_obj_aov, typ=2)
        f_value = n_obj_aov_table.loc['C(model_type)', 'F']
        p_value = n_obj_aov_table.loc['C(model_type)', 'PR(>F)']
        print(f"{n_obj} Referents: F = {f_value:.3f}, p = {p_value:.4f}")

# Linear trends
print("\n=== Linear trends ===")
for model in df_ra['model_type'].unique():
    model_data = df_ra[df_ra['model_type'] == model]
    slope, intercept, r_value, p_slope, std_err = linregress(
        model_data['n_objs'], model_data['test_accuracy']
    )
    print(f"{model}: slope = {slope:.3f}, p = {p_slope:.4f}, R² = {r_value**2:.3f}")

# Save results
simple_effects_df = pd.DataFrame(simple_effects_results)
simple_effects_df.to_csv(os.path.join(ANALYSIS_DIR, 'simple_effects_ra.csv'), index=False)

summary_table = df_ra.groupby(['model_type', 'n_objs'])['test_accuracy'].agg(['mean', 'std', 'count']).round(3)
summary_table.to_csv(os.path.join(ANALYSIS_DIR, 'summary_stats_ra.csv'))

#######################
# Make the figures

print("\n=== Making figures ===")

# Prep data for plotting
summary = df_ra.groupby(['model_type', 'n_objs'])['test_accuracy'].agg([
    'mean', 'std', 'count'
]).reset_index()
summary['se'] = summary['std'] / np.sqrt(summary['count'])

# Human data from Yu & Smith 2007
human_data = pd.DataFrame({
    'model_type': ['Human'] * 3,
    'n_objs': [2, 3, 4],
    'mean': [16/18, 13/18, 10/18],  # From their paper
    'se': [0.06, 0.1, 0.07]
})

plot_data = pd.concat([human_data, summary], ignore_index=True)
plot_order = ['Human', 'VL-Baseline', 'VL-Norm', 'SGCL-Norm']
colors = {2: '#F1766F', 3: '#1EB24B', 4: '#6D95CD'}

# Figure 1: Bar plot with error bars (4 panels)
fig, axes = plt.subplots(1, 4, figsize=(20, 6), sharey=True)
for i, model in enumerate(plot_order):
    ax = axes[i]
    model_data = plot_data[plot_data['model_type'] == model]
    
    bars = ax.bar(model_data['n_objs'].astype(str), 
                  model_data['mean'],
                  color=[colors[x] for x in model_data['n_objs']],
                  width=0.8)
    
    ax.errorbar(model_data['n_objs'].astype(str), 
                model_data['mean'],
                yerr=model_data['se'],
                fmt='none', color='black', capsize=5)
    
    ax.axhline(y=0.25, color='black', linestyle=':', alpha=0.7)
    ax.set_title(model, fontsize=16)
    if i == 0:
        ax.set_ylabel('Proportion Correct', fontsize=14)
    
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.text(0.5, 0.02, 'Number of words and referents', ha='center', fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(ANALYSIS_DIR, 'referential_ambiguity_plot_with_human.png'), dpi=300, bbox_inches='tight')
plt.show()

# Figure 2: Box plots by model
fig, axes = plt.subplots(1, 4, figsize=(20, 6), sharey=True)

# Need synthetic human data for boxplot
np.random.seed(42)
human_synthetic_data = []
for _, row in human_data.iterrows():
    synthetic_points = np.random.normal(row['mean'], row['se']*2, 20)
    synthetic_points = np.clip(synthetic_points, 0, 1)
    for point in synthetic_points:
        human_synthetic_data.append({
            'model_type': 'Human',
            'n_objs': row['n_objs'],
            'test_accuracy': point
        })

human_synthetic_df = pd.DataFrame(human_synthetic_data)
boxplot_data = pd.concat([human_synthetic_df, df_ra[['model_type', 'n_objs', 'test_accuracy']]], ignore_index=True)

for i, model in enumerate(plot_order):
    ax = axes[i]
    model_data = boxplot_data[boxplot_data['model_type'] == model]
    
    sns.boxplot(data=model_data, x='n_objs', y='test_accuracy', 
                ax=ax, palette=['#F1766F', '#1EB24B', '#6D95CD'],
                width=0.7)
    
    ax.axhline(y=0.25, color='black', linestyle=':', alpha=0.7)
    ax.set_title(model, fontsize=16)
    if i == 0:
        ax.set_ylabel('Proportion Correct', fontsize=14)
    
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.text(0.5, 0.02, 'Number of words and referents', ha='center', fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(ANALYSIS_DIR, 'accuracy_distributions_with_human.png'), dpi=300, bbox_inches='tight')
plt.show()

# Figure 3: Comparison across models at each referent count
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
model_order = ['Human', 'VL-Baseline', 'VL-Norm', 'SGCL-Norm']
model_colors = {'Human': '#999999', 'VL-Baseline': '#F1766F', 'VL-Norm': '#1EB24B', 'SGCL-Norm': '#6D95CD'}

for i, n_obj in enumerate([2, 3, 4]):
    ax = axes[i]
    n_obj_data = boxplot_data[boxplot_data['n_objs'] == n_obj]
    
    sns.boxplot(data=n_obj_data, x='model_type', y='test_accuracy', 
                ax=ax, order=model_order, palette=[model_colors[model] for model in model_order],
                width=0.7)
    
    ax.axhline(y=0.25, color='black', linestyle=':', alpha=0.7)
    ax.set_title(f'{n_obj} Words & Referents', fontsize=16, pad=20)
    if i == 0:
        ax.set_ylabel('Proportion Correct', fontsize=14)
    
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.text(0.5, 0.02, 'Model Type', ha='center', fontsize=14)
plt.tight_layout(rect=[0, 0.05, 1, 0.90])
plt.savefig(os.path.join(ANALYSIS_DIR, 'accuracy_by_model_comparison.png'), dpi=300, bbox_inches='tight')
plt.show()

