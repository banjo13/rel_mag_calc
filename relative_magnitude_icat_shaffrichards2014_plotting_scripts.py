# ----------------------
# Cleaned Imports
# ----------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------
# Functions
# ----------------------

def plot_catalog_vs_relative_magnitude(df):
    """
    Plot catalog magnitude vs. estimated relative magnitude (scatterplot).
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='cat_mag', y='relmag_mean', hue='method', alpha=0.6)

    min_val = df[['cat_mag', 'relmag_mean']].min().min()
    max_val = df[['cat_mag', 'relmag_mean']].max().max()
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x')

    plt.xlabel("Catalog Magnitude")
    plt.ylabel("Estimated Relative Magnitude (Mean)")
    plt.title("Catalog vs. Estimated Relative Magnitude (Per Method)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_catalog_vs_relative_magnitude_per_method(df):
    """
    Plot catalog vs estimated magnitude separately for each method with error bars.
    """
    methods = df['method'].unique()
    for method in methods:
        df_method = df[df['method'] == method]

        fig = plt.figure(figsize=(8, 6))
        plt.errorbar(
            df_method['cat_mag'],
            df_method['relmag_mean'],
            yerr=df_method['relmag_std'],
            fmt='o', ecolor='gray', capsize=2, markersize=5,
            linestyle='None', color='k', alpha=0.7, label=None, zorder=2
        )
        scatter = plt.scatter(
            df_method['cat_mag'], df_method['relmag_mean'],
            c=df_method['n_estimates'], cmap='viridis', s=25,
            edgecolor='k', alpha=0.8, zorder=3
        )

        min_val = min(df_method['cat_mag'].min(), df_method['relmag_mean'].min())
        max_val = max(df_method['cat_mag'].max(), df_method['relmag_mean'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x', zorder=1)

        plt.xlabel("Catalog Magnitude")
        plt.ylabel("Estimated Relative Magnitude")
        plt.title(f"Catalog vs Estimated Magnitude — {method.upper()} Method")
        plt.colorbar(scatter, label="Number of Estimates")
        plt.grid(True)
        plt.tight_layout()
        fig.savefig(f'catalog_estimated_{method}.png', dpi=300, bbox_inches='tight')
        plt.show()

def plot_histogram_of_residuals(df):
    """
    Plot histogram of residuals (relative - catalog magnitude).
    """
    fig = plt.figure()
    sns.histplot(data=df, x='resid_mean', hue='method', kde=True, bins=40)
    plt.axvline(0, color='k', linestyle='--')
    plt.xlabel("Residual (Relative - Catalog Magnitude)")
    plt.title("Histogram of Residuals")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_violin_residuals_top_events(long_df):
    """
    Plot violin plots of residuals for top 20 events.
    """
    top_events = (
        long_df.groupby('evid')
        .count()
        .sort_values('relmag', ascending=False)
        .head(20)
        .index
    )

    subset = long_df[long_df['evid'].isin(top_events)]

    plt.figure(figsize=(12, 6))
    sns.violinplot(data=subset, x='evid', y='residual', hue='method', split=True, inner='box')
    plt.axhline(0, color='k', linestyle='--')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Residual (Relative – Catalog Magnitude)")
    plt.title("Residual Distributions per Event (Top 20)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_strip_relative_magnitudes_top_events(long_df, all_mags):
    """
    Plot strip plots of relative magnitudes for top 10 events with catalog mag reference lines.
    """
    top_events = (
        long_df.groupby('evid')
        .count()
        .sort_values('relmag', ascending=False)
        .head(10)
        .index
    )

    subset = long_df[long_df['evid'].isin(top_events)]
    cat_mag_map = {evid: all_mags[int(evid)]['cat'] for evid in top_events}

    plt.figure(figsize=(12, 6))
    sns.stripplot(data=subset, x='evid', y='relmag', hue='method', dodge=True, alpha=0.6, jitter=True)

    for i, evid in enumerate(top_events):
        plt.hlines(cat_mag_map[evid], i - 0.3, i + 0.3, colors='red', linestyles='--', linewidth=1)

    plt.ylabel("Relative Magnitude Estimates")
    plt.title("Relative Magnitude Estimates per Event (Top 10)")
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.legend()
    plt.show()

def compute_b_value(magnitudes, Mmin, dM=0.1):
    """
    Compute the b-value and its standard deviation using the maximum likelihood method.
    """
    mags = np.array(magnitudes)
    mags_sel = mags[mags >= Mmin]
    N = len(mags_sel)
    mean_M = np.mean(mags_sel)
    b_val = np.log10(np.e) / (mean_M - (Mmin - dM / 2))
    sigma = b_val / np.sqrt(N)
    return b_val, sigma, N

def plot_gutenberg_richter_distribution(magnitudes, b, dM=0.1):
    """
    Plot cumulative Gutenberg-Richter magnitude-frequency distribution.
    """
    bins = np.arange(min(magnitudes), max(magnitudes) + dM, dM)
    hist, edges = np.histogram(magnitudes, bins=bins)
    cum_counts = np.cumsum(hist[::-1])[::-1]
    bin_centers = edges[:-1] + dM / 2

    plt.figure()
    plt.plot(bin_centers, np.log10(cum_counts), marker='o')
    plt.xlabel('Magnitude')
    plt.ylabel('Log$_{10}$ Cumulative Number')
    plt.title(f'Gutenberg–Richter Distribution (b = {b:.2f})')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ----------------------
# Main Execution
# ----------------------

# Load precomputed relative magnitude DataFrame
df = pd.read_pickle('hoodF_1D_coh_rmag.pkl')

# Step 1: Build all_mags dictionary
all_mags = {}
for _, row in df.iterrows():
    evid1 = row['evid1']
    evid2 = row['evid2']
    cat_mag1 = row['cat_mag1']
    
    rel_dmag_dot = row['m_dmagDot']
    rel_dmag_L2 = row['m_dmagL2']

    for evid, relmag, method in [(evid2, cat_mag1 + rel_dmag_dot, 'dot'), (evid2, cat_mag1 + rel_dmag_L2, 'L2')]:
        if evid not in all_mags:
            all_mags[evid] = {'dot': [], 'L2': [], 'cat': row['cat_mag2']}
        all_mags[evid][method].append(relmag)

# Step 2: Aggregate residuals into relmag_df
rows = []
for evid, mags in all_mags.items():
    for method in ['dot', 'L2']:
        estimates = mags[method]
        cat = mags['cat']
        rows.append({
            'evid': evid,
            'method': method,
            'n_estimates': len(estimates),
            'relmag_mean': np.mean(estimates),
            'relmag_median': np.median(estimates),
            'relmag_std': np.std(estimates),
            'cat_mag': cat,
            'resid_mean': np.mean(np.array(estimates) - cat),
            'resid_median': np.median(np.array(estimates) - cat),
            'resid_std': np.std(np.array(estimates) - cat)
        })
relmag_df = pd.DataFrame(rows)

# Step 3: Create long-form DataFrame
long_data = []
for evid, mags in all_mags.items():
    cat_mag = mags['cat']
    for method in ['dot', 'L2']:
        for est in mags[method]:
            long_data.append({
                'evid': str(evid),
                'method': method,
                'relmag': est,
                'residual': est - cat_mag
            })
long_df = pd.DataFrame(long_data)

# Step 4: Plot Results
plot_catalog_vs_relative_magnitude(relmag_df)
plot_catalog_vs_relative_magnitude_per_method(relmag_df)
plot_histogram_of_residuals(relmag_df)
plot_violin_residuals_top_events(long_df)
plot_strip_relative_magnitudes_top_events(long_df, all_mags)

# Step 5: Compute and plot b-value
icat = pd.read_csv('/Users/bnjo/home/bnjo/hoodF/util/hoodF_icat.csv')
icat['evids'] = [str(e) for e in icat['evids']]

magnitudes = icat['mag'].values
Mmin = 0.8
b_val, sigma, N = compute_b_value(magnitudes, Mmin)

print(f"Number of events (M ≥ {Mmin}): {N}")
print(f"Estimated b-value: {b_val:.3f}")
print(f"Standard deviation σ(b): {sigma:.3f}")

plot_gutenberg_richter_distribution(magnitudes, b_val)
