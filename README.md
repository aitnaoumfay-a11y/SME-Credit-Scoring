# SME-Credit-Scoring
Credit risk scoring framework for 150 Moroccan SMEs — financial ratios, qualitative governance factors, and rule-based rating model (A→E). Python · pandas · seaborn.
# SME Credit Scoring Dashboard 🇲🇦

**Moroccan SME credit risk analysis — financial & qualitative factors**

> Part of Master's thesis research on SME access to bank financing in Morocco  
> FSJES Marrakech — Centre d'Excellence | Applied Finance (Sustainable Finance)

---

## Overview

This project builds a **credit scoring framework** for 150 Moroccan SMEs using financial ratios and qualitative governance variables. It connects directly to empirical fieldwork conducted during an internship at Banque Populaire (Corporate Banking, Business Center).

The central research question: *do qualitative factors (management quality, governance, crisis resilience) add meaningful explanatory power to purely financial scoring models?*

---

## Dataset

- **150 SME observations** across 8 sectors (BTP, Commerce, Agroalimentaire, Services B2B…)
- **54 variables** — financial (CA, EBE, ratios de levier, trésorerie) + qualitative (plan de succession, incidents bancaires, concentration clientèle…)
- Variable `Mode_financement` : 6 categories (Crédit CT, Crédit CT + Garantie, Autofinancement, Crédit MT, Leasing, Refus)

---

## Methodology

### Derived ratios computed
| Ratio | Formula |
|---|---|
| Levier financier | Endettement bancaire / Fonds propres |
| Couverture charges fin. | EBE / Charges financières |
| FDR/BFR | Fonds de roulement / Besoin en fonds de roulement |
| Score risque qualitatif | Encodage ordinal de 4 variables de gouvernance |

### Credit scoring (rule-based)
A weighted rule-based score (0–100) penalises:
- Low EBE margin (< 2% → −20 pts)
- Negative net margin (−15 pts)
- High leverage (D/E > 3 → −20 pts)
- Insufficient debt coverage (coverage < 1.2 → −20 pts)
- Negative net cash position (−15 pts)
- Banking incidents (−5 pts per incident, capped at −15)
- Qualitative risk score (−2 pts per unit)

Ratings: **A** (≥85) · **B** (70–84) · **C** (55–69) · **D** (40–54) · **E** (<40)

---

## Key findings

| Mode de financement | N | Score moyen | EBE/CA moy. | Incidents moy. |
|---|---|---|---|---|
| Refus | 3 | 25.0 | 2.52% | 1.67 |
| Autofinancement | 27 | 46.3 | 4.02% | 0.93 |
| Crédit CT | 54 | 51.9 | 4.30% | 0.78 |
| Crédit CT + Garantie | 40 | 63.6 | 6.02% | 0.68 |
| Leasing | 8 | 61.3 | 4.89% | 0.25 |
| Crédit MT | 18 | 73.5 | 7.01% | 0.72 |

---

## Outputs

| Figure | Description |
|---|---|
| `fig1_financement.png` | Distribution des modes de financement par secteur |
| `fig2_ratios_boxplot.png` | Ratios financiers clés par mode de financement |
| `fig3_correlation.png` | Matrice de corrélation — variables financières + score |
| `fig4_dashboard.png` | Dashboard principal — score crédit et ratings |
| `fig5_qualitative.png` | Impact des facteurs qualitatifs sur le financement |
| `sme_scored.csv` | Dataset complet avec scores et ratings |
| `summary_by_financing.csv` | Statistiques agrégées par mode de financement |

---

## Usage

```bash
pip install pandas numpy matplotlib seaborn scikit-learn odfpy
python sme_credit_scoring.py
```

To use with your own dataset, replace the `load_data()` path and ensure your columns match the expected schema (see top of script).

---

## Stack

`Python 3.10+` · `pandas` · `numpy` · `matplotlib` · `seaborn` · `scikit-learn`

---
"""
SME Credit Scoring Dashboard
=============================
Author : Fadoua Ait Naoum — FSJES Marrakech, Centre d'Excellence
Master of Excellence in Applied Finance (Sustainable Finance)

Context
-------
This project analyses a dataset of 150 Moroccan SMEs to build a
credit-risk scoring framework based on financial and qualitative
variables. The analysis supports the thesis research on SME access
to bank financing in Morocco.

Data : ID_PME_24.ods  (150 observations, 54 variables)
Tools: pandas, matplotlib, seaborn, scikit-learn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ── 0. Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.dpi": 150,
})

PALETTE = {
    "Crédit CT":            "#378ADD",
    "Crédit CT + Garantie": "#1D9E75",
    "Autofinancement":      "#EF9F27",
    "Crédit MT":            "#D4537E",
    "Leasing":              "#7F77DD",
    "Refus":                "#E24B4A",
}

# ── 1. Load data ───────────────────────────────────────────────────────────────
def load_data(path: str = "ID_PME_24.CSV") -> pd.DataFrame:
    df = pd.read_excel(path, engine="odf")
    return df

# ── 2. Feature engineering — derived financial ratios ─────────────────────────
def compute_ratios(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Leverage : net bank debt / equity
    df["Levier_financier"] = df["Endettement_bancaire_MMAD"] / df["Fonds_propres_MMAD"].replace(0, np.nan)

    # Debt coverage : EBE / financial charges
    df["Couverture_charges_fin"] = df["EBE_MMAD"] / df["Charges_fin_MMAD"].replace(0, np.nan)

    # Working capital sufficiency : FDR / BFR
    df["FDR_BFR"] = df["FDR_MMAD"] / df["BFR_MMAD"].replace(0, np.nan)

    # Short-term debt share
    df["Part_dette_CT"] = df["Dette_CT_MMAD"] / (
        df["Dette_CT_MMAD"] + df["Dette_LMT_MMAD"]
    ).replace(0, np.nan)

    # Cash flow margin
    df["Cash_flow_CA_calc"] = df["Cash_flow_MMAD"] / df["CA_MMAD"].replace(0, np.nan) * 100

    # Qualitative risk score (0–5)  — ordinal encoding of key soft variables
    risk_map = {
        "Plan_succession":              {"Oui": 0, "Non": 2},
        "Performance_derniere_crise":   {"Bonne": 0, "Moyenne": 1, "Mauvaise": 2},
        "Respect_remise_docs_compta":   {"Toujours": 0, "Parfois": 1, "Rarement": 2},
        "Assurance_mandataire":         {"Oui": 0, "Non": 1},
    }
    df["Score_qualitatif_risque"] = 0
    for col, mapping in risk_map.items():
        df["Score_qualitatif_risque"] += df[col].map(mapping).fillna(1)

    return df

# ── 3. Credit scoring (rule-based + weighted) ──────────────────────────────────
def credit_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def score_row(row):
        score = 100  # start from 100

        # Profitability
        if row["EBE_CA_pct"] < 2:      score -= 20
        elif row["EBE_CA_pct"] < 5:    score -= 10

        if row["Resultat_net_CA_pct"] < 0:   score -= 15
        elif row["Resultat_net_CA_pct"] < 2: score -= 7

        # Leverage
        lev = row.get("Levier_financier", np.nan)
        if pd.notna(lev):
            if lev > 3:    score -= 20
            elif lev > 2:  score -= 10

        # Debt coverage
        cov = row.get("Couverture_charges_fin", np.nan)
        if pd.notna(cov):
            if cov < 1.2:  score -= 20
            elif cov < 2:  score -= 10

        # Liquidity
        if row["Tresorerie_nette_MMAD"] < 0:  score -= 15
        elif row["Tresorerie_nette_MMAD"] < 0.5: score -= 5

        # Incidents
        score -= min(row["Nb_incidents_12m"] * 5, 15)

        # Qualitative risk
        score -= row.get("Score_qualitatif_risque", 0) * 2

        return max(score, 0)

    df["Credit_Score"] = df.apply(score_row, axis=1)

    df["Rating"] = pd.cut(
        df["Credit_Score"],
        bins=[0, 40, 55, 70, 85, 100],
        labels=["E — Très risqué", "D — Risqué", "C — Modéré", "B — Acceptable", "A — Faible risque"],
        include_lowest=True,
    )
    return df

# ── 4. Visualisations ─────────────────────────────────────────────────────────

def fig1_distribution_financement(df):
    """Bar chart — financing mode distribution by sector."""
    ct = pd.crosstab(df["Secteur"], df["Mode_financement"])
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # left — absolute counts
    counts = df["Mode_financement"].value_counts()
    bars = axes[0].barh(counts.index, counts.values,
                        color=[PALETTE.get(m, "#888") for m in counts.index])
    axes[0].set_xlabel("Nombre de PME")
    axes[0].set_title("Répartition par mode de financement", fontweight="bold")
    for bar, val in zip(bars, counts.values):
        axes[0].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                     str(val), va="center", fontsize=10)

    # right — heatmap by sector
    sns.heatmap(ct_pct, ax=axes[1], cmap="Blues", annot=True, fmt=".0f",
                linewidths=0.5, cbar_kws={"label": "%"})
    axes[1].set_title("Mode de financement par secteur (%)", fontweight="bold")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig("fig1_financement.png", bbox_inches="tight")
    plt.close()
    print("✓ fig1_financement.png saved")


def fig2_ratio_boxplots(df):
    """Boxplots of key financial ratios by financing mode."""
    ratios = {
        "EBE_CA_pct":          "Marge EBE (%)",
        "Resultat_net_CA_pct": "Marge nette (%)",
        "Levier_financier":    "Levier financier",
        "Tresorerie_nette_MMAD": "Trésorerie nette (MMAD)",
    }
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    order = list(PALETTE.keys())
    colors = list(PALETTE.values())

    for ax, (col, label) in zip(axes, ratios.items()):
        # clip outliers for readability
        q99 = df[col].quantile(0.99)
        plot_df = df[df[col] <= q99]
        sns.boxplot(
            data=plot_df, x="Mode_financement", y=col, ax=ax,
            order=[o for o in order if o in df["Mode_financement"].unique()],
            palette=PALETTE, linewidth=0.8, fliersize=3,
        )
        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle("Ratios financiers clés par mode de financement", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("fig2_ratios_boxplot.png", bbox_inches="tight")
    plt.close()
    print("✓ fig2_ratios_boxplot.png saved")


def fig3_correlation_heatmap(df):
    """Correlation heatmap — financial variables."""
    num_cols = [
        "CA_MMAD", "EBE_CA_pct", "Resultat_net_CA_pct",
        "Levier_financier", "Couverture_charges_fin",
        "Tresorerie_nette_MMAD", "Nb_incidents_12m",
        "Score_qualitatif_risque", "Credit_Score",
    ]
    corr = df[num_cols].corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, linewidths=0.5, ax=ax,
                cbar_kws={"label": "Corrélation de Spearman (approx.)"})
    ax.set_title("Matrice de corrélation — variables financières et score crédit",
                 fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig("fig3_correlation.png", bbox_inches="tight")
    plt.close()
    print("✓ fig3_correlation.png saved")


def fig4_credit_score_dashboard(df):
    """Main dashboard — credit score distribution and rating breakdown."""
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    rating_colors = {
        "A — Faible risque":   "#1D9E75",
        "B — Acceptable":      "#378ADD",
        "C — Modéré":          "#EF9F27",
        "D — Risqué":          "#D4537E",
        "E — Très risqué":     "#E24B4A",
    }

    # (0,0) histogram of scores
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.hist(df["Credit_Score"], bins=20, color="#378ADD", alpha=0.8, edgecolor="white")
    ax1.axvline(df["Credit_Score"].mean(), color="#E24B4A", linestyle="--",
                label=f"Moyenne : {df['Credit_Score'].mean():.1f}")
    ax1.set_title("Distribution du score crédit (0–100)", fontweight="bold")
    ax1.set_xlabel("Score")
    ax1.set_ylabel("Nombre de PME")
    ax1.legend()

    # (0,2) donut — rating distribution
    ax2 = fig.add_subplot(gs[0, 2])
    rating_counts = df["Rating"].value_counts().sort_index()
    wedges, texts, autotexts = ax2.pie(
        rating_counts.values,
        labels=None,
        autopct="%1.0f%%",
        colors=[rating_colors.get(r, "#aaa") for r in rating_counts.index],
        startangle=90,
        wedgeprops=dict(width=0.55),
    )
    ax2.legend(rating_counts.index, loc="lower center", bbox_to_anchor=(0.5, -0.35),
               fontsize=8, frameon=False)
    ax2.set_title("Répartition par rating", fontweight="bold")

    # (1,0) score by sector
    ax3 = fig.add_subplot(gs[1, :2])
    sector_score = df.groupby("Secteur")["Credit_Score"].mean().sort_values()
    bars = ax3.barh(sector_score.index, sector_score.values,
                    color=["#1D9E75" if v >= 65 else "#EF9F27" if v >= 50 else "#E24B4A"
                           for v in sector_score.values])
    ax3.axvline(65, color="gray", linestyle=":", alpha=0.7, label="Seuil acceptable (65)")
    ax3.set_title("Score crédit moyen par secteur", fontweight="bold")
    ax3.set_xlabel("Score moyen")
    ax3.legend(fontsize=9)
    for bar, val in zip(bars, sector_score.values):
        ax3.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f}", va="center", fontsize=9)

    # (1,2) incidents vs score scatter
    ax4 = fig.add_subplot(gs[1, 2])
    scatter = ax4.scatter(
        df["Nb_incidents_12m"], df["Credit_Score"],
        c=df["Score_qualitatif_risque"], cmap="RdYlGn_r",
        alpha=0.6, s=30, edgecolors="white", linewidths=0.3,
    )
    plt.colorbar(scatter, ax=ax4, label="Score risque qualitatif")
    ax4.set_xlabel("Incidents (12 mois)")
    ax4.set_ylabel("Score crédit")
    ax4.set_title("Incidents vs score crédit", fontweight="bold")

    fig.suptitle("SME Credit Scoring Dashboard — Maroc (n=150)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.savefig("fig4_dashboard.png", bbox_inches="tight")
    plt.close()
    print("✓ fig4_dashboard.png saved")


def fig5_qualitative_impact(df):
    """Stacked bar — qualitative factor influence on financing mode."""
    qual_vars = [
        ("Plan_succession",            "Plan de succession"),
        ("Performance_derniere_crise", "Performance crise"),
        ("Respect_remise_docs_compta", "Respect docs compta"),
        ("Assurance_mandataire",       "Assurance mandataire"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.flatten()

    for ax, (col, label) in zip(axes, qual_vars):
        ct = pd.crosstab(df[col], df["Mode_financement"], normalize="index") * 100
        ct.plot(kind="bar", ax=ax, color=list(PALETTE.values())[:len(ct.columns)],
                edgecolor="white", linewidth=0.5)
        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("% PME")
        ax.tick_params(axis="x", rotation=20)
        ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("Impact des facteurs qualitatifs sur le mode de financement (%)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("fig5_qualitative.png", bbox_inches="tight")
    plt.close()
    print("✓ fig5_qualitative.png saved")


# ── 5. Summary statistics table ───────────────────────────────────────────────
def export_summary(df):
    summary = df.groupby("Mode_financement").agg(
        N=("ID_PME", "count"),
        CA_moyen_MMAD=("CA_MMAD", "mean"),
        EBE_CA_moy_pct=("EBE_CA_pct", "mean"),
        Marge_nette_moy=("Resultat_net_CA_pct", "mean"),
        Levier_moy=("Levier_financier", "mean"),
        Score_credit_moy=("Credit_Score", "mean"),
        Incidents_moy=("Nb_incidents_12m", "mean"),
    ).round(2)
    summary.to_csv("summary_by_financing.csv")
    print("✓ summary_by_financing.csv saved")
    print(summary.to_string())


# ── 6. Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data...")
    df = load_data("ID_PME_24.ods")

    print("Computing derived ratios...")
    df = compute_ratios(df)

    print("Computing credit scores...")
    df = credit_score(df)

    print("\nGenerating figures...")
    fig1_distribution_financement(df)
    fig2_ratio_boxplots(df)
    fig3_correlation_heatmap(df)
    fig4_credit_score_dashboard(df)
    fig5_qualitative_impact(df)

    print("\nExporting summary table...")
    export_summary(df)

    df.to_csv("sme_scored.csv", index=False)
    print("\n✓ sme_scored.csv — full dataset with scores exported")
    print("\nAll done. 5 figures + 2 CSV files generated.")
    plt.show()
