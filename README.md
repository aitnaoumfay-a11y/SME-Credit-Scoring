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

## Figures

### Financing mode distribution by sector
![Financing distribution](fig1_financement(1).png)

### Key financial ratios by financing mode
![Ratio boxplots](fig2_ratios_boxplot(1).png)

### Correlation matrix
![Correlation heatmap](fig3_correlation(1).png)

### Impact of qualitative factors
![Qualitative factors](fig5_qualitative(1).png)
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
