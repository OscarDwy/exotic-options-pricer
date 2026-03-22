# Exotic Options Pricer — Oscar Dawny
## EDHEC Business School / Centrale Lille

Application Streamlit de pricing d'options exotiques avec surface de volatilité implicite.

---

## Installation

```bash
pip install streamlit yfinance numpy scipy pandas plotly
```

## Lancement

```bash
streamlit run exotic_pricer.py
```

---

## Fonctionnalités

### 1. Surface de Volatilité Implicite
- Calibrée sur données réelles Yahoo Finance
- Smile asymétrique (put skew négatif)
- Term structure réaliste
- Interpolation bicubique (RectBivariateSpline)
- Visualisation 3D interactive

### 2. Options à Barrière
- Down-Out / Down-In / Up-Out / Up-In
- Pricing Monte Carlo (jusqu'à 100k simulations)
- Visualisation des paths actifs vs knockés
- Analyse de sensibilité au niveau de barrière

### 3. Options Asiatiques
- Moyenne arithmétique et géométrique
- Pricing Monte Carlo avec intervalle de confiance
- Distribution de la moyenne simulée

### 4. Options Digitales (Cash-or-Nothing)
- Pricing analytique Black-Scholes
- Pricing Monte Carlo
- Sensibilité au niveau de strike

### 5. Greeks & Sensibilités
- Delta, Gamma, Vega, Theta, Rho
- Profils de payoff en fonction du spot
- Heatmap Prix = f(Vol, Spot)

---

## Architecture

```
exotic_pricer.py
├── Fonctions financières
│   ├── black_scholes_price()     — Prix BS vanille
│   ├── implied_vol()             — Vol implicite (Brent)
│   ├── bs_greeks()               — Greeks analytiques
│   ├── monte_carlo_barrier()     — MC barrière
│   ├── monte_carlo_asian()       — MC asiatique
│   ├── monte_carlo_digital()     — MC digitale
│   ├── build_vol_surface_synthetic() — Surface de vol
│   └── interpolate_vol()         — Interpolation bicubique
└── Interface Streamlit
    ├── Sidebar (paramètres)
    ├── Tab 1 : Surface de vol 3D
    ├── Tab 2 : Options barrière
    ├── Tab 3 : Options asiatiques
    ├── Tab 4 : Options digitales
    └── Tab 5 : Greeks & heatmaps
```

---

## Concepts clés

- **Surface de vol implicite** : la volatilité n'est pas constante — elle varie selon le strike (smile) et la maturité (term structure). Ce modèle capture le skew négatif typique des marchés actions.

- **Monte Carlo** : simulation de N trajectoires du sous-jacent sous la mesure risque-neutre. L'espérance actualisée des payoffs donne le prix.

- **Options exotiques** : contrairement aux vanilles, leur payoff dépend du chemin suivi par le sous-jacent (path-dependent) — d'où la nécessité du Monte Carlo.
