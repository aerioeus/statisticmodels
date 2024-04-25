"""
This script performs a linear regression analysis using the OLS model from statsmodels.
It adds a constant to the data for the intercept, fits the model, and calculates
various statistics such as predicted values, residuals, and DFFITS values.
"""

import numpy as np
import pandas as pd
from statsmodels.api import OLS, add_constant



# Datenbasis
data = {
    "X": [0.10000, 0.45401, 1.09765, 1.27936, 2.20611, 2.50064, 3.04030, 3.23583, 4.45308,
          4.16990, 5.28474, 5.59238, 5.92091, 6.66066, 6.79953, 7.97943, 8.41536, 8.71607,
          8.70156, 9.16463, 4.00000],
    "y": [-0.0716, 4.1673, 6.5703, 13.8150, 11.4501, 12.9554, 20.1575, 17.5633, 26.0317,
          22.7573, 26.3030, 30.6885, 33.9402, 30.9228, 34.1100, 44.4536, 46.5022, 50.0568,
          46.5475, 45.7762, 40.0000]
}

# Erstellen eines DataFrame
df = pd.DataFrame(data)

# Hinzufügen einer Konstanten für den Intercept
X = add_constant(df["X"])

# Lineare Regression Modell
model = OLS(df["y"], X).fit()

# Vorhersagen und Residuen
df['y_pred'] = model.predict(X)
df['residuals'] = model.resid

# Leverages und geschätzte Varianz der Residuen ohne die i-te Beobachtung
influence = model.get_influence()
sigma2 = np.mean(model.resid ** 2)

# DFFITS Werte
dffits = influence.dffits[0]

# Ergebnisse zusammenstellen
results = pd.DataFrame({
    "X": df["X"],
    "y": df["y"],
    "y_pred": df["y_pred"],
    "residuals": df['residuals'],
    "DFFITS": dffits
})
print(results)
