# GDP Forecasting with VAR/VECM Models

## 📌 Theoretical Foundations

### 1. GDP as the Core Indicator
Gross Domestic Product (GDP) measures the total value of goods and services produced in a country over a given period.  
It is influenced by:
- **Domestic demand** (consumption, investment),
- **External demand** (net exports, trade balance),
- **Public policies** (fiscal and monetary).

---

### 2. Econometric Approach: Time Series Models
This project relies on **multivariate time series models** to capture the dynamic interactions between GDP, investment, and trade balance.

#### a) VAR (Vector AutoRegressive) – *Sims, 1980*
- All variables are treated as **endogenous**.  
- Each variable depends on its own past values and the past values of the other variables.  
- Requires **stationary data** (constant mean and variance over time).  

#### b) VECM (Vector Error Correction Model) – *Engle & Granger, 1987; Johansen, 1991*
- Used when variables are **non-stationary but cointegrated**.  
- Captures:
  - **Short-term dynamics** (differences of variables),  
  - **Long-term equilibrium** (cointegration relationships).  

👉 In this project, the Johansen test failed to converge, so a **differenced VAR model** was retained.

---

### 3. Economic Justification of Variables
- **Investment**: key driver of growth (capital accumulation → higher production).  
- **Trade Balance (% of GDP)**: reflects external openness and foreign demand.  
- **GDP**: target variable, but also influences others (income effects, imports, savings).  

This setup follows growth models in an **open economy context** (Solow, Harrod-Domar, post-Keynesian approaches).

---

### 4. Underlying Assumptions
- Stationarity (or achieved by differencing).  
- Endogeneity: all variables are jointly determined.  
- Linearity: VAR assumes linear dynamic relations.  
- Errors are white noise (no autocorrelation, homoscedastic).  

---

### 5. Policy Relevance
- Short-term **GDP forecasting**.  
- Understanding the role of investment and trade in economic growth.  
- Basis for **impulse response analysis** and **variance decomposition** (future extensions).

---

## 🎯 Summary
This project applies **VAR/VECM econometric theory** to model GDP, focusing on the **dynamic interactions** between GDP, investment, and trade balance in Benin.  
It bridges **macroeconomic theory** and **applied forecasting** for policy and decision-making.
