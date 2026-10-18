# 🍃 Air Quality Analysis System

A comprehensive, interactive Python software mini-project that ingests, validates, explores, and analyzes environmental air quality datasets without requiring physical hardware sensors. Built with **Streamlit**, **Pandas**, **NumPy**, **Matplotlib**, **Scikit-learn**, and **ReportLab**.

---

## 🌟 Key Features

1. **Modern Glassmorphism UI & Ambient Dark Theme**:
   - Translucent frosted glass containers (`backdrop-filter: blur(16px)`), neon cyan/rose glowing accents, luminous badges, and ambient midnight gradients.
   - Clean, publication-grade dark-themed charts seamlessly integrated with the glass cards.

2. **Intuitive Dropdown-Driven Navigation & Quick Presets**:
   - Quick date window presets: *All Records, Last 7 Days, Last 14 Days, Last 30 Days, Custom Calendar Range*.
   - Simple location filtering and criteria pollutant selectors.

3. **⚖️ Head-to-Head Location Comparison (City A vs City B)**:
   - Dedicated comparison tab with interactive dropdowns to choose any two monitoring stations.
   - Dual-line comparative time series chart with shaded areas and sample counts.
   - Side-by-side multi-pollutant fingerprint grouped bar chart.
   - Automatic delta metrics and percentage difference calculations.

4. **🔬 Bivariate Scatter Diagnostics & Inter-Pollutant Heatmap**:
   - Dropdown selectors for X-Axis and Y-Axis indicators (e.g., PM2.5 vs AQI, NO2 vs O3).
   - Scatter plot with dynamic linear regression trendline and Pearson correlation ($r$).
   - Full inter-indicator Pearson correlation matrix heatmap.

5. **🩺 Personalized Health Risk & Activity Advisory**:
   - Demographic profile dropdown: *General Healthy Population, Children & Senior Citizens, Asthmatic & Respiratory Patients, Outdoor Athletes & Commuters*.
   - Context-aware recommendations for outdoor exercise, N95/FFP2 mask usage, and indoor HEPA air filtration.

6. **🔮 What-If Scenario Simulator**:
   - Interactive dropdown to simulate interventions: *EV & Public Transit Policy (-25% PM2.5), Monsoon Rainfall Washout (-40%), Winter Inversion Episode (+35%), Arid Dust Storm (+60% PM10), or Custom Sliders*.
   - Instant visual prediction of projected AQI, delta, and regulatory category shifts.

7. **Robust Data Ingestion & Quality Auditing**:
   - Safe CSV ingestion with automatic header normalization and column alias resolution.
   - Strict validation: identifies missing, non-numeric, and negative AQI values (< 0).
   - Non-destructive processing: never mutates raw files and never performs silent imputation.

8. **Official AQI Standard Classification**:
   - **India National Air Quality Index (NAQI - Central Pollution Control Board CPCB)**
   - **United States EPA AQI Standard**
   - **Raw Numeric Mode** (suppresses arbitrary categorization when standard is unverified)

9. **Advanced Report Exporting**:
   - **Executive PDF Report Export** (`ReportLab`): Downloads an executive report containing metadata, summary tables, location rankings, and audit notes.
   - **Filtered CSV Export**: One-click download of processed data.


---

## 📂 Project Architecture

```
Air Quality Analysis System/
├── app.py                      # Main Streamlit dashboard interface
├── requirements.txt            # Project dependencies
├── README.md                   # Documentation, mathematical guide & run instructions
├── .gitignore                  # Git ignore rules for virtualenvs, caches, reports
├── data/
│   └── sample_air_quality.csv  # Synthetic benchmark demonstration dataset
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # CSV loading, normalization, validation & cleaning
│   ├── analysis.py             # Reusable statistical & grouping functions
│   ├── charts.py               # Reusable styled Matplotlib visualizations
│   ├── pdf_exporter.py         # ReportLab PDF analytical report generator
│   ├── ml_predictor.py         # Evaluated Scikit-learn predictive modeling
│   └── generate_demo_data.py   # Demonstration dataset generator
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py     # Automated unit tests for data loader
│   └── test_analysis.py        # Automated unit tests for statistics, charts, PDF, ML
└── reports/
    └── .gitkeep                # Directory for generated reports
```

---

## 🚀 Installation & Running Instructions

### Prerequisites
- Python 3.9 or higher (tested on Python 3.13)
- Windows PowerShell or Command Prompt

### Step 1: Clone / Navigate to Project Directory
```powershell
cd "D:\PIMV Site\Air Quality Analysis System"
```

### Step 2: (Recommended) Create & Activate Virtual Environment

**Using Windows PowerShell:**
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```
> *Note:* If PowerShell displays an `Execution_Policies` restriction script error, run:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```
> Or use Command Prompt:
> ```cmd
> .venv\Scripts\activate.bat
> ```

### Step 3: Install Required Dependencies
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Step 4: Run the Streamlit Dashboard
```powershell
python -m streamlit run app.py
```
Streamlit will launch the web application in your default browser at `http://localhost:8501`.

---

## 🧪 Automated Testing

The project includes an automated test suite with 21 unit tests covering CSV parsing, edge cases, negative AQI filtering, statistical calculations, AQI breakpoints, PDF generation, and machine learning models.

To run all tests:
```powershell
python -m pytest tests/ -v
```

---

## 📊 Dataset Schema

The system supports standard CSV files with flexible headers. Aliases and case variations are automatically normalized:

| Target Field | Required? | Accepted Header Aliases | Example Value | Notes |
|:---|:---:|:---|:---|:---|
| **AQI** | **Required** | `aqi`, `air quality index`, `air_quality_index`, `overall aqi` | `142` | Must be numeric and $\ge 0$. |
| **Date** | Optional | `date`, `datetime`, `timestamp`, `time`, `observation_date` | `2026-01-15` | Enables time-series trends and rolling averages. |
| **Location**| Optional | `location`, `city`, `station`, `site`, `station_name` | `Delhi` | Enables spatial rankings and comparative charts. |
| **PM2.5** | Optional | `pm2.5`, `pm25`, `pm_2.5`, `pm2_5` | `54.2` | Particulate matter $\le 2.5\ \mu\text{m}$. |
| **PM10** | Optional | `pm10`, `pm_10` | `98.0` | Particulate matter $\le 10\ \mu\text{m}$. |
| **NO2** | Optional | `no2`, `nitrogen_dioxide` | `32.4` | Nitrogen Dioxide. |
| **SO2** | Optional | `so2`, `sulfur_dioxide` | `14.1` | Sulfur Dioxide. |
| **CO** | Optional | `co`, `carbon_monoxide` | `1.15` | Carbon Monoxide. |
| **O3** | Optional | `o3`, `ozone` | `28.6` | Ground-level Ozone. |

---

## 📐 Mathematical & Statistical Formulations

- **Arithmetic Mean**:
  $$\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i$$
- **Sample Standard Deviation** (unbiased, $n-1$ degrees of freedom):
  $$s = \sqrt{\frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2}$$
- **Data Range**:
  $$\text{Range} = \max(x) - \min(x)$$
- **Interquartile Range**:
  $$IQR = Q_3 - Q_1$$

---

## 🏛️ Regulatory AQI Breakpoints

### India National Air Quality Index (CPCB)
| Category | AQI Range | Color | Primary Health Impact |
|:---|:---:|:---:|:---|
| **Good** | 0 – 50 | `#00b050` | Minimal impact |
| **Satisfactory** | 51 – 100 | `#92d050` | Minor breathing discomfort to sensitive people |
| **Moderate** | 101 – 200 | `#ffff00` | Breathing discomfort to people with lung disease/asthma |
| **Poor** | 201 – 300 | `#ff9900` | Breathing discomfort to most people on prolonged exposure |
| **Very Poor** | 301 – 400 | `#ff0000` | Respiratory illness on prolonged exposure |
| **Severe** | 401 – 500 | `#7030a0` | Affects healthy people and seriously impacts vulnerable groups |

### US EPA Air Quality Index
| Category | AQI Range | Color | Advisory Description |
|:---|:---:|:---:|:---|
| **Good** | 0 – 50 | `#00e400` | Air quality is satisfactory; poses little or no risk |
| **Moderate** | 51 – 100 | `#ffff00` | Acceptable quality; moderate health concern for sensitive individuals |
| **Unhealthy for Sensitive Groups** | 101 – 150 | `#ff7e00` | General public not likely affected; sensitive groups at risk |
| **Unhealthy** | 151 – 200 | `#ff0000` | Everyone may begin to experience adverse effects |
| **Very Unhealthy** | 201 – 300 | `#8f3f97` | Health alert: entire population likely affected |
| **Hazardous** | 301 – 500 | `#7e0023` | Health warnings of emergency conditions |

---

## ⚠️ Scientific Limitations & Disclosures

1. **Descriptive, Not Causal**: This application performs descriptive and diagnostic statistics on ingested observations. It does not measure or deduce atmospheric chemical reaction pathways or clinical epidemiology.
2. **Sub-Index Principle**: Official regulatory AQI is determined through pollutant-specific sub-index formulas (taking the highest sub-index as overall AQI). Simply averaging concentrations of different pollutants together on one scale is scientifically invalid.
3. **Synthetic Demonstration Data**: The bundled dataset `sample_air_quality.csv` is generated synthetically to model realistic regional variations for software demonstration and testing. It must not be cited as official regulatory telemetry.
4. **Machine Learning Estimates**: The ML module provides empirical statistical approximations using holdout testing. Predictions are mathematical estimates, not sensor ground truth or official meteorological forecasts.
