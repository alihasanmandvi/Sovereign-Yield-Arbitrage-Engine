import streamlit as st
import requests
import datetime
import io
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize

#logic 

@st.cache_data(ttl=3600) # Caches data for 1 hour so it doesn't spam the MUFAP server on every refresh
def get_live_data(anchor_date_str="17062026", anchor_id=25134):
    anchor_date = datetime.datetime.strptime(anchor_date_str, "%d%m%Y").date()
    today_date = datetime.date.today()
    days_elapsed = (today_date - anchor_date).days
    predicted_id = anchor_id + (days_elapsed * 5)
    
    date_str = today_date.strftime("%d%m%Y")
    base_url = "https://mufap.com.pk/Upload/WebDoc/IndustryStatictics/PKRV"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36"}

    target_url = None
    for current_id in range(predicted_id - 2, predicted_id + 3):
        url = f"{base_url}{date_str}{current_id}.csv"
        try:
            if requests.head(url, headers=headers, timeout=5).status_code == 200:
                target_url = url
                break
        except:
            continue
            
    if not target_url:
        # BUG FIX: If falling back to yesterday, we must also revert to yesterday's ID (-5)
        yesterday_date = (today_date - datetime.timedelta(days=1)).strftime("%d%m%Y")
        yesterday_id = predicted_id - 5 
        target_url = f"{base_url}{yesterday_date}{yesterday_id}.csv"

    try:
        response = requests.get(target_url, headers=headers, timeout=15)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        
        tenor_map = {'1M': 0.083, '3M': 0.25, '6M': 0.50, '1Y': 1.0, '2Y': 2.0, '3Y': 3.0, '5Y': 5.0, '7Y': 7.0, '10Y': 10.0, '15Y': 15.0, '20Y': 20.0}
        df['tau'] = df['Tenor'].map(tenor_map)
        df['y'] = pd.to_numeric(df['Mid Rate'], errors='coerce')
        clean_df = df.dropna(subset=['tau', 'y']).sort_values('tau')
        return clean_df['tau'].values, clean_df['y'].values, target_url
    except Exception as e:
        # Expose the error to the UI instead of silently failing
        st.error(f"Network Diagnostics: Failed to fetch {target_url} - Error: {e}")
        return None, None, None

def nss_model(t, b0, b1, b2, b3, tau1, tau2):
    t = np.where(t == 0, 1e-5, t)
    factor1 = (1 - np.exp(-t / tau1)) / (t / tau1)
    factor2 = (1 - np.exp(-t / tau2)) / (t / tau2)
    return b0 + b1 * factor1 + b2 * (factor1 - np.exp(-t / tau1)) + b3 * (factor2 - np.exp(-t / tau2))

def objective_function(params, t_data, y_data):
    if params[4] <= 0 or params[5] <= 0: return np.inf
    return np.sum((y_data - nss_model(t_data, *params)) ** 2)

#dashboard

st.set_page_config(page_title="PKRV Arbitrage Scanner", page_icon="📈", layout="wide")

st.title("PKRV Sovereign Yield Curve Engine")
st.markdown("Live **Nelson-Siegel-Svensson** optimization against daily MUFAP PKRV matrices.")

# Run the pipeline
with st.spinner("Mining live PKRV matrix and calculating non-linear bounds..."):
    t_data, y_data, source_url = get_live_data()

if t_data is not None:
    st.success(f"Data Pipeline Secured. Connected to: {source_url}")
    
    # Calculate Math
    bounds = [(0, 30), (-20, 20), (-20, 20), (-20, 20), (0.1, 15.0), (0.1, 15.0)]
    result = minimize(objective_function, [12.0, -1.0, 0.0, 0.0, 1.0, 2.0], args=(t_data, y_data), method='L-BFGS-B', bounds=bounds)
    
    if result.success:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Yield Curve Architecture")
            
            #chart
            t_cont = np.linspace(0.1, max(t_data) + 2, 1000)
            y_cont = nss_model(t_cont, *result.x)
            
            fig = go.Figure()
            
            # The continuous mathematical baseline
            fig.add_trace(go.Scatter(
                x=t_cont, y=y_cont, 
                mode='lines', 
                name='NSS Mathematical Baseline',
                line=dict(color='blue', width=3),
                hovertemplate="Maturity: %{x:.2f} Yrs<br>Fair Yield: %{y:.2f}%<extra></extra>"
            ))
            
            # The market quotes
            fig.add_trace(go.Scatter(
                x=t_data, y=y_data, 
                mode='markers', 
                name='Live Market Quotes',
                marker=dict(color='red', size=10, line=dict(color='white', width=1)),
                hovertemplate="Maturity: %{x:.2f} Yrs<br>Market Yield: %{y:.2f}%<extra></extra>"
            ))
            
            fig.update_layout(
                xaxis_title="Residual Maturity (Years)",
                yaxis_title="Yield (%)",
                hovermode="x unified",
                legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor="rgba(255,255,255,0.5)"),
                margin=dict(l=0, r=0, t=30, b=0),
                plot_bgcolor="rgba(0,0,0,0.03)"
            )
            
            # Render the interactive chart directly in Streamlit
            st.plotly_chart(fig, width='stretch')
            
        with col2:
            st.subheader("Arbitrage Signals")
            threshold = st.slider("Signal Threshold (bps)", min_value=1.0, max_value=20.0, value=5.0, step=1.0)
            
            y_theo = nss_model(t_data, *result.x)
            df_arb = pd.DataFrame({
                'Tenor (Yrs)': t_data,
                'Yield (%)': y_data,
                'Fair Value (%)': np.round(y_theo, 4),
                'Spread (bps)': np.round((y_data - y_theo) * 100, 2)
            })
            
            conditions = [(df_arb['Spread (bps)'] >= threshold), (df_arb['Spread (bps)'] <= -threshold)]
            df_arb['Action'] = np.select(conditions, ['BUY', 'SELL'], default='HOLD')
            
            # Color code the table
            def color_signals(val):
                color = '#00C851' if val == 'BUY' else '#ff4444' if val == 'SELL' else 'gray'
                return f'color: {color}; font-weight: bold'
            
            st.dataframe(df_arb.style.map(color_signals, subset=['Action']), width='stretch')
            
        st.caption(f"Optimized Parameters: β0={result.x[0]:.4f}, β1={result.x[1]:.4f}, β2={result.x[2]:.4f}, β3={result.x[3]:.4f}, τ1={result.x[4]:.4f}, τ2={result.x[5]:.4f}")
    else:
        st.error("Optimization Engine Failed to Converge.")
else:
    st.error("Data Engine Failed to connect to MUFAP. Matrix missing.")
