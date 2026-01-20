
import streamlit as st
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go


st.title("Call Options Price Calculator")

# Variables
st.subheader('Variables')

col1, col2 = st.columns(2)

with col1:
    S = st.number_input('Stock Price (S)', min_value=0.0, max_value=10000.0, value=50.0, step=0.01)
    K = st.number_input('Strike Price (K)', min_value=0.0, max_value=10000.0, value=50.0, step=0.01)
    T = st.number_input('Time to Maturity (years)', min_value=0.0, max_value=10.0, value=1.0, step=0.01)

with col2:
    R = st.number_input('Risk-Free Interest Rate', min_value=0.0, max_value=1.0, value=0.04, step=0.01, format="%.2f")
    sigma = st.number_input('Standard Deviation', min_value=0.0, max_value=2.0, value=0.4, step=0.01, format="%.2f")

# Calculations
if st.button('Calculate'):
    d1 = (np.log(S / K) + (R + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = S * norm.cdf(d1) - K * np.exp(-R * T) * norm.cdf(d2)

    delta = np.exp(-R * T) * norm.cdf(d1)
    gamma = np.exp(-R * T) * norm.cdf(d1) / (sigma * price * np.sqrt(T))
    vega = np.exp(-R * T) * norm.cdf(d1) * price * np.sqrt(T)
    theta = -1 * np.exp(-R * T) * norm.cdf(d1) * (sigma / 2 * np.sqrt(T)) + R * np.exp(-R * T) * norm.cdf(d1) - R * K * np.exp(-R * T) * norm.cdf(d2)
    rho = K * T * np.exp(-R * T) * norm.cdf(d2)

    st.success('Calculations finished.')

    # Two columns. Results on the left, graph on right
    left_results, right_plot = st.columns([1, 2])

    with left_results:
        st.subheader("Results")
        st.metric('Call Option Price', f'{price:.2f}')
        st.metric('Delta', f'{delta:.2f}')
        st.metric('Gamma', f'{gamma:.2f}')
        st.metric('Vega', f'{vega:.2f}')
        st.metric('Theta', f'{theta:.2f}')
        st.metric('Rho', f'{rho:.2f}')

    with right_plot:
        st.subheader("Call Price Surface")

        try:
            # Create grid for surface plot
            strike_range = np.linspace(S * 0.5, S * 1.5, 30)
            time_range = np.linspace(0.1, 3.0, 30)
            K_grid, T_grid = np.meshgrid(strike_range, time_range)

            # Calculate prices for each combination
            price_grid = np.zeros_like(K_grid)
            for i in range(len(time_range)):
                for j in range(len(strike_range)):
                    d1_temp = (np.log(S / K_grid[i, j]) + (R + sigma ** 2 / 2) * T_grid[i, j]) / (
                                sigma * np.sqrt(T_grid[i, j]))
                    d2_temp = d1_temp - sigma * np.sqrt(T_grid[i, j])
                    price_grid[i, j] = S * norm.cdf(d1_temp) - K_grid[i, j] * np.exp(-R * T_grid[i, j]) * norm.cdf(
                        d2_temp)

            # Create 3D surface plot
            fig = go.Figure(data=[go.Surface(
                x=K_grid,
                y=T_grid,
                z=price_grid,
                colorscale='Viridis'
            )])

            fig.update_layout(
                scene=dict(
                    xaxis_title='Strike Price (K)',
                    yaxis_title='Time to Maturity (T)',
                    zaxis_title='Call Price',
                ),
                height=500,
                margin=dict(l=0, r=0, t=0, b=0)
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error creating plot: {e}")
