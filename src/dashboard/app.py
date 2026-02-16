"""Streamlit dashboard for Moneyball system."""

import sys
from pathlib import Path

# Add project root to path for src imports
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from pathlib import Path
import json

from src.config import get_config
from src.models.predict import InferencePipeline
from src.models.backtest import run_backtest


@st.cache_resource
def load_pipeline():
    """Load inference pipeline (cached)."""
    return InferencePipeline()


@st.cache_data(ttl=3600)
def load_recommendations(_pipeline, top_n, asset_types):
    """Load recommendations (cached for 1 hour)."""
    return _pipeline.generate_recommendations(top_n=top_n, asset_types=asset_types)


def main():
    st.set_page_config(page_title="Moneyball", layout="wide")
    st.title("Moneyball - AI Investment Discovery")
    st.caption("Finding undervalued assets before they break out")
    
    config = get_config()
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["Scout Dashboard", "Backtest Simulator", "Audit Logs"])
    
    with tab1:
        render_scout_dashboard(config)
    
    with tab2:
        render_backtest_simulator(config)
    
    with tab3:
        render_audit_logs(config)


def render_scout_dashboard(config):
    """Scout Dashboard: ranked asset recommendations."""
    st.header("Top Investment Opportunities")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        top_n = st.slider("Number of Assets", 5, 50, 20)
    with col2:
        asset_types = st.multiselect("Asset Universe", ["crypto", "stock"], default=["crypto"])
    with col3:
        if st.button("Refresh", type="primary"):
            st.cache_data.clear()
            st.rerun()
    
    try:
        # Load pipeline and generate recommendations
        with st.spinner("Generating recommendations..."):
            pipeline = load_pipeline()
            recommendations = load_recommendations(pipeline, top_n, asset_types)
        
        if recommendations.empty:
            st.warning("No recommendations found. Ensure features and model are available.")
            return
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Opportunities", len(recommendations))
        with col2:
            avg_score = recommendations['breakout_score'].mean()
            st.metric("Avg Breakout Score", f"{avg_score:.1%}")
        with col3:
            high_conviction = (recommendations['breakout_score'] > 0.7).sum()
            st.metric("High Conviction", high_conviction)
        with col4:
            regime = recommendations['regime'].iloc[0] if not recommendations.empty else "Unknown"
            st.metric("Market Regime", regime)
        
        # Score distribution chart
        fig = px.histogram(recommendations, x='breakout_score', nbins=20,
                          title="Breakout Score Distribution",
                          labels={'breakout_score': 'Breakout Score'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Display recommendations
        st.subheader("Ranked Opportunities")
        
        for idx, rec in recommendations.iterrows():
            score_color = "[High]" if rec['breakout_score'] > 0.7 else "[Med]" if rec['breakout_score'] > 0.5 else "[Low]"
            
            with st.expander(f"{score_color} #{idx+1} | {rec['asset'].upper()} | Score: {rec['breakout_score']:.1%} | ${rec['current_price']:.2f}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(rec['memo'])
                
                with col2:
                    st.metric("Asset Type", rec['asset_type'].upper())
                    st.metric("Current Price", f"${rec['current_price']:.2f}")
                    st.metric("Breakout Score", f"{rec['breakout_score']:.1%}")
                    st.metric("Regime", rec['regime'])
        
        # Download button
        csv = recommendations.to_csv(index=False)
        st.download_button(
            label="Download Recommendations (CSV)",
            data=csv,
            file_name=f"moneyball_recommendations_{datetime.now():%Y%m%d}.csv",
            mime="text/csv"
        )
    
    except FileNotFoundError as e:
        st.error(f"{e}")
        st.info("Run the feature pipeline and train the model first:\n```bash\npython -m src.features.pipeline\npython -m src.models.train\n```")
    except Exception as e:
        st.error(f"Error: {e}")
        st.exception(e)


def render_backtest_simulator(config):
    """Backtest Simulator: historical performance analysis."""
    st.header("Backtest Simulator")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        start_date = st.date_input("Start Date", value=datetime.today() - timedelta(days=180))
    with col2:
        end_date = st.date_input("End Date", value=datetime.today())
    with col3:
        top_n = st.number_input("Portfolio Size", 5, 50, 10)
    with col4:
        rebalance_days = st.number_input("Rebalance (days)", 7, 90, 30)
    
    if st.button("Run Backtest", type="primary"):
        with st.spinner("Running backtest..."):
            try:
                results = run_backtest(
                    start_date=pd.Timestamp(start_date),
                    end_date=pd.Timestamp(end_date),
                    top_n=top_n,
                    rebalance_days=rebalance_days
                )
                
                # Display results
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Return", f"{results['total_return']:.1%}")
                with col2:
                    st.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
                with col3:
                    st.metric("Max Drawdown", f"{results['max_drawdown']:.1%}")
                with col4:
                    st.metric("Win Rate", f"{results['win_rate']:.1%}")
                
                # Portfolio value chart
                if 'portfolio_values' in results:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=results['dates'],
                        y=results['portfolio_values'],
                        mode='lines',
                        name='Portfolio',
                        line=dict(color='green', width=2)
                    ))
                    fig.update_layout(
                        title="Portfolio Value Over Time",
                        xaxis_title="Date",
                        yaxis_title="Portfolio Value ($)",
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Drawdown chart
                if 'drawdowns' in results:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=results['dates'],
                        y=results['drawdowns'],
                        mode='lines',
                        name='Drawdown',
                        fill='tozeroy',
                        line=dict(color='red', width=1)
                    ))
                    fig.update_layout(
                        title="Drawdown Over Time",
                        xaxis_title="Date",
                        yaxis_title="Drawdown (%)",
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                st.success("Backtest completed successfully!")
                
            except Exception as e:
                st.error(f"Backtest failed: {e}")
                st.info("Ensure you have sufficient historical data and trained models.")
    else:
        st.info("Configure parameters and click 'Run Backtest' to simulate historical performance.")


def render_audit_logs(config):
    """Audit Logs: searchable event log."""
    st.header("Audit Logs")
    
    log_dir = config.log_dir
    if not log_dir.exists():
        st.warning("No audit logs found.")
        return
    
    log_files = sorted(list(log_dir.glob("audit_*.jsonl")), reverse=True)
    if not log_files:
        st.warning("No audit log files found.")
        return
    
    # File selector
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_file = st.selectbox("Select Log File", [f.name for f in log_files])
    with col2:
        max_lines = st.number_input("Max Lines", 10, 1000, 100)
    
    if selected_file:
        # Load and parse logs
        logs = []
        with open(log_dir / selected_file) as f:
            for line in f:
                try:
                    logs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        
        if not logs:
            st.warning("No valid log entries found.")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(logs)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            event_types = st.multiselect("Event Type", df['event_type'].unique(), default=df['event_type'].unique())
        with col2:
            agents = st.multiselect("Agent", df['agent'].unique(), default=df['agent'].unique())
        with col3:
            search_asset = st.text_input("Search Asset", "")
        
        # Apply filters
        filtered = df[df['event_type'].isin(event_types) & df['agent'].isin(agents)]
        if search_asset:
            filtered = filtered[filtered['asset'].str.contains(search_asset, case=False, na=False)]
        
        filtered = filtered.tail(max_lines)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Events", len(filtered))
        with col2:
            avg_latency = filtered['latency_ms'].mean() if 'latency_ms' in filtered else 0
            st.metric("Avg Latency", f"{avg_latency:.1f}ms")
        with col3:
            unique_assets = filtered['asset'].nunique()
            st.metric("Unique Assets", unique_assets)
        with col4:
            time_range = (filtered['timestamp'].max() - filtered['timestamp'].min()).total_seconds() / 3600
            st.metric("Time Range", f"{time_range:.1f}h")
        
        # Event timeline
        event_counts = filtered.groupby(['timestamp', 'event_type']).size().reset_index(name='count')
        fig = px.line(event_counts, x='timestamp', y='count', color='event_type',
                     title="Event Timeline")
        st.plotly_chart(fig, use_container_width=True)
        
        # Latency distribution
        if 'latency_ms' in filtered.columns:
            fig = px.box(filtered, x='agent', y='latency_ms', title="Latency by Agent")
            st.plotly_chart(fig, use_container_width=True)
        
        # Log table
        st.subheader("Event Log")
        display_cols = ['timestamp', 'event_type', 'agent', 'asset', 'latency_ms']
        display_cols = [col for col in display_cols if col in filtered.columns]
        st.dataframe(filtered[display_cols].sort_values('timestamp', ascending=False), use_container_width=True)
        
        # Detailed view
        with st.expander("View Raw Log Details"):
            selected_idx = st.number_input("Event Index", 0, len(filtered)-1, 0)
            if selected_idx < len(filtered):
                event = filtered.iloc[selected_idx]
                st.json(event.to_dict())


if __name__ == "__main__":
    main()
