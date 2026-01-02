import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Learner Behavior Insights",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────────────────────────────────────
# Beautiful Single Page Theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
    
    .stApp {
        background: #ffffff;
    }
    
    .main .block-container {
        padding: 1.5rem 4rem 3rem 4rem;
        max-width: 1500px;
    }
    
    h1, h2, h3, h4, h5, h6, p, span, div {
        font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    .dashboard-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 2.5rem 3rem;
        border-radius: 24px;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .dashboard-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 60%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
    }
    
    .dashboard-header h1 {
        color: white !important;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        margin: 0 !important;
        text-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .dashboard-header p {
        color: rgba(255,255,255,0.9) !important;
        font-size: 1.05rem !important;
        margin: 0.75rem 0 0 0 !important;
    }
    
    .insight-section {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 20px;
        padding: 1.5rem 2rem 2rem 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.04);
    }
    
    .insight-section:hover {
        box-shadow: 0 8px 30px rgba(0,0,0,0.08);
        border-color: #cbd5e1;
    }
    
    section[data-testid="stSidebar"] { display: none !important; }
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }
    
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #f1f5f9; border-radius: 4px; }
    ::-webkit-scrollbar-thumb { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────
def style_plot(fig):
    fig.update_layout(
        font_family="DM Sans",
        title_font_size=16,
        title_font_color="#1e293b",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=dict(gridcolor='rgba(0,0,0,0.05)', tickfont=dict(size=11, color='#64748b')),
        yaxis=dict(gridcolor='rgba(0,0,0,0.05)', tickfont=dict(size=11, color='#64748b'))
    )
    return fig

@st.cache_data(show_spinner=False, ttl=3600)
def load_data():
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    csv_path = script_dir / "base_data_30.csv"
    return pd.read_csv(csv_path)

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="dashboard-header">
    <h1>🎓 Learner Behavior Insights</h1>
    <p>Cohort: Users active on <b>29th December 2025</b> • Features calculated over their <b>last 30 days</b> of activity</p>
</div>
""", unsafe_allow_html=True)

# Load Data
with st.spinner("Loading dataset..."):
    df = load_data()

# ─────────────────────────────────────────────────────────────────────────────
# Cohort Overview
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 📈 Cohort Overview")

total_users = len(df)
avg_hours = df['learning_hours_d30'].mean()
avg_active_days = df['active_days_d30'].mean()
avg_session_dur = df['avg_session_dur_min_d30'].dropna().mean()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div style="background: white; padding: 1.5rem; border-radius: 16px; text-align: center;
                border: 2px solid #e2e8f0; box-shadow: 0 4px 15px rgba(0,0,0,0.05);">
        <div style="font-size: 2.5rem; font-weight: 700; color: #667eea;">{total_users:,}</div>
        <div style="color: #64748b; font-size: 0.9rem; margin-top: 0.5rem;">Total Learners</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div style="background: white; padding: 1.5rem; border-radius: 16px; text-align: center;
                border: 2px solid #e2e8f0; box-shadow: 0 4px 15px rgba(0,0,0,0.05);">
        <div style="font-size: 2.5rem; font-weight: 700; color: #10b981;">{avg_hours:.1f}</div>
        <div style="color: #64748b; font-size: 0.9rem; margin-top: 0.5rem;">Avg Hours/Learner</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div style="background: white; padding: 1.5rem; border-radius: 16px; text-align: center;
                border: 2px solid #e2e8f0; box-shadow: 0 4px 15px rgba(0,0,0,0.05);">
        <div style="font-size: 2.5rem; font-weight: 700; color: #f59e0b;">{avg_active_days:.1f}</div>
        <div style="color: #64748b; font-size: 0.9rem; margin-top: 0.5rem;">Avg Active Days</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div style="background: white; padding: 1.5rem; border-radius: 16px; text-align: center;
                border: 2px solid #e2e8f0; box-shadow: 0 4px 15px rgba(0,0,0,0.05);">
        <div style="font-size: 2.5rem; font-weight: 700; color: #8b5cf6;">{avg_session_dur:.0f} min</div>
        <div style="color: #64748b; font-size: 0.9rem; margin-top: 0.5rem;">Avg Session Duration</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# KEY INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="margin: 1rem 0 2rem 0;">
    <h2 style="color: #1e293b; margin: 0; font-size: 1.5rem; font-weight: 600;">📊 Key Insights</h2>
    <p style="color: #64748b; margin: 0.5rem 0 0 0; font-size: 0.95rem;">Deep dive into learner behavior patterns and engagement metrics</p>
</div>
""", unsafe_allow_html=True)

sample_df = df.sample(min(10000, len(df)), random_state=42)

# --- 1. CONTENT MIX ---
st.markdown("""<div class="insight-section">
<h4 style="color: #1e293b; margin: 0 0 0.5rem 0; display: flex; align-items: center; gap: 0.5rem;">
    <span style="background: #10b981; color: white; padding: 0.35rem 0.75rem; border-radius: 8px; font-size: 0.9rem; font-weight: 600;">1</span>
    Content Mix: What are learners consuming?
</h4>
<p style="color: #64748b; margin: 0 0 1rem 0; font-size: 0.9rem;">Content difficulty levels and assessment vs hands-on learning preferences</p>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    diff_cols = ['share_beginner_rows_d30', 'share_intermediate_rows_d30', 'share_advanced_rows_d30']
    diff_labels = ['🟢 Beginner', '🟡 Intermediate', '🔴 Advanced']
    diff_colors = ['#22c55e', '#eab308', '#ef4444']
    avg_diff = [df[col].mean() for col in diff_cols]
    
    fig = go.Figure(data=[go.Pie(
        labels=diff_labels, values=avg_diff, hole=0.55,
        marker=dict(colors=diff_colors, line=dict(color='white', width=2)),
        textinfo='label+percent', textfont=dict(size=13),
        hovertemplate='<b>%{label}</b><br>Share: %{percent}<extra></extra>'
    )])
    fig = style_plot(fig)
    fig.update_layout(title='Content Difficulty Distribution', showlegend=False, height=400,
                      annotations=[dict(text='Difficulty<br>Level', x=0.5, y=0.5, font_size=13, showarrow=False)])
    st.plotly_chart(fig, use_container_width=True)

with col2:
    graded_labs_df = sample_df[(sample_df['share_graded_lh_d30'].notna()) & (sample_df['share_labs_lh_d30'].notna())].copy()
    fig = px.scatter(graded_labs_df, x='share_graded_lh_d30', y='share_labs_lh_d30', color='learning_hours_d30',
                     opacity=0.5, title='Graded Content vs Labs Focus', color_continuous_scale='Tealgrn',
                     labels={'share_graded_lh_d30': 'Graded Content Share', 'share_labs_lh_d30': 'Labs Share', 'learning_hours_d30': 'Learning Hrs'})
    fig = style_plot(fig)
    fig.update_traces(marker=dict(size=7), hovertemplate='<b>Graded:</b> %{x:.0%}<br><b>Labs:</b> %{y:.0%}<br><b>Hours:</b> %{marker.color:.1f}<extra></extra>')
    fig.update_layout(height=400, xaxis=dict(tickformat='.0%'), yaxis=dict(tickformat='.0%'))
    fig.add_annotation(x=0.8, y=0.1, text="📝 Assessment Focused", showarrow=False, font=dict(size=11, color="#0d9488"), bgcolor="rgba(13,148,136,0.1)")
    fig.add_annotation(x=0.1, y=0.6, text="🧪 Hands-on Learners", showarrow=False, font=dict(size=11, color="#7c3aed"), bgcolor="rgba(124,58,237,0.1)")
    st.plotly_chart(fig, use_container_width=True)

avg_graded_share = df['share_graded_lh_d30'].mean() * 100
avg_labs_share = df['share_labs_lh_d30'].mean() * 100
high_graded = (df['share_graded_lh_d30'] > 0.5).sum() / len(df) * 100
high_labs = (df['share_labs_lh_d30'] > 0.3).sum() / len(df) * 100
st.markdown(f"""
<div style="background: #f0fdfa; border: 1px solid #99f6e4; padding: 1rem 1.25rem; border-radius: 12px; margin: 1rem 0 0 0;">
    <span style="color: #0f766e; font-weight: 600;">📌 Key Takeaway:</span>
    <span style="color: #115e59;">On average, learners spend <b>{avg_graded_share:.0f}%</b> of their time on graded assessments and <b>{avg_labs_share:.0f}%</b> on hands-on labs. <b>{high_graded:.1f}%</b> are assessment-focused (50%+ graded), while <b>{high_labs:.1f}%</b> are lab-heavy (30%+ labs).</span>
</div>
</div>
""", unsafe_allow_html=True)

# --- 2. EXPLORATION VS FOCUS ---
st.markdown("""<div class="insight-section">
<h4 style="color: #1e293b; margin: 0 0 0.5rem 0; display: flex; align-items: center; gap: 0.5rem;">
    <span style="background: #f59e0b; color: white; padding: 0.35rem 0.75rem; border-radius: 8px; font-size: 0.9rem; font-weight: 600;">2</span>
    Exploration vs Focus: Are learners diving deep or sampling broadly?
</h4>
<p style="color: #64748b; margin: 0 0 1rem 0; font-size: 0.9rem;">Course switching behavior vs depth of engagement per course</p>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    explore_df = sample_df[(sample_df['items_per_course_d30'].notna()) & (sample_df['items_per_course_d30'] < sample_df['items_per_course_d30'].quantile(0.95))].copy()
    fig = px.scatter(explore_df, x='switch_rate_d30', y='items_per_course_d30', color='distinct_courses_d30', opacity=0.5,
                     title='Exploration vs Focus', color_continuous_scale='Oranges',
                     labels={'switch_rate_d30': 'Switch Rate', 'items_per_course_d30': 'Items per Course', 'distinct_courses_d30': 'Courses'})
    fig = style_plot(fig)
    fig.update_traces(marker=dict(size=7), hovertemplate='<b>Switch Rate:</b> %{x:.0%}<br><b>Items/Course:</b> %{y:.0f}<br><b>Courses:</b> %{marker.color:.0f}<extra></extra>')
    fig.update_layout(height=400, xaxis=dict(tickformat='.0%'))
    fig.add_annotation(x=0.02, y=explore_df['items_per_course_d30'].quantile(0.85), text="🎯 Deep Divers", showarrow=False, font=dict(size=11, color="#ea580c"), bgcolor="rgba(234,88,12,0.1)")
    fig.add_annotation(x=0.2, y=explore_df['items_per_course_d30'].quantile(0.2), text="🔍 Explorers", showarrow=False, font=dict(size=11, color="#64748b"), bgcolor="rgba(100,116,139,0.1)")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = go.Figure(data=[go.Histogram(x=df[df['distinct_courses_d30'] <= 30]['distinct_courses_d30'], nbinsx=30, marker_color='#f59e0b', marker_line_width=0.5, marker_line_color='white', opacity=0.9, hovertemplate='<b>Courses:</b> %{x:.0f}<br><b>Learners:</b> %{y:,}<extra></extra>')])
    fig = style_plot(fig)
    fig.update_layout(title='How Many Courses Do Learners Touch?', xaxis_title='Distinct Courses (d30)', yaxis_title='Number of Learners', height=400, bargap=0.05)
    st.plotly_chart(fig, use_container_width=True)

avg_courses = df['distinct_courses_d30'].mean()
single_course = (df['distinct_courses_d30'] == 1).sum() / len(df) * 100
multi_course = (df['distinct_courses_d30'] >= 5).sum() / len(df) * 100
st.markdown(f"""
<div style="background: #fef3c7; border: 1px solid #fcd34d; padding: 1rem 1.25rem; border-radius: 12px; margin: 1rem 0 0 0;">
    <span style="color: #b45309; font-weight: 600;">📌 Key Takeaway:</span>
    <span style="color: #92400e;">Learners interact with an average of <b>{avg_courses:.1f} courses</b> over 30 days. Notably, <b>{single_course:.1f}%</b> demonstrate focused learning on a single course, while <b>{multi_course:.1f}%</b> explore 5+ courses.</span>
</div>
</div>
""", unsafe_allow_html=True)

# --- 3. ENGAGEMENT INTENSITY ---
st.markdown("""<div class="insight-section">
<h4 style="color: #1e293b; margin: 0 0 0.5rem 0; display: flex; align-items: center; gap: 0.5rem;">
    <span style="background: #0ea5e9; color: white; padding: 0.35rem 0.75rem; border-radius: 8px; font-size: 0.9rem; font-weight: 600;">3</span>
    Engagement Intensity: Where are learners concentrated?
</h4>
<p style="color: #64748b; margin: 0 0 1rem 0; font-size: 0.9rem;">User density by active days and learning hours</p>
""", unsafe_allow_html=True)

df_heat = df.copy()
df_heat['hours_bin'] = pd.cut(df_heat['learning_hours_d30'], bins=[0, 1, 3, 5, 10, 20, 50, 100], labels=['0-1h', '1-3h', '3-5h', '5-10h', '10-20h', '20-50h', '50h+'])
df_heat['days_bin'] = pd.cut(df_heat['active_days_d30'], bins=[0, 3, 7, 14, 21, 30], labels=['1-3 days', '4-7 days', '8-14 days', '15-21 days', '22-30 days'])
heatmap_data = df_heat.groupby(['days_bin', 'hours_bin']).size().unstack(fill_value=0)

fig = px.imshow(heatmap_data, title='Engagement Intensity Heatmap', labels={'x': 'Learning Hours', 'y': 'Active Days', 'color': 'Learners'}, color_continuous_scale='Blues', aspect='auto', text_auto=True)
fig = style_plot(fig)
fig.update_layout(height=400)
fig.update_traces(texttemplate='%{z:,.0f}', textfont=dict(size=10))
st.plotly_chart(fig, use_container_width=True)

low_engagement = ((df['learning_hours_d30'] <= 3) & (df['active_days_d30'] <= 7)).sum() / len(df) * 100
high_engagement = ((df['learning_hours_d30'] > 10) & (df['active_days_d30'] > 14)).sum() / len(df) * 100
st.markdown(f"""
<div style="background: #e0f2fe; border: 1px solid #7dd3fc; padding: 1rem 1.25rem; border-radius: 12px; margin: 1rem 0 0 0;">
    <span style="color: #0369a1; font-weight: 600;">📌 Key Takeaway:</span>
    <span style="color: #075985;">The cohort shows varied engagement: <b>{high_engagement:.1f}%</b> are highly engaged (10+ hrs, 14+ days), while <b>{low_engagement:.1f}%</b> demonstrate lower engagement. This suggests opportunities for targeted re-engagement.</span>
</div>
</div>
""", unsafe_allow_html=True)

# --- 4. LEARNER SEGMENTS ---
st.markdown("""<div class="insight-section">
<h4 style="color: #1e293b; margin: 0 0 0.5rem 0; display: flex; align-items: center; gap: 0.5rem;">
    <span style="background: #1e293b; color: white; padding: 0.35rem 0.75rem; border-radius: 8px; font-size: 0.9rem; font-weight: 600;">4</span>
    Segment Summary: Who are our learners?
</h4>
<p style="color: #64748b; margin: 0 0 1rem 0; font-size: 0.9rem;">Learner segments based on engagement intensity and frequency</p>
""", unsafe_allow_html=True)

df['segment'] = 'Casual'
df.loc[(df['learning_hours_d30'] > 10) & (df['active_days_d30'] > 15), 'segment'] = 'Power Learner'
df.loc[(df['learning_hours_d30'] > 5) & (df['active_days_d30'] <= 5), 'segment'] = 'Intensive Burster'
df.loc[(df['learning_hours_d30'] <= 5) & (df['active_days_d30'] > 15), 'segment'] = 'Consistent Nibbler'
df.loc[(df['learning_hours_d30'] > 5) & (df['learning_hours_d30'] <= 10) & (df['active_days_d30'] > 5) & (df['active_days_d30'] <= 15), 'segment'] = 'Moderate'

segment_counts = df['segment'].value_counts()
segment_colors = {'Power Learner': '#8b5cf6', 'Intensive Burster': '#f59e0b', 'Consistent Nibbler': '#0ea5e9', 'Moderate': '#10b981', 'Casual': '#94a3b8'}

col1, col2 = st.columns([1, 1])

with col1:
    fig = go.Figure(data=[go.Pie(labels=segment_counts.index, values=segment_counts.values, hole=0.5,
        marker=dict(colors=[segment_colors.get(s, '#94a3b8') for s in segment_counts.index], line=dict(color='white', width=2)),
        textinfo='label+percent', textfont=dict(size=12), hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>')])
    fig = style_plot(fig)
    fig.update_layout(title='Learner Segments', showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    segment_descriptions = {
        'Power Learner': ('🔥', '>10 hrs & >15 days', 'High volume, high frequency'),
        'Intensive Burster': ('⚡', '>5 hrs & ≤5 days', 'High intensity, low frequency'),
        'Consistent Nibbler': ('📅', '≤5 hrs & >15 days', 'Low intensity, high frequency'),
        'Moderate': ('📈', '5-10 hrs & 6-15 days', 'Balanced engagement'),
        'Casual': ('👋', 'Other combinations', 'Light or irregular patterns')
    }
    for seg in segment_counts.index:
        count = segment_counts[seg]
        pct = count / len(df) * 100
        emoji, criteria, desc = segment_descriptions.get(seg, ('•', '', ''))
        color = segment_colors.get(seg, '#94a3b8')
        st.markdown(f"""
        <div style="background: #f8fafc; padding: 1rem 1.25rem; border-radius: 12px; margin-bottom: 0.75rem; border-left: 4px solid {color};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-weight: 600; color: #1e293b;">{emoji} {seg}</span>
                <span style="font-weight: 700; color: {color};">{count:,} ({pct:.1f}%)</span>
            </div>
            <div style="color: #64748b; font-size: 0.85rem; margin-top: 0.25rem;">{criteria} — {desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# --- 5. SESSION BEHAVIOR ---
st.markdown("""<div class="insight-section">
<h4 style="color: #1e293b; margin: 0 0 0.5rem 0; display: flex; align-items: center; gap: 0.5rem;">
    <span style="background: #667eea; color: white; padding: 0.35rem 0.75rem; border-radius: 8px; font-size: 0.9rem; font-weight: 600;">5</span>
    Session Behavior: When and how long do learners study?
</h4>
<p style="color: #64748b; margin: 0 0 1rem 0; font-size: 0.9rem;">Understanding learning session patterns across time of day</p>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    time_cols = ['share_morning_sessions_d30', 'share_afternoon_sessions_d30', 'share_evening_sessions_d30', 'share_night_sessions_d30']
    time_labels = ['🌅 Morning', '☀️ Afternoon', '🌆 Evening', '🌙 Night']
    time_colors = ['#fbbf24', '#f97316', '#8b5cf6', '#3b82f6']
    avg_time = [df[col].mean() for col in time_cols]
    fig = go.Figure(data=[go.Pie(labels=time_labels, values=avg_time, hole=0.55, marker=dict(colors=time_colors, line=dict(color='white', width=2)),
        textinfo='label+percent', textfont=dict(size=13), hovertemplate='<b>%{label}</b><br>Share: %{percent}<extra></extra>')])
    fig = style_plot(fig)
    fig.update_layout(title='When Do Learners Study?', showlegend=False, height=380, annotations=[dict(text='Time of<br>Day', x=0.5, y=0.5, font_size=14, showarrow=False)])
    st.plotly_chart(fig, use_container_width=True)

with col2:
    df_dur = df[df['avg_session_dur_min_d30'].notna() & (df['avg_session_dur_min_d30'] > 0) & (df['avg_session_dur_min_d30'] < df['avg_session_dur_min_d30'].quantile(0.95))].copy()
    fig = go.Figure(data=[go.Histogram(x=df_dur['avg_session_dur_min_d30'], nbinsx=40, marker_color='#667eea', marker_line_width=0.5, marker_line_color='white', opacity=0.9, hovertemplate='<b>Duration:</b> %{x:.0f} min<br><b>Learners:</b> %{y:,}<extra></extra>')])
    fig = style_plot(fig)
    fig.update_layout(title='Session Duration Distribution', xaxis_title='Avg Session Duration (minutes)', yaxis_title='Number of Learners', height=380, bargap=0.05)
    st.plotly_chart(fig, use_container_width=True)

avg_dur = df['avg_session_dur_min_d30'].median()
total_sessions = df['sessions_d30'].sum()
morning_sessions = (df['share_morning_sessions_d30'] * df['sessions_d30']).sum()
afternoon_sessions = (df['share_afternoon_sessions_d30'] * df['sessions_d30']).sum()
night_sessions = (df['share_night_sessions_d30'] * df['sessions_d30']).sum()
morning_pct = (morning_sessions / total_sessions) * 100 if total_sessions > 0 else 0
afternoon_pct = (afternoon_sessions / total_sessions) * 100 if total_sessions > 0 else 0
night_pct = (night_sessions / total_sessions) * 100 if total_sessions > 0 else 0
peak_time = "morning" if morning_pct >= max(afternoon_pct, night_pct) else ("afternoon" if afternoon_pct >= night_pct else "night")

st.markdown(f"""
<div style="background: #f0fdf4; border: 1px solid #bbf7d0; padding: 1rem 1.25rem; border-radius: 12px; margin: 1rem 0 0 0;">
    <span style="color: #15803d; font-weight: 600;">📌 Key Takeaway:</span>
    <span style="color: #166534;">The typical learning session lasts <b>{avg_dur:.0f} minutes</b>. Session distribution: <b>{morning_pct:.0f}%</b> morning, <b>{afternoon_pct:.0f}%</b> afternoon, <b>{night_pct:.0f}%</b> night — peak activity occurs during <b>{peak_time}</b> hours.</span>
</div>
</div>
""", unsafe_allow_html=True)

# --- 6. NIGHT OWLS VS EARLY BIRDS ---
st.markdown("""<div class="insight-section">
<h4 style="color: #1e293b; margin: 0 0 0.5rem 0; display: flex; align-items: center; gap: 0.5rem;">
    <span style="background: #8b5cf6; color: white; padding: 0.35rem 0.75rem; border-radius: 8px; font-size: 0.9rem; font-weight: 600;">6</span>
    🌙 Night Owls vs 🌅 Early Birds
</h4>
<p style="color: #64748b; margin: 0 0 1rem 0; font-size: 0.9rem;">Learning time preferences reveal distinct learner personas</p>
""", unsafe_allow_html=True)

fig = px.scatter(sample_df, x='share_morning_sessions_d30', y='share_night_sessions_d30', color='learning_hours_d30', size='sessions_d30', opacity=0.5,
    title='Learning Time Preferences', color_continuous_scale='Sunsetdark',
    labels={'share_morning_sessions_d30': 'Morning Session Share', 'share_night_sessions_d30': 'Night Session Share', 'learning_hours_d30': 'Learning Hrs'})
fig = style_plot(fig)
fig.update_traces(marker=dict(sizemin=4, sizeref=3), hovertemplate='<b>Morning:</b> %{x:.0%}<br><b>Night:</b> %{y:.0%}<br><b>Hours:</b> %{marker.color:.1f}<extra></extra>')
fig.update_layout(height=420, xaxis=dict(tickformat='.0%'), yaxis=dict(tickformat='.0%'))
fig.add_annotation(x=0.7, y=0.15, text="🌅 Early Birds", showarrow=False, font=dict(size=13, color="#f59e0b"), bgcolor="rgba(251,191,36,0.15)")
fig.add_annotation(x=0.15, y=0.7, text="🌙 Night Owls", showarrow=False, font=dict(size=13, color="#6366f1"), bgcolor="rgba(99,102,241,0.15)")
fig.add_annotation(x=0.5, y=0.5, text="⚖️ Balanced", showarrow=False, font=dict(size=11, color="#64748b"), bgcolor="rgba(100,116,139,0.1)")
st.plotly_chart(fig, use_container_width=True)

early_birds = (df['share_morning_sessions_d30'] > 0.5).sum() / len(df) * 100
night_owls = (df['share_night_sessions_d30'] > 0.5).sum() / len(df) * 100
afternoon_heavy = (df['share_afternoon_sessions_d30'] > 0.5).sum() / len(df) * 100
no_dominant = 100 - early_birds - night_owls - afternoon_heavy

st.markdown(f"""
<div style="background: #f5f3ff; border: 1px solid #ddd6fe; padding: 1rem 1.25rem; border-radius: 12px; margin: 1rem 0 0 0;">
    <span style="color: #6d28d9; font-weight: 600;">📌 Key Takeaway:</span>
    <span style="color: #5b21b6;">Time preference breakdown: <b>{early_birds:.1f}%</b> are Early Birds (50%+ morning), <b>{afternoon_heavy:.1f}%</b> favor afternoons, <b>{night_owls:.1f}%</b> are Night Owls, and <b>{no_dominant:.0f}%</b> have no dominant time preference.</span>
</div>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; padding: 3rem 2rem; color: #94a3b8; font-size: 0.9rem; margin-top: 3rem;">
    <div style="display: inline-block; padding: 1rem 2rem; background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); border-radius: 12px;">
        📊 <b>Learner Behavior Insights Dashboard</b> • Built with Streamlit & Plotly
    </div>
</div>
""", unsafe_allow_html=True)
