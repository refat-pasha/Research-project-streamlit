import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# Enhanced Dashboard Data
DASHBOARD_DATA = {
    "github_stats": {
        "total_repos": 52,
        "total_commits": 1247,
        "total_stars": 486,
        "total_forks": 142,
        "languages": {
            "Python": 35,
            "JavaScript": 28,
            "TypeScript": 15,
            "Java": 12,
            "C++": 8,
            "Other": 2
        },
        "commits_per_month": [
            {"month": "Jan", "commits": 89},
            {"month": "Feb", "commits": 102},
            {"month": "Mar", "commits": 156},
            {"month": "Apr", "commits": 134},
            {"month": "May", "commits": 178},
            {"month": "Jun", "commits": 165},
            {"month": "Jul", "commits": 143},
            {"month": "Aug", "commits": 167},
            {"month": "Sep", "commits": 123},
            {"month": "Oct", "commits": 98},
            {"month": "Nov", "commits": 92}
        ]
    },
    "project_stats": {
        "total_projects": 15,
        "completed": 12,
        "in_progress": 2,
        "planning": 1,
        "project_categories": {
            "Web Development": 6,
            "Mobile Apps": 3,
            "AI/ML": 4,
            "Data Science": 2
        },
        "tech_usage": {
            "React": 8,
            "Node.js": 6,
            "Python": 7,
            "Flutter": 3,
            "TensorFlow": 4,
            "MongoDB": 5,
            "PostgreSQL": 4
        }
    },
    "research_metrics": {
        "papers_published": 8,
        "citations": 245,
        "h_index": 6,
        "conferences": 5,
        "research_areas": {
            "Machine Learning": 4,
            "Computer Vision": 2,
            "Natural Language Processing": 2
        }
    },
    "skills_proficiency": {
        "Programming Languages": {
            "Python": 95,
            "JavaScript": 88,
            "Java": 82,
            "TypeScript": 78,
            "C++": 75
        },
        "Frameworks": {
            "React": 90,
            "Node.js": 85,
            "TensorFlow": 80,
            "Flutter": 75,
            "Django": 85
        }
    },
    "activity_metrics": {
        "daily_commits": [12, 8, 15, 22, 18, 9, 5],
        "code_reviews": 45,
        "issues_resolved": 78,
        "pull_requests": 156,
        "active_streak": 23
    }
}

def create_modern_metric_card(title, value, subtitle="", delta=None, delta_color="normal"):
    """Create a modern looking metric card"""
    delta_html = ""
    if delta:
        color = "#28a745" if delta_color == "normal" else "#dc3545"
        delta_html = f"<div style='color: {color}; font-size: 14px; margin-top: 5px;'>↗ {delta}</div>"
    
    card_html = f"""
    <div style='
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
        color: white;
        margin: 10px 0;
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    '>
        <div style='font-size: 32px; font-weight: 700; margin-bottom: 5px;'>{value}</div>
        <div style='font-size: 16px; font-weight: 500; opacity: 0.9;'>{title}</div>
        <div style='font-size: 12px; opacity: 0.8;'>{subtitle}</div>
        {delta_html}
    </div>
    """
    return card_html

def create_enhanced_dashboard():
    """Create the enhanced dashboard"""
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .dashboard-container {
        background: linear-gradient(120deg, #a8edea 0%, #fed6e3 100%);
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
    }
    .metric-row {
        display: flex;
        gap: 15px;
        margin: 20px 0;
    }
    .chart-container {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 15px 0;
    }
    .section-header {
        font-size: 24px;
        font-weight: 600;
        color: #2c3e50;
        margin: 25px 0 15px 0;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<h1 style='text-align: center; color: #2c3e50; margin-bottom: 30px;'>📊 Developer Analytics Dashboard</h1>", unsafe_allow_html=True)
    
    # === OVERVIEW METRICS ROW ===
    st.markdown("<div class='section-header'>📈 Key Metrics Overview</div>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(create_modern_metric_card(
            "Total Repositories", 
            DASHBOARD_DATA["github_stats"]["total_repos"],
            "GitHub Projects",
            "+3 this month"
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown(create_modern_metric_card(
            "Total Commits", 
            f"{DASHBOARD_DATA['github_stats']['total_commits']:,}",
            "Lines of Impact",
            "+89 this week"
        ), unsafe_allow_html=True)
    
    with col3:
        st.markdown(create_modern_metric_card(
            "GitHub Stars", 
            DASHBOARD_DATA["github_stats"]["total_stars"],
            "Community Love",
            "+12 this month"
        ), unsafe_allow_html=True)
    
    with col4:
        st.markdown(create_modern_metric_card(
            "Research Papers", 
            DASHBOARD_DATA["research_metrics"]["papers_published"],
            "Published Works",
            "+2 this year"
        ), unsafe_allow_html=True)
    
    # === CHARTS ROW 1 ===
    st.markdown("<div class='section-header'>💻 Development Activity</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Programming Languages Donut Chart
        languages = DASHBOARD_DATA["github_stats"]["languages"]
        fig_lang = px.pie(
            values=list(languages.values()),
            names=list(languages.keys()),
            title="Programming Languages Distribution",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_lang.update_traces(textposition='inside', textinfo='percent+label')
        fig_lang.update_layout(
            height=400,
            showlegend=True,
            title_font_size=16,
            title_x=0.5,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_lang, use_container_width=True)
    
    with col2:
        # Monthly Commits Line Chart
        commits_data = DASHBOARD_DATA["github_stats"]["commits_per_month"]
        months = [item["month"] for item in commits_data]
        commits = [item["commits"] for item in commits_data]
        
        fig_commits = go.Figure()
        fig_commits.add_trace(go.Scatter(
            x=months,
            y=commits,
            mode='lines+markers',
            name='Commits',
            line=dict(color='#FF6B6B', width=3),
            marker=dict(size=8, color='#FF6B6B'),
            fill='tonexty',
            fillcolor='rgba(255,107,107,0.1)'
        ))
        
        fig_commits.update_layout(
            title="Monthly Commit Activity",
            xaxis_title="Month",
            yaxis_title="Commits",
            height=400,
            title_font_size=16,
            title_x=0.5,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        st.plotly_chart(fig_commits, use_container_width=True)
    
    # === CHARTS ROW 2 ===
    col1, col2 = st.columns(2)
    
    with col1:
        # Project Status Gauge Chart
        project_data = DASHBOARD_DATA["project_stats"]
        completed_percentage = (project_data["completed"] / project_data["total_projects"]) * 100
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=completed_percentage,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Project Completion Rate"},
            delta={'reference': 75},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "#1f77b4"},
                'steps': [
                    {'range': [0, 50], 'color': "#ffcccc"},
                    {'range': [50, 80], 'color': "#ffffcc"},
                    {'range': [80, 100], 'color': "#ccffcc"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig_gauge.update_layout(
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col2:
        # Skills Proficiency Radar Chart
        skills = DASHBOARD_DATA["skills_proficiency"]["Programming Languages"]
        categories = list(skills.keys())
        values = list(skills.values())
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Skills',
            line_color='#FF6B6B',
            fillcolor='rgba(255,107,107,0.25)'
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=False,
            title="Programming Skills Proficiency",
            title_x=0.5,
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    
    # === CHARTS ROW 3 ===
    st.markdown("<div class='section-header'>🚀 Project & Research Analytics</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Technology Usage Bar Chart
        tech_usage = DASHBOARD_DATA["project_stats"]["tech_usage"]
        tech_names = list(tech_usage.keys())
        tech_counts = list(tech_usage.values())
        
        fig_tech = px.bar(
            x=tech_counts,
            y=tech_names,
            orientation='h',
            title="Technology Usage Across Projects",
            color=tech_counts,
            color_continuous_scale='Viridis'
        )
        fig_tech.update_layout(
            height=400,
            title_x=0.5,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        st.plotly_chart(fig_tech, use_container_width=True)
    
    with col2:
        # Research Areas Distribution
        research_areas = DASHBOARD_DATA["research_metrics"]["research_areas"]
        fig_research = px.pie(
            values=list(research_areas.values()),
            names=list(research_areas.keys()),
            title="Research Focus Areas",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_research.update_traces(textposition='inside', textinfo='percent+label')
        fig_research.update_layout(
            height=400,
            title_x=0.5,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_research, use_container_width=True)
    
    # === ACTIVITY HEATMAP ===
    st.markdown("<div class='section-header'>📅 Weekly Activity Pattern</div>", unsafe_allow_html=True)
    
    # Generate sample weekly activity data
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    hours = list(range(24))
    
    # Create random activity data for visualization
    np.random.seed(42)
    activity_data = np.random.randint(0, 50, size=(7, 24))
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=activity_data,
        x=hours,
        y=days,
        colorscale='RdYlBu_r',
        showscale=True
    ))
    
    fig_heatmap.update_layout(
        title="Coding Activity Heatmap (Hour vs Day of Week)",
        xaxis_title="Hour of Day",
        yaxis_title="Day of Week",
        height=300,
        title_x=0.5,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # === FINAL METRICS ROW ===
    st.markdown("<div class='section-header'>🎯 Recent Achievements</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(create_modern_metric_card(
            "Citations", 
            DASHBOARD_DATA["research_metrics"]["citations"],
            "Research Impact"
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown(create_modern_metric_card(
            "Active Streak", 
            f"{DASHBOARD_DATA['activity_metrics']['active_streak']} days",
            "Consistent Development"
        ), unsafe_allow_html=True)
    
    with col3:
        st.markdown(create_modern_metric_card(
            "Code Reviews", 
            DASHBOARD_DATA["activity_metrics"]["code_reviews"],
            "Quality Contributions"
        ), unsafe_allow_html=True)

# Replace the dashboard section in your main code with:
if page.startswith("📈"):
    create_enhanced_dashboard()
