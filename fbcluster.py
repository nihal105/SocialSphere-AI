import streamlit as st
import numpy as np
import joblib
import time

st.set_page_config(
    page_title="Facebook User Segmentation AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

pipeline = joblib.load("facebook_clustering_pipeline.pkl")

css = """
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }

.stApp {
    background: linear-gradient(-45deg, #0f172a, #1a1f3a, #2d1b4e, #1e3c72, #0f172a);
    background-size: 400% 400%;
    animation: gradientShift 15s ease infinite;
    min-height: 100vh;
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.main-title {
    text-align: center;
    font-size: 65px;
    font-weight: 900;
    color: #00d4ff;
    animation: titleFloat 3s ease-in-out infinite;
    margin-bottom: 15px;
    letter-spacing: -2px;
    text-shadow: 0 0 20px rgba(0, 212, 255, 0.6), 0 0 40px rgba(0, 150, 255, 0.4);
}

@keyframes titleFloat {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
}

.sub-title {
    text-align: center;
    color: #7dd3fc;
    font-size: 20px;
    font-weight: 300;
    margin-bottom: 50px;
    letter-spacing: 1px;
    animation: subtitleSlideIn 1s ease-out;
}

@keyframes subtitleSlideIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.input-section {
    background: rgba(20, 30, 50, 0.6);
    backdrop-filter: blur(15px);
    border: 1px solid rgba(100, 200, 255, 0.2);
    border-radius: 25px;
    padding: 40px;
    margin: 30px 0;
    animation: containerSlideIn 0.8s ease-out;
    box-shadow: 0 0 30px rgba(0, 212, 255, 0.1);
}

@keyframes containerSlideIn {
    from { opacity: 0; transform: translateY(40px); }
    to { opacity: 1; transform: translateY(0); }
}

.stNumberInput > div > div > input {
    background: rgba(30, 41, 59, 0.8) !important;
    border: 2px solid rgba(100, 200, 255, 0.2) !important;
    border-radius: 15px !important;
    color: #00d4ff !important;
    font-weight: 600 !important;
    padding: 15px 20px !important;
    transition: all 0.3s ease !important;
}

.stNumberInput > div > div > input:focus {
    border-color: #00d4ff !important;
    box-shadow: 0 0 25px rgba(0, 212, 255, 0.3) !important;
    transform: translateY(-2px);
}

.stNumberInput > label {
    color: #7dd3fc !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    letter-spacing: 0.5px !important;
    text-transform: uppercase !important;
}

.stButton > button {
    width: 100% !important;
    height: 70px !important;
    border-radius: 20px !important;
    font-size: 22px !important;
    font-weight: 900 !important;
    border: 2px solid rgba(0, 212, 255, 0.3) !important;
    background: linear-gradient(135deg, rgba(0, 100, 255, 0.8), rgba(0, 212, 255, 0.8), rgba(100, 50, 255, 0.8)) !important;
    color: white !important;
    transition: all 0.4s ease !important;
    margin-top: 30px !important;
    box-shadow: 0 0 30px rgba(0, 212, 255, 0.3) !important;
    letter-spacing: 1px !important;
}

.stButton > button:hover {
    transform: translateY(-5px) scale(1.02) !important;
    box-shadow: 0 10px 40px rgba(0, 212, 255, 0.5) !important;
}

.result-box {
    padding: 50px;
    border-radius: 30px;
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.3), rgba(5, 150, 105, 0.3));
    backdrop-filter: blur(20px);
    border: 2px solid rgba(16, 185, 129, 0.4);
    color: #10b981;
    text-align: center;
    font-size: 50px;
    font-weight: 900;
    animation: resultPop 0.6s cubic-bezier(0.34, 1.56, 0.64, 1);
    box-shadow: 0 0 40px rgba(16, 185, 129, 0.3);
    margin: 30px 0;
    letter-spacing: 2px;
}

@keyframes resultPop {
    0% { opacity: 0; transform: scale(0.5) translateY(30px); }
    60% { transform: scale(1.1) translateY(-10px); }
    100% { opacity: 1; transform: scale(1) translateY(0); }
}

.info-box {
    padding: 35px;
    border-radius: 25px;
    background: rgba(30, 41, 59, 0.7);
    backdrop-filter: blur(20px);
    border: 2px solid rgba(125, 211, 252, 0.3);
    color: #7dd3fc;
    margin-top: 25px;
    animation: infoSlideUp 0.8s ease-out;
    box-shadow: 0 0 30px rgba(125, 211, 252, 0.15);
    border-left: 5px solid #7dd3fc;
}

.info-box h3 {
    font-size: 24px;
    margin-bottom: 15px;
    color: #00d4ff;
    font-weight: 900;
}

.info-box p {
    font-size: 16px;
    line-height: 1.8;
    color: #b0e7e8;
    font-weight: 500;
}

@keyframes infoSlideUp {
    from { opacity: 0; transform: translateY(40px); }
    to { opacity: 1; transform: translateY(0); }
}

hr { border: 0; height: 2px; background: linear-gradient(90deg, transparent, rgba(0, 212, 255, 0.5), transparent); margin: 50px 0; }

.section-header {
    font-weight: 700;
    color: #7dd3fc;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 20px;
    text-shadow: 0 0 10px rgba(125, 211, 252, 0.3);
}

@media (max-width: 768px) {
    .main-title { font-size: 45px; }
    .sub-title { font-size: 16px; }
    .result-box { font-size: 35px; padding: 35px; }
}

::-webkit-scrollbar { width: 12px; }
::-webkit-scrollbar-track { background: rgba(15, 23, 42, 0.5); }
::-webkit-scrollbar-thumb { background: linear-gradient(180deg, #00d4ff, #00a0ff); border-radius: 10px; }
</style>
"""

st.markdown(css, unsafe_allow_html=True)

st.markdown('<div class="main-title">🚀 Facebook User Segmentation AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">✨ Next-Level Machine Learning User Behavior Analysis ✨</div>', unsafe_allow_html=True)

st.markdown('<div class="input-section">', unsafe_allow_html=True)
st.markdown('<div class="section-header">👤 Personal Information</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="medium")

with col1:
    age = st.number_input("Age", min_value=1, max_value=100, value=25, step=1)
    tenure = st.number_input("Tenure (Days on Facebook)", min_value=0, value=365, step=10)
    friend_count = st.number_input("Friend Count", min_value=0, value=100, step=5)
    friendships_initiated = st.number_input("Friendships Initiated", min_value=0, value=50, step=5)
    likes = st.number_input("Likes", min_value=0, value=100, step=10)

with col2:
    st.markdown('<div class="section-header">📱 Engagement Metrics</div>', unsafe_allow_html=True)
    likes_received = st.number_input("Likes Received", min_value=0, value=50, step=5)
    mobile_likes = st.number_input("Mobile Likes", min_value=0, value=50, step=5)
    mobile_likes_received = st.number_input("Mobile Likes Received", min_value=0, value=50, step=5)
    www_likes = st.number_input("WWW Likes", min_value=0, value=50, step=5)
    www_likes_received = st.number_input("WWW Likes Received", min_value=0, value=50, step=5)

st.markdown('</div>', unsafe_allow_html=True)

col_button = st.columns([1, 2, 1])
with col_button[1]:
    if st.button("🔍 Predict User Segment", use_container_width=True):
        user_data = np.array([[
            age, tenure, friend_count, friendships_initiated, likes,
            likes_received, mobile_likes, mobile_likes_received, www_likes, www_likes_received
        ]])

        cluster = pipeline.predict(user_data)[0]

        cluster_names = {
            0: "🌱 New User",
            1: "⭐ Influencer",
            2: "👴 Senior User",
            3: "👑 Celebrity User",
            4: "🔥 Highly Active User"
        }

        descriptions = {
            0: "New users with low engagement and fewer social connections. These users are just beginning their journey on the platform and have limited interaction history.",
            1: "Influencers who receive high engagement and many interactions. These users have developed a strong presence and attract significant audience attention.",
            2: "Long-term users with high platform tenure and experience. These users have been on the platform for extended periods and have built stable engagement patterns.",
            3: "Extremely popular users with very large reach and audience engagement. These are the most prominent users on the platform with exceptional visibility and influence.",
            4: "Highly active users who frequently interact and build friendships. These users are constantly engaged, creating content and strengthening their social network."
        }

        time.sleep(0.3)
        st.markdown(f'<div class="result-box">{cluster_names.get(cluster, "Unknown")}</div>', unsafe_allow_html=True)
        time.sleep(0.2)
        st.markdown(f'<div class="info-box"><h3>📋 Segment Description</h3><p>{descriptions[cluster]}</p></div>', unsafe_allow_html=True)
        st.balloons()

st.markdown("---")
st.markdown('<div style="text-align: center; color: #94a3b8; font-size: 14px; font-weight: 600; letter-spacing: 1px; margin-top: 40px;">🎨 Built with Advanced CSS3 Animations • Python • Scikit-Learn • K-Means Clustering • Streamlit<br><span style="color: #7dd3fc; margin-top: 15px; display: block;">✨ Powered by Next-Gen UI/UX Design ✨</span></div>', unsafe_allow_html=True)
