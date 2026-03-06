import os
import subprocess
import sys

# 1. 自愈补丁：如果云端环境没同步好，强制安装缺失的库
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import joblib
except ImportError:
    install('joblib')
    import joblib

import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 2. 核心修复：确保文件名是小写 rf.pkl
@st.cache_resource
def load_model():
    return joblib.load('rf.pkl')

try:
    model = load_model()
except FileNotFoundError:
    st.error("错误：找不到模型文件 'rf.pkl'。请确认文件已上传到 GitHub 仓库根目录。")
    st.stop()

# 特征范围定义
feature_ranges = {
    "NtproBNP": {"type": "numerical", "min": 0.000, "max": 50000.000, "default": 670.236},
    "BMI": {"type": "numerical", "min": 10.000, "max": 50.000, "default": 24.555},
    "LeftAtrialDiam": {"type": "numerical", "min": 1.0, "max": 80.0, "default": 3.7},
    "AFCourse": {"type": "numerical", "min": 0, "max": 100, "default": 12},
    "AtrialFibrillationType": {"type": "categorical", "options": [0, 1], "default": 0},
    "SystolicBP": {"type": "numerical", "min": 50, "max": 200, "default": 116},
    "Age": {"type": "numerical", "min": 18, "max": 100, "default": 71},
    "AST": {"type": "numerical", "min": 0, "max": 1000, "default": 24}
}

st.title("Cardiovascular Disease Prediction App")
st.write("Please adjust the sliders and dropdowns to input patient characteristics.")

# 动态生成侧边栏输入项
feature_values = []
for feature, properties in feature_ranges.items():
    if properties["type"] == "numerical":
        value = st.sidebar.slider(
            label=f"{feature} ({properties['min']} - {properties['max']})",
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
        )
    else:
        value = st.sidebar.selectbox(
            label=f"{feature} (Select a value)",
            options=properties["options"],
        )
    feature_values.append(value)

features = np.array([feature_values])

# 预测与可视化
if st.button("Predict"):
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]
    probability = predicted_proba[predicted_class] * 100

    # 3. 字体修复：删掉了 fontname='Times New Roman'
    text = f"Based on feature values, predicted possibility of disease is {probability:.2f}%"
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.text(
        0.5, 0.5, text,
        fontsize=16,
        ha='center', va='center',
        transform=ax.transAxes
    )
    ax.axis('off')
    st.pyplot(fig) # 直接用这个显示，比保存成图片更快

    # SHAP 可视化
    st.write("### Model Explanation (SHAP Value)")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_ranges.keys()))
    
    # 兼容处理 shap_values 格式
    if isinstance(shap_values, list):
        current_shap_values = shap_values[predicted_class][0]
        base_value = explainer.expected_value[predicted_class]
    else:
        current_shap_values = shap_values[0, :, predicted_class]
        base_value = explainer.expected_value[predicted_class]

    # 画力导向图
    shap_fig = shap.force_plot(
        base_value, 
        current_shap_values, 
        pd.DataFrame([feature_values], columns=feature_ranges.keys()),
        matplotlib=True,
        show=False
    )
    st.pyplot(plt.gcf())
