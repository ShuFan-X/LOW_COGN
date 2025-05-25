import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
from lightgbm import Booster
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

df2 = pd.read_csv('x_test.csv')

x_test_CERAD = df2[['Age', 'Marital_married or living with a partner', 'PIR1', 'Education_level', 'BMI1', 'Smoking', 'Hyperlipidemia', 'LC9', 'PLT', 'CR']]
x_test_AFT = df2[['Age', 'Race_non-hispanic black', 'Marital_married or living with a partner', 'PIR1', 'Education_level', 'BMI1', 'Alcohol', 'Smoking', 'Hyperlipidemia', 'LC9', 'TC', 'CR']]
x_test_DSST = df2[['Age', 'Race_non-hispanic black', 'PIR', 'Education_level', 'CVD', 'CR']]

model_CERAD = joblib.load('CERAD_MLP.pkl')
model_AFT = Booster(model_file='AFT_LGBM.txt')
model_DSST = joblib.load('DSST_MLP.pkl')


feature_names_CERAD = ['Age', 'Marital_married or living with a partner', 'PIR1', 'Education_level', 'BMI1', 'Smoking', 'Hyperlipidemia', 'LC9', 'PLT', 'CR']
feature_names_AFT = ['Age', 'Race_non-hispanic black', 'Marital_married or living with a partner', 'PIR1', 'Education_level', 'BMI1', 'Alcohol', 'Smoking', 'Hyperlipidemia', 'LC9', 'TC', 'CR']
feature_names_DSST = ['Age', 'Race_non-hispanic black', 'PIR', 'Education_level', 'CVD', 'CR']

# 设置 Streamlit 应用的标题
st.title("Low cognitive performance diagnostic model")
st.sidebar.header("Selection Panel")  # 则边栏的标题
st.sidebar.subheader("Picking up paraneters")
Age = st.number_input("Age", min_value=0, max_value=120, value=1)
Gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x:"Male"if x == 1 else "Female")
Height = st.number_input("Height", min_value=0, max_value=250, value=1)
Weight = st.number_input("Weight", min_value=0, max_value=150, value=1)
Race = st.selectbox("Race", options=[0, 1], format_func=lambda x:"non-hispanic black"if x == 1 else "others")
Marital_status = st.selectbox("Marital_status", options=[0, 1], format_func=lambda x:"married or living with a partner"if x == 1 else "others")
Education_level = st.selectbox("Education", options=[1, 2, 3], format_func=lambda x: ("Less than High school" if x == 1 else "High school" if x == 2 else "More than High school" if x == 3 else ""))
PIR = st.number_input("PIR", min_value=0, max_value=10, value=1)
Nicotin_exposure = st.selectbox("Smoking", options=[100, 75, 50, 25, 0], format_func=lambda x: ("Never smoker, fewer than 100 cigarettes in lifetime" if x == 100 else "Former smoker, quit ≥5 years" if x == 75 else "Former smoker, quit ≥1 but <5 years" if x == 50 else "Former smoker, quit <1 year, or currently using inhaled NDS" if x == 25 else "Current smoker" if x == 0 else ""  ))
Alcohol = st.selectbox("Alcohol", options=[1, 2, 3, 4, 5], format_func=lambda x: ("Never drinkers, fewer than 12 lifetime drinks" if x == 1 else "Former drinkers, 12 or more drinks annually but have abstained in the past year" if x == 2 else "Mild drinkers, 1–2 drinks per day for females, 2–3 drinks per day for males" if x == 3 else "Moderate drinkers, 2–3 drinks per day for females, 3–4 drinks per day for males, or 2–4 binge episodes per month" if x == 4 else "Heavy drinkers, 3 or more drinks per day for females, 4 or more drinks per day for males, or 5 or more binge episodes per month" if x == 5 else ""  ))
Physical_activity = st.selectbox("Physical activity", options=[100, 90, 80, 60, 40, 20, 0], format_func=lambda x: ("≥150 minutes per week" if x == 100 else "120 – 149 minutes per week" if x == 90 else "90 – 119 minutes per week" if x == 80 else "60 – 89 minutes per week" if x == 60 else "30 – 59 minutes per week" if x == 40 else "1 – 29 minutes per week" if x == 20 else "0 minutes per week" if x == 0 else "" ))
Sleep_health = st.selectbox("Sleep health", options=[100, 90, 70, 40, 20, 0], format_func=lambda x: ("7 - 9 hours per night" if x == 100 else "9 - 10 hours per night" if x == 90 else "6 - 7 hours per night" if x == 70 else "5 – 6 or ≥10 hours per night" if x == 40 else "4 – 5 hours per night" if x == 20 else "<4 hours per night" if x == 0 else "" ))
Blood_glucose = st.selectbox("Blood glucose", options=[100, 60, 40, 30, 20, 10, 0], format_func=lambda x: ("No history of diabetes and FBG <100 (or HbA1c < 5.7)" if x == 100 else "No diabetes and FBG 100 – 125 (or HbA1c 5.7-6.4) (Pre-diabetes)" if x == 60 else "Diabetes with HbA1c <7.0" if x == 40 else "Diabetes with HbA1c 7.0 – 7.9" if x == 30 else "Diabetes with HbA1c 8.0 – 8.9" if x == 20 else "Diabetes with Hb A1c 9.0 – 9.9" if x == 10 else "Diabetes with HbA1c ≥10.0" if x == 0 else "" ))
Blood_pressure = st.selectbox("Blood pressure", options=[100, 75, 50, 25, 0], format_func=lambda x: ("SBP <120 or DBP <80 (Optimal)" if x == 100 else "SBP 120-129 or DBP <80 (Elevated)" if x == 75 else "SBP 130-139 or DBP 80-89 (Stage I HTN)" if x == 50 else "SBP 140-159 or DBP 90-99" if x == 25 else "SBP ≥160 or DBP ≥100" if x == 0 else ""  ))
CVD = st.selectbox("Cardiovascular Disease", options=[0, 1], format_func=lambda x:"YES"if x == 1 else "NO")
TC = st.number_input("Total cholesterol", min_value=0, max_value=250, value=1)
TG = st.number_input("Triglycerides", min_value=0, max_value=250, value=1)
HDL = st.number_input("HDL", min_value=0, max_value=250, value=1)
LDL = st.number_input("LDL", min_value=0, max_value=250, value=1)
PLT = st.number_input("PLT", min_value=0, max_value=250, value=1)
CR = st.number_input("CR", min_value=0, max_value=250, value=1)
PHQ_9 = st.selectbox("PHQ-9", options=[100, 70, 30, 0], format_func=lambda x: ("the score of 0 to 4 points" if x == 100 else "the score of 5 to 9 points" if x == 70 else "the score of 10 to 14 points" if x == 30 else "the score of 15 to 27 points" if x == 0 else ""  ))
Diet_score = st.selectbox("Healthy Eating Index-2015 diet score", options=[100, 80, 50, 25, 0], format_func=lambda x: ("≥95 percentile (top/ideal diet)" if x == 100 else "75 – 94 percentile" if x == 80 else "50 – 74 percentile" if x == 50 else "25 – 49 percentile" if x == 25 else "1 – 24 percentile (bottom/least ideal quartile" if x == 0 else ""  ))

if Nicotin_exposure == 100:
    Smoking = 1
elif 50 <= Nicotin_exposure <= 75:
    Smoking = 2
else:
    Smoking = 3

if PIR >=1:
    PIR1 = 1
else:
    PIR1 = 0

BMI = Weight*10000/Height/Height
if BMI < 25.0:
    BMI_score = 100
elif 25.0 <= BMI <= 29.9:
    BMI_score = 70
elif 30.0 <= BMI <= 34.9:
    BMI_score = 30 
elif 35.0 <= BMI <= 39.9:
    BMI_score = 15
else:
    BMI_score = 0

if BMI < 25.0:
    BMI1 = 1
elif 25.0 <= BMI < 30:
    BMI1 = 2
else:
    BMI1 = 3
    
Non_HDL = TC-HDL
if Non_HDL < 130:
    Blood_lipids = 100
elif 130 <= Non_HDL <= 159:
    Blood_lipids = 60 
elif 160 <= Non_HDL <= 189:
    Blood_lipids = 40
elif 190 <= Non_HDL <= 219:
    Blood_lipids = 20
else:
    Blood_lipids = 0

Hyperlipidemia = int(
    (TG >= 150) or 
    (TC >= 200) or 
    (LDL >= 130) or 
    ((Gender == 1 and HDL < 40) or (Gender == 0 and HDL < 50))
)

LC9 = (PHQ_9 + Diet_score + Physical_activity + Nicotin_exposure + Sleep_health + BMI_score + Blood_lipids + Blood_glucose + Blood_pressure)/9

feature_values_CERAD = [Age, Marital_status, PIR1, Education_level, BMI1, Smoking, Hyperlipidemia, LC9, PLT, CR ]
feature_values_AFT = [Age, Race, Marital_status, PIR1, Education_level, BMI1, Alcohol, Smoking, Hyperlipidemia, LC9, TC, CR ]
feature_values_DSST = [Age, Race, PIR, Education_level, CVD, CR ]
features_CERAD = np.array([feature_values_CERAD])
features_AFT = np.array([feature_values_AFT])
features_DSST = np.array([feature_values_DSST])

if st.button("Predict"):
    ##########  CERAD  ####################
    st.title("Diagnosis of LOW_CERAD")
    predicted_class = model_CERAD.predict(features_CERAD)[0]
    predicted_proba = model_CERAD.predict_proba(features_CERAD)[0]
    st.write(f"**Predicted Class:** {predicted_class} (0: HIGH_CERAD, 1: LOW_CERAD)")
    st.write(f"**Predicted Probabilities:** {predicted_proba}")
    probability = predicted_proba[predicted_class] * 100
    # 如果预测类别为1（高风险）
    if predicted_class == 1:
        advice = (
            f"According to our model, you have a high risk of LOW_CERAD. "
            f"The model predicts that your probability of having LOW_CERAD is {probability:.1f}%."
            "It's advised to consult with your healthcare provider for further evaluation and possible intervention."
        )

    # 如果预测类别为0（低风险）
    else:
        advice = (
            f"According to our model, you have a low risk of  LOW_CERAD. "
            f"The model predicts that your probability of not having LOW_CERAD is {probability:.1f}%."
            "However, maintaining a healthy lifestyle is important. Please continue regular check-ups with your healthcare provider."
        )
    # 显示建议
    st.write(advice)
    # SHAP 解释
    st.subheader("SHAP Force Plot Explanation")

    explainer_shap = shap.KernelExplainer(model_CERAD.predict, data=x_test_CERAD)
    shap_values = explainer_shap.shap_values(pd.DataFrame([feature_values_CERAD], columns=feature_names_CERAD))
    if predicted_class == 1:
        shap.force_plot(explainer_shap.expected_value, shap_values[0], pd.DataFrame([feature_values_CERAD], columns=feature_names_CERAD), matplotlib=True)

    else:
        shap.force_plot(explainer_shap.expected_value, shap_values[0], pd.DataFrame([feature_values_CERAD], columns=feature_names_CERAD), matplotlib=True)

    plt.savefig("shap_force_plot_CERAD.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot_CERAD.png", caption='SHAP Force Plot Explanation')

    # LIME Explanation
    st.subheader("LIME Explanation")
    lime_explainer = LimeTabularExplainer(
        training_data=x_test_CERAD.values,
        feature_names=x_test_CERAD.columns.tolist(),
        class_names=['HIGH_CERAD', 'LOW_CERAD'],  # Adjust class names to match your classification task
        mode='classification'
    )

    # Explain the instance
    lime_exp = lime_explainer.explain_instance(
        data_row=features_CERAD.flatten(),
        predict_fn=model_CERAD.predict_proba
    )

    # Display the LIME explanation without the feature value table
    lime_html = lime_exp.as_html(show_table=False)  # Disable feature value table
    st.components.v1.html(lime_html, height=800, scrolling=True)
    ##########  AFT  ####################
    st.title("Diagnosis of LOW_AFT")
    #predicted_class = model_AFT.predict(features_AFT)[0]
    #predicted_proba = model_AFT.predict_proba(features_AFT)[0]
    raw_scores = model_AFT.predict(features_AFT)  # 返回 [0, 1] 的概率值
    predicted_proba = np.vstack([1 - raw_scores, raw_scores]).T[0]  # 转为 [[p0, p1], ...]
    predicted_class = (raw_scores > 0.5).astype(int)[0]  # 阈值 0.5
    st.write(f"**Predicted Class:** {predicted_class} (0: HIGH_AFT, 1: LOW_AFT)")
    st.write(f"**Predicted Probabilities:** {predicted_proba}")
    probability = predicted_proba[predicted_class] * 100
    # 如果预测类别为1（高风险）
    if predicted_class == 1:
        advice = (
            f"According to our model, you have a high risk of LOW_AFT. "
            f"The model predicts that your probability of having LOW_AFT is {probability:.1f}%."
            "It's advised to consult with your healthcare provider for further evaluation and possible intervention."
        )

    # 如果预测类别为0（低风险）
    else:
        advice = (
            f"According to our model, you have a low risk of  LOW_AFT. "
            f"The model predicts that your probability of not having LOW_AFT is {probability:.1f}%."
            "However, maintaining a healthy lifestyle is important. Please continue regular check-ups with your healthcare provider."
        )
    # 显示建议
    st.write(advice)
    # SHAP 解释
    st.subheader("SHAP Force Plot Explanation")
    explainer_shap = shap.KernelExplainer(model_AFT.predict, data=x_test_AFT)
    shap_values = explainer_shap.shap_values(pd.DataFrame([feature_values_AFT], columns=feature_names_AFT))
    if predicted_class == 1:
        shap.force_plot(explainer_shap.expected_value, shap_values[0], pd.DataFrame([feature_values_AFT], columns=feature_names_AFT), matplotlib=True)

    else:
        shap.force_plot(explainer_shap.expected_value, shap_values[0], pd.DataFrame([feature_values_AFT], columns=feature_names_AFT), matplotlib=True)

    plt.savefig("shap_force_plot_AFT.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot_AFT.png", caption='SHAP Force Plot Explanation')

    # LIME Explanation
    st.subheader("LIME Explanation")
    lime_explainer = LimeTabularExplainer(
        training_data=x_test_AFT.values,
        feature_names=x_test_AFT.columns.tolist(),
        class_names=['HIGH_AFT', 'LOW_AFT'],  # Adjust class names to match your classification task
        mode='classification'
    )
    def booster_predict_proba(self, X):
        raw_pred = self.predict(X)
        return np.vstack([1 - raw_pred, raw_pred]).T
    
    # Explain the instance
    lime_exp = lime_explainer.explain_instance(
        data_row=features_AFT.flatten(),
        predict_fn=booster_predict_proba.__get__(model_AFT)
    )

    # Display the LIME explanation without the feature value table
    lime_html = lime_exp.as_html(show_table=False)  # Disable feature value table
    st.components.v1.html(lime_html, height=800, scrolling=True)
    ##########  DSST  ####################
    st.title("Diagnosis of LOW_DSST")
    predicted_class = model_DSST.predict(features_DSST)[0]
    predicted_proba = model_DSST.predict_proba(features_DSST)[0]
    st.write(f"**Predicted Class:** {predicted_class} (0: HIGH_DSST, 1: LOW_DSST)")
    st.write(f"**Predicted Probabilities:** {predicted_proba}")
    probability = predicted_proba[predicted_class] * 100
    # 如果预测类别为1（高风险）
    if predicted_class == 1:
        advice = (
            f"According to our model, you have a high risk of LOW_DSST. "
            f"The model predicts that your probability of having LOW_DSST is {probability:.1f}%."
            "It's advised to consult with your healthcare provider for further evaluation and possible intervention."
        )

    # 如果预测类别为0（低风险）
    else:
        advice = (
            f"According to our model, you have a low risk of  LOW_DSST. "
            f"The model predicts that your probability of not having LOW_DSST is {probability:.1f}%."
            "However, maintaining a healthy lifestyle is important. Please continue regular check-ups with your healthcare provider."
        )
    # 显示建议
    st.write(advice)
    # SHAP 解释
    st.subheader("SHAP Force Plot Explanation")

    explainer_shap = shap.KernelExplainer(model_DSST.predict, data=x_test_DSST)
    shap_values = explainer_shap.shap_values(pd.DataFrame([feature_values_DSST], columns=feature_names_DSST))
    if predicted_class == 1:
        shap.force_plot(explainer_shap.expected_value, shap_values[0], pd.DataFrame([feature_values_DSST], columns=feature_names_DSST), matplotlib=True)

    else:
        shap.force_plot(explainer_shap.expected_value, shap_values[0], pd.DataFrame([feature_values_DSST], columns=feature_names_DSST), matplotlib=True)

    plt.savefig("shap_force_plot_DSST.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot_DSST.png", caption='SHAP Force Plot Explanation')

    # LIME Explanation
    st.subheader("LIME Explanation")
    lime_explainer = LimeTabularExplainer(
        training_data=x_test_DSST.values,
        feature_names=x_test_DSST.columns.tolist(),
        class_names=['HIGH_DSST', 'LOW_DSST'],  # Adjust class names to match your classification task
        mode='classification'
    )

    # Explain the instance
    lime_exp = lime_explainer.explain_instance(
        data_row=features_DSST.flatten(),
        predict_fn=model_DSST.predict_proba
    )

    # Display the LIME explanation without the feature value table
    lime_html = lime_exp.as_html(show_table=False)  # Disable feature value table
    st.components.v1.html(lime_html, height=800, scrolling=True)

    
