import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# ================== تنظیمات صفحه ==================
st.set_page_config(
    page_title="HR Attrition Dashboard",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
body {
    background-color: #0e1117;  /* رنگ مشکی خیلی تیره */
    color: white;                /* متن سفید */
}
h1, h2, h3, h4 {
    color: white;                /* تیترها سفید */
}
.stDataFrame th {
    color: white;
    background-color: #1f2937;   /* هدر دیتافریم تیره */
}
.stDataFrame td {
    background-color: #111827;   /* ردیف‌های دیتافریم تیره */
    color: white;
}
</style>
""", unsafe_allow_html=True)


st.title("📊 داشبورد مدیریتی تحلیل ترک شغل کارکنان")
#======================================================
# ===== فونت فارسی Vazir و بولد =====
st.markdown("""
<style>
@import url('https://cdn.jsdelivr.net/gh/rastikerdar/vazir-font@33.003/Vazir-font-face.css');

body, h1, h2, h3, h4, h5, h6, p, div {
    font-family: 'Vazir', sans-serif;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ================== 1. بارگذاری داده ==================
df_original = pd.read_csv(r"data\WA_Fn-UseC_-HR-Employee-Attrition.csv")
df = df_original.copy()
df['AttritionBinary'] = df['Attrition'].map({'Yes':1,'No':0})

# ================== 2. آماده‌سازی داده ==================
numerical_cols = df.select_dtypes(include=['int64','float64']).columns.drop(
    ['EmployeeCount','StandardHours','EmployeeNumber','AttritionBinary']
)

scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

X = df.drop(['Attrition','AttritionBinary'], axis=1)
X = pd.get_dummies(X, drop_first=True)
y = df['AttritionBinary']

# ================== 3. مدل پیش‌بینی ==================
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

model = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)
model.fit(X_resampled, y_resampled)

df['RiskProbability'] = model.predict_proba(X)[:,1]

# ================== 4. تنظیمات سایدبار ==================
st.sidebar.header("🎛 تنظیمات داشبورد")
selected_department = st.sidebar.multiselect(
    "انتخاب دپارتمان",
    options=df['Department'].unique(),
    default=df['Department'].unique()
)
threshold = st.sidebar.slider("سطح ریسک", 0.3, 0.9, 0.6)

df_filtered = df[df['Department'].isin(selected_department)].copy()
df_filtered['HighRisk'] = df_filtered['RiskProbability'] > threshold

# ================== 5. KPIها ==================
attrition_rate = df_filtered['AttritionBinary'].mean() * 100
high_risk_count = df_filtered['HighRisk'].sum()
high_risk_percent = (high_risk_count / len(df_filtered)) * 100
financial_risk = high_risk_count * 30000

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

kpi1.markdown(f"""
<div style='background-color:#1f2937;padding:20px;border-radius:10px'>
<h4 style='color:white'>نرخ ترک</h4>
<h2 style='color:#00ffff'>{attrition_rate:.2f}%</h2>
</div>
""", unsafe_allow_html=True)

kpi2.markdown(f"""
<div style='background-color:#1f2937;padding:20px;border-radius:10px'>
<h4 style='color:white'>کارکنان پرریسک</h4>
<h2 style='color:#ff6b6b'>{high_risk_count}</h2>
</div>
""", unsafe_allow_html=True)

kpi3.markdown(f"""
<div style='background-color:#1f2937;padding:20px;border-radius:10px'>
<h4 style='color:white'>درصد پرریسک</h4>
<h2 style='color:#ffa600'>{high_risk_percent:.2f}%</h2>
</div>
""", unsafe_allow_html=True)

kpi4.markdown(f"""
<div style='background-color:#1f2937;padding:20px;border-radius:10px'>
<h4 style='color:white'>ریسک مالی بالقوه</h4>
<h2 style='color:#00ff88'>${financial_risk:,.0f}</h2>
</div>
""", unsafe_allow_html=True)

# ================== 6. نمودارها دو ستونه ==================
colA, colB = st.columns(2)

with colA:
    st.subheader("📊 نرخ ترک بر اساس دپارتمان")
    dept_rate = df_filtered.groupby('Department')['AttritionBinary'].mean() * 100
    fig_dept = px.bar(
        dept_rate.reset_index(),
        x='Department', y='AttritionBinary',
        color='AttritionBinary', color_continuous_scale='tealrose',
        labels={'AttritionBinary':'درصد ترک (%)'}
    )
    st.plotly_chart(fig_dept, use_container_width=True)

with colB:
    st.subheader("📈 توزیع احتمال ترک")
    fig_hist = px.histogram(df_filtered, x='RiskProbability', nbins=20, color_discrete_sequence=['#00d4ff'])
    st.plotly_chart(fig_hist, use_container_width=True)

# ================== 7. لیست کارکنان پرریسک ==================
st.subheader("👥 لیست کارکنان پرریسک")
st.dataframe(df_filtered[df_filtered['HighRisk']][['Age','Department','MonthlyIncome','RiskProbability']])

# ================== 8. مهم‌ترین عوامل ==================
importances = model.feature_importances_
feature_names = X.columns
feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False).head(10)
st.subheader("🔥 مهم‌ترین عوامل موثر در ترک شغل")
fig_feat = px.bar(feat_df, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='tealrose')
st.plotly_chart(fig_feat, use_container_width=True)

# ================== 9. پیش‌بینی کارمند خاص ==================
st.subheader("🔮 پیش‌بینی برای کارمند خاص")
emp_id = st.number_input("شماره کارمند", min_value=0, max_value=len(df)-1, value=0)
row = X.iloc[[emp_id]]
prob = model.predict_proba(row)[0][1]
st.write(f"احتمال ترک: **{prob:.2%}**")
if prob > threshold:
    st.error("⚠️ این کارمند در ریسک بالای ترک قرار دارد")
else:
    st.success("✅ ریسک ترک پایین است")
