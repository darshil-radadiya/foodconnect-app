"""
FoodConnect — Streamlit Application
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
import json
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FoodConnect",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── global ── */
[data-testid="stAppViewContainer"] { background: #0f1117; }
[data-testid="stSidebar"]          { background: #1a1d27; border-right: 1px solid #2d3047; }
h1,h2,h3,h4 { color: #f0f2f6; }
p, li        { color: #c9cdd6; }

/* ── metric cards ── */
.metric-card {
    background: linear-gradient(135deg,#1e2130,#252838);
    border: 1px solid #2d3047;
    border-radius: 14px;
    padding: 22px 28px;
    text-align: center;
    box-shadow: 0 4px 18px rgba(0,0,0,.4);
    margin-bottom: 14px;
}
.metric-card .label { font-size:.8rem; color:#9ca3af; letter-spacing:.08em; text-transform:uppercase; }
.metric-card .value { font-size:2.2rem; font-weight:700; color:#f9fafb; margin:6px 0 0; }
.metric-card .unit  { font-size:.85rem; color:#6b7280; }

/* ── prediction result card ── */
.pred-card {
    background: linear-gradient(135deg,#1c2e1c,#1e3320);
    border: 1px solid #2d6a2d;
    border-radius: 18px;
    padding: 32px;
    text-align: center;
    box-shadow: 0 6px 24px rgba(0,180,0,.15);
}
.pred-card .rating-num { font-size:5rem; font-weight:800; color:#4ade80; line-height:1; }
.pred-card .rating-lbl { font-size:1rem; color:#86efac; margin-top:8px; }
.pred-card .stars       { font-size:2rem; color:#facc15; margin: 12px 0 6px; }

/* ── section header ── */
.section-header {
    background: linear-gradient(90deg,#6366f1,#8b5cf6);
    border-radius:10px; padding:12px 22px; margin-bottom:18px;
    font-size:1.1rem; font-weight:700; color:#fff;
}

/* ── upload box ── */
.stFileUploader > div { border: 2px dashed #4f46e5 !important; border-radius:12px !important; }

/* ── sidebar nav ── */
.sidebar-brand {
    text-align:center; padding:18px 0 10px;
    font-size:1.6rem; font-weight:800; color:#818cf8;
    letter-spacing:.04em;
}
.sidebar-sub { text-align:center; font-size:.78rem; color:#6b7280; margin-top:-8px; margin-bottom:20px; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# LOAD MODEL BUNDLE
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_bundle():
    with open("foodconnect_model.pkl", "rb") as f:
        return pickle.load(f)

try:
    bundle = load_bundle()
    pipeline      = bundle['pipeline']
    FEATURES      = bundle['features']
    encoding_maps = bundle['encoding_maps']
    g_defaults    = bundle['global_defaults']
    locations     = bundle['locations']
    cuisines      = bundle['cuisines']
    rest_types    = bundle['restaurant_types']
    MODEL_LOADED  = True
except FileNotFoundError:
    MODEL_LOADED  = False
    st.error("❌ `foodconnect_model.pkl` not found. Run `python training.py` first.")
    st.stop()


# ──────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────
def te_lookup(col_name, value, stat='te'):
    """Look up target-encoded value for a given column/value."""
    m = encoding_maps.get(col_name, {})
    if value in m:
        return m[value][stat]
    return g_defaults.get(f'{col_name}_{stat}', bundle['rating_mean'])


def build_feature_row(location, cost, cuisine, votes, online_order, table_booking,
                      restaurant_type=None, local_address=None):
    """Construct the model input DataFrame from user inputs."""
    if restaurant_type is None:
        restaurant_type = rest_types[0]
    if local_address is None:
        local_address = location

    row = {
    'cost': cost,
    'votes': votes,
    'online_order': int(online_order),
    'table_booking': int(table_booking),

    'log_votes': np.log1p(votes),
    'log_cost': np.log1p(cost),

    'votes_cost_ratio': votes / (cost + 1),
    'cost_per_vote': cost / (votes + 1),
    'popularity': votes * 3.5,   # approx rating impact
    'sqrt_votes': np.sqrt(votes),
}

    for col, val in [('location', location), ('cuisine', cuisine),
                     ('restaurant_type', restaurant_type), ('local_address', local_address)]:
        for stat in ['te', 'std', 'cnt']:
            row[f'{col}_{stat}'] = te_lookup(col, val, stat)

    return pd.DataFrame([row])[FEATURES]


def predict_rating(row_df):
    """Run pipeline prediction and clip to valid range."""
    pred = pipeline.predict(row_df)[0]
    return float(np.clip(pred, 1.0, 5.0))


def stars(rating):
    full  = int(rating)
    half  = 1 if (rating - full) >= 0.35 else 0
    empty = 5 - full - half
    return "★" * full + "½" * half + "☆" * empty


def rating_label(r):
    if r >= 4.5: return "🌟 Excellent"
    if r >= 4.0: return "😍 Very Good"
    if r >= 3.5: return "👍 Good"
    if r >= 3.0: return "🙂 Average"
    if r >= 2.0: return "😐 Below Average"
    return "😞 Poor"


def bulk_predict(df_up):
    """
    Auto-map columns, engineer features, and predict on uploaded dataframe.
    Returns (result_df, error_message)
    """
    col_aliases = {
        'location': ['location','area','city','zone','place','address'],
        'cost':     ['cost','avg cost','avg cost (two people)','price','budget'],
        'cuisine':  ['cuisine','cuisines type','food type','food','cuisines'],
        'votes':    ['votes','num of ratings','ratings count','reviews','num_ratings'],
        'online_order': ['online_order','online order','delivery','online'],
        'table_booking':['table_booking','table booking','reservation','dine_in'],
        'restaurant_type': ['restaurant_type','restaurant type','type','rest type'],
        'local_address': ['local_address','local address','neighbourhood','locality'],
    }

    col_map = {}
    lower_cols = {c.lower().strip(): c for c in df_up.columns}
    for target, aliases in col_aliases.items():
        for a in aliases:
            if a.lower() in lower_cols:
                col_map[target] = lower_cols[a.lower()]
                break

    required = ['location','cost','cuisine','votes']
    missing  = [r for r in required if r not in col_map]
    if missing:
        return None, f"Missing required columns: {', '.join(missing)}. Found: {list(df_up.columns)}"

    result = df_up.copy()
    preds  = []

    for _, row_raw in df_up.iterrows():
        def get(field, default=None):
            c = col_map.get(field)
            if c and c in row_raw.index:
                v = row_raw[c]
                if pd.isna(v):
                    return default
                return v
            return default

        location = str(get('location', locations[0]))
        cuisine  = str(get('cuisine',  cuisines[0]))
        try:
            cost  = float(get('cost',  bundle['cost_median']))
        except: cost = bundle['cost_median']
        try:
            votes = float(get('votes', bundle['votes_median']))
        except: votes = bundle['votes_median']

        oo = get('online_order', 'No')
        tb = get('table_booking', 'No')
        online_order  = 1 if str(oo).strip().lower() in ['yes','1','true','y'] else 0
        table_booking = 1 if str(tb).strip().lower() in ['yes','1','true','y'] else 0

        rest_type = str(get('restaurant_type', rest_types[0]))
        local_addr = str(get('local_address', location))

        row_df = build_feature_row(location, cost, cuisine, votes, online_order, table_booking, rest_type, local_addr)
        preds.append(round(predict_rating(row_df), 2))

    result.insert(0, '⭐ Predicted Rating', preds)
    return result, None


def show_analysis_charts(df_a):
    """Render 6 analysis charts for the uploaded/demo dataset."""
    if 'rating' not in df_a.columns:
        for alias in ['rate (out of 5)','rate','predicted rating','⭐ predicted rating']:
            if alias in [c.lower() for c in df_a.columns]:
                df_a = df_a.rename(columns={c: 'rating' for c in df_a.columns if c.lower()==alias})
                break
        else:
            st.warning("No 'rating' column found. Showing structural analysis only.")
            return

    st.markdown('<div class="section-header">📊 Data Analysis Dashboard</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Chart 1: Rating distribution
        fig, ax = plt.subplots(figsize=(6,4))
        fig.patch.set_facecolor('#1e2130'); ax.set_facecolor('#1e2130')
        ax.hist(df_a['rating'].dropna(), bins=20, color='#6366f1', edgecolor='white', linewidth=0.7)
        ax.axvline(df_a['rating'].mean(), color='#f87171', linestyle='--', label=f"Mean: {df_a['rating'].mean():.2f}")
        ax.set_title("Rating Distribution", color='white', fontweight='bold')
        ax.set_xlabel("Rating", color='#9ca3af'); ax.set_ylabel("Count", color='#9ca3af')
        ax.tick_params(colors='#9ca3af'); ax.legend(labelcolor='white')
        for spine in ax.spines.values(): spine.set_edgecolor('#2d3047')
        st.pyplot(fig); plt.close()

    with col2:
        # Chart 2: Cost vs Rating
        if 'cost' in df_a.columns or 'avg cost (two people)' in df_a.columns:
            cost_col = 'cost' if 'cost' in df_a.columns else 'avg cost (two people)'
            fig, ax = plt.subplots(figsize=(6,4))
            fig.patch.set_facecolor('#1e2130'); ax.set_facecolor('#1e2130')
            ax.scatter(pd.to_numeric(df_a[cost_col], errors='coerce'),
                       pd.to_numeric(df_a['rating'], errors='coerce'),
                       alpha=0.3, color='#34d399', s=12)
            ax.set_title("Cost vs Rating", color='white', fontweight='bold')
            ax.set_xlabel("Cost (₹)", color='#9ca3af'); ax.set_ylabel("Rating", color='#9ca3af')
            ax.tick_params(colors='#9ca3af')
            for spine in ax.spines.values(): spine.set_edgecolor('#2d3047')
            st.pyplot(fig); plt.close()

    col3, col4 = st.columns(2)

    with col3:
        # Chart 3: Top locations
        loc_col = next((c for c in df_a.columns if c.lower() in ['location','area']), None)
        if loc_col:
            top_loc = df_a[loc_col].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(6,4))
            fig.patch.set_facecolor('#1e2130'); ax.set_facecolor('#1e2130')
            ax.barh(top_loc.index[::-1], top_loc.values[::-1], color='#fb923c')
            ax.set_title("Top 10 Locations", color='white', fontweight='bold')
            ax.set_xlabel("Count", color='#9ca3af')
            ax.tick_params(colors='#9ca3af')
            for spine in ax.spines.values(): spine.set_edgecolor('#2d3047')
            st.pyplot(fig); plt.close()

    with col4:
        # Chart 4: Cuisine popularity
        cuis_col = next((c for c in df_a.columns if 'cuisine' in c.lower()), None)
        if cuis_col:
            top_cuis = df_a[cuis_col].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(6,4))
            fig.patch.set_facecolor('#1e2130'); ax.set_facecolor('#1e2130')
            ax.barh(top_cuis.index[::-1], top_cuis.values[::-1], color='#a78bfa')
            ax.set_title("Top 10 Cuisines", color='white', fontweight='bold')
            ax.set_xlabel("Count", color='#9ca3af')
            ax.tick_params(colors='#9ca3af')
            for spine in ax.spines.values(): spine.set_edgecolor('#2d3047')
            st.pyplot(fig); plt.close()

    col5, col6 = st.columns(2)

    with col5:
        # Chart 5: Votes vs Rating
        votes_col = next((c for c in df_a.columns if 'vote' in c.lower() or 'rating' in c.lower() and c != 'rating'), None)
        if votes_col:
            fig, ax = plt.subplots(figsize=(6,4))
            fig.patch.set_facecolor('#1e2130'); ax.set_facecolor('#1e2130')
            ax.scatter(pd.to_numeric(df_a[votes_col], errors='coerce'),
                       pd.to_numeric(df_a['rating'], errors='coerce'),
                       alpha=0.3, color='#f472b6', s=12)
            ax.set_title("Votes vs Rating", color='white', fontweight='bold')
            ax.set_xlabel("Votes", color='#9ca3af'); ax.set_ylabel("Rating", color='#9ca3af')
            ax.tick_params(colors='#9ca3af')
            for spine in ax.spines.values(): spine.set_edgecolor('#2d3047')
            st.pyplot(fig); plt.close()

    with col6:
        # Chart 6: Correlation heatmap
        num_cols = df_a.select_dtypes(include='number').columns[:6]
        if len(num_cols) >= 2:
            fig, ax = plt.subplots(figsize=(6,4))
            fig.patch.set_facecolor('#1e2130'); ax.set_facecolor('#1e2130')
            corr = df_a[num_cols].corr()
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax,
                        linewidths=0.5, annot_kws={'color':'white','size':8})
            ax.set_title("Correlation Heatmap", color='white', fontweight='bold')
            ax.tick_params(colors='#9ca3af')
            st.pyplot(fig); plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-brand">🍽️ FoodConnect</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Restaurant Rating Predictor</div>', unsafe_allow_html=True)
    st.divider()

    nav = st.radio(
        "Navigate",
        ["🎯 Manual Prediction", "📂 Bulk Scanner", "📊 Data Analysis"],
        label_visibility="collapsed",
    )

   

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — MANUAL PREDICTION
# ──────────────────────────────────────────────────────────────────────────────
if nav == "🎯 Manual Prediction":
    st.markdown("# 🎯 Manual Prediction")
    st.markdown("Fill in the restaurant details below to instantly predict its rating.")
    st.divider()

    with st.form("predict_form"):
        c1, c2 = st.columns(2)

        with c1:
            location = st.selectbox("📍 Location", locations)
            cuisine  = st.selectbox("🍜 Cuisine Type", cuisines)
            rest_type = st.selectbox("🏪 Restaurant Type", rest_types)

        with c2:
            cost          = st.number_input("💰 Avg Cost for 2 (₹)", min_value=50, max_value=5000, value=500, step=50)
            votes         = st.number_input("🗳️ Number of Votes/Ratings", min_value=1, max_value=50000, value=200, step=10)
            online_order  = st.selectbox("🛵 Online Delivery Available?", ["Yes", "No"])
            table_booking = st.selectbox("📅 Table Booking Available?", ["No", "Yes"])

        submitted = st.form_submit_button("⚡ Predict Rating", use_container_width=True, type="primary")

    if submitted:
        oo = 1 if online_order == "Yes" else 0
        tb = 1 if table_booking == "Yes" else 0
        row_df  = build_feature_row(location, cost, cuisine, votes, oo, tb, rest_type, location)
        pred    = predict_rating(row_df)
        s       = stars(pred)
        lbl     = rating_label(pred)

        st.markdown("<br>", unsafe_allow_html=True)
        col_a, col_b, col_c = st.columns([1,2,1])
        with col_b:
            st.markdown(f"""
            <div class="pred-card">
                <div style="font-size:.9rem;color:#86efac;margin-bottom:6px;">PREDICTED RATING</div>
                <div class="rating-num">{pred:.1f}</div>
                <div class="stars">{s}</div>
                <div class="rating-lbl">{lbl}</div>
                <hr style="border-color:#2d6a2d;margin:16px 0;">
                <div style="display:flex;justify-content:space-around;font-size:.85rem;color:#86efac;">
                    <span>📍 {location}</span>
                    <span>🍜 {cuisine[:25]}</span>
                </div>
                <div style="display:flex;justify-content:space-around;font-size:.85rem;color:#86efac;margin-top:8px;">
                    <span>💰 ₹{cost}</span>
                    <span>🗳️ {votes} votes</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        for col, label, val, unit in [
            (m1, "Predicted Rating", f"{pred:.2f}", "/ 5.0"),
            (m2, "Star Grade",       s[:5],          ""),
            (m3, "Assessment",       lbl.split(" ",1)[0], lbl.split(" ",1)[-1]),
            (m4, "Cost per Head",    f"₹{cost//2}",  "per person"),
        ]:
            col.markdown(f"""
            <div class="metric-card">
                <div class="label">{label}</div>
                <div class="value">{val}</div>
                <div class="unit">{unit}</div>
            </div>""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — BULK SCANNER
# ──────────────────────────────────────────────────────────────────────────────
elif nav == "📂 Bulk Scanner":
    st.markdown("# 📂 Bulk Scanner")
    st.markdown("Upload a dataset to predict ratings for all restaurants at once.")
    st.divider()

    # ================= SAMPLE FILE DOWNLOAD =================

st.subheader("📥 Download Sample File")

sample_data = pd.DataFrame({
    "location": ["Ahmedabad", "Mumbai"],
    "cuisine": ["North Indian", "Chinese"],
    "cost": [500, 800],
    "votes": [120, 250],
    "online_order": ["Yes", "No"],
    "table_booking": ["No", "Yes"]
})

# CSV download
csv_sample = sample_data.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📄 Download Sample CSV",
    data=csv_sample,
    file_name="sample_foodconnect.csv",
    mime="text/csv"
)

# Excel download
buffer = io.BytesIO()
sample_data.to_excel(buffer, index=False, engine='openpyxl')

st.download_button(
    label="📊 Download Sample Excel",
    data=buffer.getvalue(),
    file_name="sample_foodconnect.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

st.info("👉 Use this format to upload your file for bulk prediction.")

upload_type = st.radio("Upload format", ["CSV", "Excel", "JSON", "Google Drive Link"],
                           horizontal=True)

df_uploaded = None

    if upload_type in ["CSV", "Excel", "JSON"]:
        ext_map = {"CSV": ["csv"], "Excel": ["xlsx","xls"], "JSON": ["json"]}
        file = st.file_uploader(f"Upload your {upload_type} file",
                                type=ext_map[upload_type])
        if file:
            try:
                if upload_type == "CSV":
                    df_uploaded = pd.read_csv(file)
                elif upload_type == "Excel":
                    df_uploaded = pd.read_excel(file)
                else:
                    df_uploaded = pd.read_json(file)
            except Exception as e:
                st.error(f"Could not parse file: {e}")

    elif upload_type == "Google Drive Link":
        gdrive_url = st.text_input("Paste Google Drive shareable CSV link")
        if gdrive_url and st.button("Load from Drive"):
            try:
                file_id   = gdrive_url.split("/d/")[1].split("/")[0]
                direct    = f"https://drive.google.com/uc?id={file_id}&export=download"
                df_uploaded = pd.read_csv(direct)
                st.success("Loaded from Google Drive!")
            except Exception as e:
                st.error(f"Failed to load from Drive: {e}")

    if df_uploaded is not None:
        st.markdown("#### 📋 Dataset Preview (first 5 rows)")
        st.dataframe(df_uploaded.head(), use_container_width=True)
        st.caption(f"Shape: {df_uploaded.shape[0]} rows × {df_uploaded.shape[1]} columns")

        if st.button("🚀 Run Bulk Prediction", type="primary"):
            with st.spinner("Predicting ratings…"):
                result_df, err = bulk_predict(df_uploaded)

            if err:
                st.error(f"⚠️ {err}\n\nThe uploaded dataset structure is not compatible with the model. "
                         "Required columns: location, cost, cuisine, votes.")
            else:
                st.success(f"✅ Predictions complete for {len(result_df)} restaurants!")
                st.dataframe(result_df, use_container_width=True)

                # Summary stats
                preds = result_df['⭐ Predicted Rating']
                a1, a2, a3, a4 = st.columns(4)
                for col, lab, v in [(a1,"Avg Rating",f"{preds.mean():.2f}"),
                                    (a2,"Highest",   f"{preds.max():.2f}"),
                                    (a3,"Lowest",    f"{preds.min():.2f}"),
                                    (a4,"Restaurants",str(len(result_df)))]:
                    col.markdown(f'<div class="metric-card"><div class="label">{lab}</div>'
                                 f'<div class="value">{v}</div></div>', unsafe_allow_html=True)

                st.markdown("#### ⬇️ Download Results")
                dl1, dl2, dl3, dl4 = st.columns(4)

                csv_bytes = result_df.to_csv(index=False).encode()
                dl1.download_button("📄 CSV", csv_bytes, "predictions.csv", "text/csv",
                                    use_container_width=True)

                buf = io.BytesIO()
                result_df.to_excel(buf, index=False, engine='openpyxl')
                dl2.download_button("📊 Excel", buf.getvalue(),
                                    "predictions.xlsx",
                                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    use_container_width=True)

                json_bytes = result_df.to_json(orient='records', indent=2).encode()
                dl3.download_button("🗃️ JSON", json_bytes, "predictions.json",
                                    "application/json", use_container_width=True)

                # SQL dump
                rows_sql = []
                for _, r in result_df.iterrows():
                    vals = ", ".join([f"'{str(v).replace(chr(39),chr(39)*2)}'" for v in r.values])
                    rows_sql.append(f"INSERT INTO predictions VALUES ({vals});")
                sql_text = "-- FoodConnect Predictions\n" + "\n".join(rows_sql)
                dl4.download_button("🗄️ SQL", sql_text.encode(), "predictions.sql",
                                    "text/plain", use_container_width=True)

    else:
        st.info("👆 Upload a file to get started. The dataset should contain columns like: "
                "location, cost, cuisine, votes (online_order and table_booking are optional).")


# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — DATA ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────
elif nav == "📊 Data Analysis":
    st.markdown("# 📊 Data Analysis")
    st.markdown("Upload a dataset for visual analysis, or explore the training data.")
    st.divider()

    use_demo = st.checkbox("Use training dataset (zomato.csv) for analysis", value=True)

    df_analysis = None

    if use_demo:
        try:
            df_analysis = pd.read_csv("zomato.csv")
            df_analysis.drop(columns=[c for c in df_analysis.columns if 'Unnamed' in c], inplace=True)
            df_analysis.rename(columns={
                'rate (out of 5)': 'rating', 'num of ratings': 'votes',
                'avg cost (two people)': 'cost', 'cuisines type': 'cuisine',
                'area': 'location', 'local address': 'local_address',
            }, inplace=True)
            df_analysis['rating'] = pd.to_numeric(df_analysis['rating'], errors='coerce')
            df_analysis['votes']  = pd.to_numeric(df_analysis['votes'],  errors='coerce')
            df_analysis['cost']   = pd.to_numeric(df_analysis['cost'],   errors='coerce')
        except:
            st.warning("zomato.csv not found next to app.py. Please upload a file.")
    else:
        af = st.file_uploader("Upload CSV / Excel for analysis", type=["csv","xlsx","xls"])
        if af:
            df_analysis = pd.read_csv(af) if af.name.endswith('.csv') else pd.read_excel(af)

    if df_analysis is not None:
        n_rows, n_cols = df_analysis.shape
        b1, b2, b3 = st.columns(3)
        b1.markdown(f'<div class="metric-card"><div class="label">Rows</div><div class="value">{n_rows:,}</div></div>', unsafe_allow_html=True)
        b2.markdown(f'<div class="metric-card"><div class="label">Columns</div><div class="value">{n_cols}</div></div>', unsafe_allow_html=True)
        if 'rating' in df_analysis.columns:
            b3.markdown(f'<div class="metric-card"><div class="label">Avg Rating</div><div class="value">{df_analysis["rating"].mean():.2f}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        show_analysis_charts(df_analysis)
    else:
        st.info("Select 'Use training dataset' or upload a file to see charts.")


