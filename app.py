import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

try:
    from src.preprocessing import engineer_features
except ImportError:
    st.error("⚠️ Error: 'src/preprocessing.py' nahi mila.")
    st.stop()

@st.cache_resource
def load_model():
    try:
        return joblib.load('models/model.joblib')
    except FileNotFoundError:
        st.error("⚠️ Error: Model file not found.")
        st.stop()

model = load_model()

@st.cache_resource
def get_shap_explainer(_model):
    # Access the actual XGBoost regressor inside the pipeline
    regressor = _model.named_steps['regressor']
    return shap.TreeExplainer(regressor)

# 2. STREAMLIT APP SETUP
st.set_page_config(page_title="Dream House Predictor", page_icon="🏡", layout="centered")

st.markdown("<h1 style='text-align: center; color: #2E86C1;'>🏡 Dream House Price Prediction</h1>", unsafe_allow_html=True)
# st.markdown("<p style='text-align: center;'>Fill in the <b>Required Fields (*)</b> to get an instant valuation.</p>", unsafe_allow_html=True)
st.markdown("---")

with st.form("prediction_form"):
    
    st.subheader("🌟 Essential Information (Required)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        neighborhood = st.selectbox("Neighborhood *", 
            ['NAmes', 'CollgCr', 'OldTown', 'Edwards', 'Somerst', 
             'NridgHt', 'Gilbert', 'Sawyer', 'NWAmes', 'SawyerW', 
             'Mitchel', 'BrkSide', 'Crawfor', 'IDOTRR', 'NoRidge', 
             'Timber', 'StoneBr', 'SWISU', 'ClearCr', 'MeadowV', 
             'BrDale', 'Blmngtn', 'Veenker', 'NPkVill', 'Blueste'])
        
        gr_liv_area = st.number_input("Living Area (sq ft) *", 500, 5000, 1500)
        
        overall_qual = st.slider("Overall Quality (1-10) *", 1, 10, 5, help="5 is Average, 10 is Luxury")

    with col2:
        year_built = st.number_input("Year Built *", 1870, 2025, 2000)
        
        tot_rms_abv_grd = st.number_input("Total Rooms *", 2, 14, 6)
        garage_cars = st.selectbox("Garage Capacity (Cars) *", [0, 1, 2, 3, 4], index=2)

    st.markdown("---")

    with st.expander("🛠️ Advanced Options (Click to Expand)"):
        st.info("You can leave these as default, or change them for better accuracy.")
        
        adv_col1, adv_col2 = st.columns(2)
        
        with adv_col1:
            lot_area = st.number_input("Lot Area (sq ft)", 1000, 100000, 10000)
            total_bsmt_sf = st.number_input("Basement Area (sq ft)", 0, 3000, 1000)
            bedroom_abv_gr = st.number_input("Bedrooms", 0, 8, 3)
            full_bath = st.number_input("Full Bathrooms", 0, 4, 2)
            half_bath = st.number_input("Half Bathrooms", 0, 2, 1)
            
        with adv_col2:
            ms_zoning = st.selectbox("Zoning", ['RL', 'RM', 'C (all)', 'FV', 'RH'])
            house_style = st.selectbox("House Style", ['1Story', '2Story', '1.5Fin', 'SLvl', 'SFoyer'])
            overall_cond = st.slider("Condition (1-10)", 1, 10, 5)
            year_remod = st.number_input("Year Remodeled", 1950, 2025, 2000)
            garage_area = st.number_input("Garage Area (sq ft)", 0, 1500, 500)

        st.markdown("Price Factors")
        ext_col1, ext_col2 = st.columns(2)
        with ext_col1:
            yr_sold = st.number_input("Year Sold", 2006, 2025, 2010)
        with ext_col2:
            mo_sold = st.selectbox("Month Sold", list(range(1, 13)))
            
        bldg_type = '1Fam' 
        lot_config = 'Inside'

    default_values = {
        'MSSubClass': 20, 'LotFrontage': 69.0, 'Street': 'Pave', 'Alley': 'None', 
        'LotShape': 'Reg', 'LandContour': 'Lvl', 'Utilities': 'AllPub', 'LandSlope': 'Gtl', 
        'Condition1': 'Norm', 'Condition2': 'Norm', 'RoofStyle': 'Gable', 
        'RoofMatl': 'CompShg', 'Exterior1st': 'VinylSd', 'Exterior2nd': 'VinylSd', 
        'MasVnrType': 'None', 'MasVnrArea': 0, 'ExterQual': 'TA', 'ExterCond': 'TA', 
        'Foundation': 'PConc', 'BsmtQual': 'TA', 'BsmtCond': 'TA', 'BsmtExposure': 'No', 
        'BsmtFinType1': 'Unf', 'BsmtFinSF1': 0, 'BsmtFinType2': 'Unf', 'BsmtFinSF2': 0, 
        'BsmtUnfSF': 0, 'Heating': 'GasA', 'HeatingQC': 'Ex', 'CentralAir': 'Y', 
        'Electrical': 'SBrkr', '1stFlrSF': 1000, '2ndFlrSF': 0, 'LowQualFinSF': 0, 
        'BsmtFullBath': 0, 'BsmtHalfBath': 0, 'KitchenAbvGr': 1, 'KitchenQual': 'TA', 
        'Functional': 'Typ', 'Fireplaces': 1, 'FireplaceQu': 'Gd', 'GarageType': 'Attchd', 
        'GarageYrBlt': 2000, 'GarageFinish': 'RFn', 'GarageQual': 'TA', 'GarageCond': 'TA', 
        'PavedDrive': 'Y', 'WoodDeckSF': 0, 'OpenPorchSF': 0, 'EnclosedPorch': 0, 
        '3SsnPorch': 0, 'ScreenPorch': 0, 'PoolArea': 0, 'PoolQC': 'None', 'Fence': 'None', 
        'MiscFeature': 'None', 'MiscVal': 0, 'SaleType': 'WD', 'SaleCondition': 'Normal',
        'BldgType': bldg_type, 'LotConfig': lot_config
    }

    submitted = st.form_submit_button("💰 Predict Value", type="primary")

if submitted:
    errors = []
    if year_remod < year_built:
        errors.append("Year Remodeled cannot be before Year Built.")
    if yr_sold < year_built:
        errors.append("Year Sold cannot be before Year Built.")
    if total_bsmt_sf > gr_liv_area * 2:
        errors.append("Basement area seems unusually large compared to living area — please double-check.")

    if errors:
        for e in errors:
            st.warning(f"⚠️ {e}")
        st.stop()
    user_input = {
        'Neighborhood': neighborhood, 'LotArea': lot_area, 'MSZoning': ms_zoning,
        'HouseStyle': house_style, 'OverallQual': overall_qual, 'OverallCond': overall_cond, 
        'YearBuilt': year_built, 'YearRemodAdd': year_remod, 'GrLivArea': gr_liv_area, 
        'TotalBsmtSF': total_bsmt_sf, 'BedroomAbvGr': bedroom_abv_gr, 'FullBath': full_bath, 
        'HalfBath': half_bath, 'TotRmsAbvGrd': tot_rms_abv_grd, 'GarageCars': garage_cars, 
        'GarageArea': garage_area, 'YrSold': yr_sold, 'MoSold': mo_sold
    }

    full_data_dict = {**default_values, **user_input}
    df_input = pd.DataFrame([full_data_dict])

    try:
        df_processed = engineer_features(df_input)
        log_prediction = model.predict(df_processed)[0]
        price = np.expm1(log_prediction)

        st.success(f"🏡 Estimated Value: **${price:,.2f}**")
        if price > 200000:
            st.balloons()

        # SHAP explainability
        with st.expander("🔍 Why this price? (SHAP Explanation)"):
            with st.spinner("Calculating feature impact..."):
                try:
                    preprocessor = model.named_steps['preprocessor']
                    df_transformed = preprocessor.transform(df_processed)
                    explainer = get_shap_explainer(model)
                    shap_values = explainer.shap_values(df_transformed)

                    feature_names = preprocessor.get_feature_names_out()

                    fig, ax = plt.subplots(figsize=(8, 6))
                    shap.summary_plot(
                        shap_values, df_transformed,
                        feature_names=feature_names,
                        plot_type="bar", show=False, max_display=10
                    )
                    st.pyplot(fig)
                    plt.close(fig)
                except Exception as shap_err:
                    st.warning(f"Could not generate SHAP explanation: {shap_err}")

    except Exception as e:
        st.error(f"Error: {str(e)}")