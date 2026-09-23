import streamlit as st
import pandas as pd
import joblib


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Manufacturing Output Predictor",
    page_icon="🏭",
    layout="wide"
)


# ==========================================================
# CONSTANTS
# ==========================================================

MODEL_FILES = {
    "Linear Regression": "manufacturing_linear_regression.pkl",
    "Ridge Regression": "manufacturing_ridge_model.pkl",
    "Lasso Regression": "manufacturing_lasso_model.pkl"
}

REQUIRED_COLUMNS = [
    "Injection_Temperature",
    "Injection_Pressure",
    "Cycle_Time",
    "Cooling_Time",
    "Material_Viscosity",
    "Ambient_Temperature",
    "Machine_Age",
    "Operator_Experience",
    "Maintenance_Hours",
    "Shift",
    "Machine_Type",
    "Material_Grade",
    "Day_of_Week",
    "Temperature_Pressure_Ratio",
    "Total_Cycle_Time",
    "Efficiency_Score",
    "Machine_Utilization",
    "Hour",
    "Month",
    "Day"
]


# ==========================================================
# SIDEBAR - MODEL INFORMATION
# ==========================================================

st.sidebar.title("🏭 Model Information")

st.sidebar.write(
    "**Project:** Manufacturing Equipment Output Prediction"
)

st.sidebar.write(
    "**Task:** Supervised Learning - Regression"
)

st.sidebar.write(
    "**Target:** Parts_Per_Hour"
)

st.sidebar.write(
    "**Models:** Linear / Ridge / Lasso"
)

st.sidebar.write(
    "**Ensemble:** Average of three model predictions"
)

st.sidebar.write(
    "**Training Split:** 80%"
)

st.sidebar.write(
    "**Testing Split:** 20%"
)


# ==========================================================
# LOAD THREE TRAINED MODELS
# ==========================================================

@st.cache_resource
def load_models():

    linear_model = joblib.load(
        MODEL_FILES["Linear Regression"]
    )

    ridge_model = joblib.load(
        MODEL_FILES["Ridge Regression"]
    )

    lasso_model = joblib.load(
        MODEL_FILES["Lasso Regression"]
    )

    return (
        linear_model,
        ridge_model,
        lasso_model
    )


try:

    (
        linear_model,
        ridge_model,
        lasso_model
    ) = load_models()

except Exception as e:

    st.error(
        "❌ Failed to load trained models."
    )

    st.exception(e)

    st.stop()


# ==========================================================
# TITLE
# ==========================================================

st.title(
    "🏭 Manufacturing Equipment Output Prediction"
)

st.write(
    "Predict hourly machine output using "
    "manufacturing operating parameters."
)

st.divider()


# ==========================================================
# MACHINE INPUT SECTION
# ==========================================================

st.subheader("⚙️ Machine Parameters")

col1, col2, col3 = st.columns(3)


# ----------------------------------------------------------
# COLUMN 1
# ----------------------------------------------------------

with col1:

    injection_temperature = st.number_input(
        "Injection Temperature (°C)",
        min_value=150.0,
        max_value=300.0,
        value=220.0
    )

    injection_pressure = st.number_input(
        "Injection Pressure (bar)",
        min_value=50.0,
        max_value=200.0,
        value=120.0
    )

    cycle_time = st.number_input(
        "Cycle Time (sec)",
        min_value=5.0,
        max_value=60.0,
        value=30.0
    )

    cooling_time = st.number_input(
        "Cooling Time (sec)",
        min_value=5.0,
        max_value=30.0,
        value=12.0
    )


# ----------------------------------------------------------
# COLUMN 2
# ----------------------------------------------------------

with col2:

    material_viscosity = st.number_input(
        "Material Viscosity",
        min_value=50.0,
        max_value=500.0,
        value=250.0
    )

    ambient_temperature = st.number_input(
        "Ambient Temperature (°C)",
        min_value=10.0,
        max_value=40.0,
        value=24.0
    )

    machine_age = st.number_input(
        "Machine Age (years)",
        min_value=0.0,
        max_value=30.0,
        value=5.0
    )

    operator_experience = st.number_input(
        "Operator Experience (months)",
        min_value=0.0,
        max_value=240.0,
        value=24.0
    )


# ----------------------------------------------------------
# COLUMN 3
# ----------------------------------------------------------

with col3:

    maintenance_hours = st.number_input(
        "Maintenance Hours",
        min_value=0.0,
        max_value=500.0,
        value=50.0
    )

    shift = st.selectbox(
        "Shift",
        [
            "Day",
            "Evening",
            "Night"
        ]
    )

    machine_type = st.selectbox(
        "Machine Type",
        [
            "Type_A",
            "Type_B"
        ]
    )

    material_grade = st.selectbox(
        "Material Grade",
        [
            "Economy",
            "Standard",
            "Premium"
        ]
    )


# ==========================================================
# ADDITIONAL INPUTS
# ==========================================================

col4, col5, col6, col7 = st.columns(4)


with col4:

    day_of_week = st.selectbox(
        "Day of Week",
        [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday"
        ]
    )


with col5:

    hour = st.number_input(
        "Hour",
        min_value=0,
        max_value=23,
        value=12
    )


with col6:

    month = st.number_input(
        "Month",
        min_value=1,
        max_value=12,
        value=1
    )


with col7:

    day = st.number_input(
        "Day",
        min_value=1,
        max_value=31,
        value=1
    )


# ==========================================================
# ENGINEERED FEATURES USED DURING TRAINING
# ==========================================================

st.subheader(
    "🧮 Engineered Features"
)

st.info(
    "These four engineered features are part of the trained model input. "
    "Enter values from the dataset when possible. They are not "
    "recalculated here because the exact training formulas were not "
    "provided."
)

col8, col9, col10, col11 = st.columns(4)


with col8:

    temperature_pressure_ratio = st.number_input(
        "Temperature Pressure Ratio",
        value=1.83,
        format="%.4f"
    )


with col9:

    total_cycle_time = st.number_input(
        "Total Cycle Time",
        value=42.0,
        format="%.2f"
    )


with col10:

    efficiency_score = st.number_input(
        "Efficiency Score",
        value=20.0,
        format="%.4f"
    )


with col11:

    machine_utilization = st.number_input(
        "Machine Utilization",
        value=0.0238,
        format="%.6f"
    )


# ==========================================================
# SINGLE MACHINE PREDICTION
# ==========================================================

st.divider()

st.subheader(
    "🔮 Output Prediction"
)


# ==========================================================
# MODEL SELECTION
# ==========================================================

selected_model = st.selectbox(
    "🤖 Select Prediction Model",
    [
        "Linear Regression",
        "Ridge Regression",
        "Lasso Regression",
        "Ensemble"
    ]
)


if st.button(
    "🔮 Predict Parts Per Hour",
    width="stretch"
):

    # ------------------------------------------------------
    # CREATE INPUT DATAFRAME - ALL 20 TRAINED FEATURES
    # ------------------------------------------------------

    input_data = pd.DataFrame({

        "Injection_Temperature":
            [injection_temperature],

        "Injection_Pressure":
            [injection_pressure],

        "Cycle_Time":
            [cycle_time],

        "Cooling_Time":
            [cooling_time],

        "Material_Viscosity":
            [material_viscosity],

        "Ambient_Temperature":
            [ambient_temperature],

        "Machine_Age":
            [machine_age],

        "Operator_Experience":
            [operator_experience],

        "Maintenance_Hours":
            [maintenance_hours],

        "Shift":
            [shift],

        "Machine_Type":
            [machine_type],

        "Material_Grade":
            [material_grade],

        "Day_of_Week":
            [day_of_week],

        "Temperature_Pressure_Ratio":
            [temperature_pressure_ratio],

        "Total_Cycle_Time":
            [total_cycle_time],

        "Efficiency_Score":
            [efficiency_score],

        "Machine_Utilization":
            [machine_utilization],

        "Hour":
            [hour],

        "Month":
            [month],

        "Day":
            [day]
    })


    # ------------------------------------------------------
    # GENERATE SELECTED MODEL PREDICTION
    # ------------------------------------------------------

    try:

        if selected_model == "Linear Regression":

            prediction = linear_model.predict(
                input_data
            )[0]


        elif selected_model == "Ridge Regression":

            prediction = ridge_model.predict(
                input_data
            )[0]


        elif selected_model == "Lasso Regression":

            prediction = lasso_model.predict(
                input_data
            )[0]


        else:

            linear_prediction = linear_model.predict(
                input_data
            )[0]

            ridge_prediction = ridge_model.predict(
                input_data
            )[0]

            lasso_prediction = lasso_model.predict(
                input_data
            )[0]

            prediction = (
                linear_prediction +
                ridge_prediction +
                lasso_prediction
            ) / 3


        prediction = float(
            prediction
        )


        # --------------------------------------------------
        # SUCCESS MESSAGE
        # --------------------------------------------------

        st.success(
            f"Prediction generated using: {selected_model}"
        )


        # --------------------------------------------------
        # PREDICTION RESULT
        # --------------------------------------------------

        st.metric(
            "Predicted Parts Per Hour",
            f"{prediction:.2f}"
        )


        # --------------------------------------------------
        # ENSEMBLE INDIVIDUAL PREDICTIONS
        # --------------------------------------------------

        if selected_model == "Ensemble":

            st.subheader(
                "🔬 Individual Model Predictions"
            )

            ensemble_col1, ensemble_col2, ensemble_col3 = (
                st.columns(3)
            )

            with ensemble_col1:

                st.metric(
                    "Linear Regression",
                    f"{linear_prediction:.2f}"
                )

            with ensemble_col2:

                st.metric(
                    "Ridge Regression",
                    f"{ridge_prediction:.2f}"
                )

            with ensemble_col3:

                st.metric(
                    "Lasso Regression",
                    f"{lasso_prediction:.2f}"
                )


        # --------------------------------------------------
        # PREDICTION INTERPRETATION
        # --------------------------------------------------

        st.subheader(
            "📊 Prediction Interpretation"
        )


        if prediction < 25:

            st.warning(
                "The predicted output is relatively low. "
                "Review cycle time, cooling time, "
                "maintenance and machine utilization."
            )

        elif prediction < 45:

            st.info(
                "The predicted output is within a "
                "moderate range for this application."
            )

        else:

            st.success(
                "The predicted output is relatively high "
                "for the defined application range."
            )


        # --------------------------------------------------
        # INPUT SUMMARY
        # --------------------------------------------------

        st.subheader(
            "📋 Input Summary"
        )

        st.write(
            "The prediction was generated using "
            "the following machine parameters:"
        )

        # Convert the transposed input summary to a clean
        # two-column dataframe so Streamlit/PyArrow does not
        # encounter mixed numeric/string types in one column.
        input_summary = (
            input_data.T
            .reset_index()
        )

        input_summary.columns = [
            "Feature",
            "Value"
        ]

        input_summary["Value"] = (
            input_summary["Value"]
            .astype(str)
        )

        st.dataframe(
            input_summary,
            width="stretch",
            hide_index=True
        )


    except Exception as e:

        st.error(
            "❌ Prediction failed."
        )

        st.exception(e)


# ==========================================================
# MODEL PERFORMANCE COMPARISON
# ==========================================================

st.divider()

st.subheader(
    "📊 Model Performance Comparison"
)

try:

    comparison_df = pd.read_csv(
        "model_comparison.csv"
    )

    st.dataframe(
        comparison_df,
        width="stretch"
    )

except FileNotFoundError:

    st.warning(
        "model_comparison.csv was not found."
    )

except Exception as comparison_error:

    st.error(
        "❌ Unable to read model_comparison.csv."
    )

    st.exception(comparison_error)


# ==========================================================
# BATCH CSV PREDICTION
# ==========================================================

st.divider()

st.subheader(
    "📂 Batch Prediction"
)

st.write(
    "Upload the original manufacturing dataset "
    "to generate batch predictions."
)


uploaded_file = st.file_uploader(
    "Upload Manufacturing CSV",
    type=["csv"],
    key="batch_csv"
)


if uploaded_file is not None:

    try:

        # --------------------------------------------------
        # READ CSV
        # --------------------------------------------------

        uploaded_df = pd.read_csv(
            uploaded_file
        )


        # --------------------------------------------------
        # DISPLAY UPLOADED DATA
        # --------------------------------------------------

        st.write(
            "### 📄 Uploaded Data"
        )

        st.dataframe(
            uploaded_df.head(20).astype(str),
            width="stretch"
        )

        st.write(
            f"Total records uploaded: **{len(uploaded_df)}**"
        )


        # --------------------------------------------------
        # SELECT BATCH MODEL
        # --------------------------------------------------

        batch_model = st.selectbox(
            "🤖 Select Batch Prediction Model",
            [
                "Linear Regression",
                "Ridge Regression",
                "Lasso Regression",
                "Ensemble"
            ],
            key="batch_model"
        )


        # --------------------------------------------------
        # GENERATE BATCH PREDICTIONS
        # --------------------------------------------------

        if st.button(
            "🚀 Generate Batch Predictions",
            width="stretch"
        ):

            prediction_df = uploaded_df.copy()


            # ==================================================
            # STEP 1: HANDLE TIMESTAMP
            # ==================================================

            if "Timestamp" in prediction_df.columns:

                prediction_df["Timestamp"] = pd.to_datetime(
                    prediction_df["Timestamp"],
                    errors="coerce"
                )

                prediction_df["Hour"] = (
                    prediction_df["Timestamp"].dt.hour
                )

                prediction_df["Month"] = (
                    prediction_df["Timestamp"].dt.month
                )

                prediction_df["Day"] = (
                    prediction_df["Timestamp"].dt.day
                )

                prediction_df.drop(
                    columns=["Timestamp"],
                    inplace=True
                )


            # ==================================================
            # STEP 2: REMOVE TARGET
            # ==================================================

            if "Parts_Per_Hour" in prediction_df.columns:

                prediction_df.drop(
                    columns=["Parts_Per_Hour"],
                    inplace=True
                )


            # ==================================================
            # STEP 3: HANDLE MISSING VALUES
            # ==================================================

            numerical_columns = [
                "Material_Viscosity",
                "Ambient_Temperature",
                "Operator_Experience"
            ]

            for column in numerical_columns:

                if column in prediction_df.columns:

                    prediction_df[column] = (
                        prediction_df[column]
                        .fillna(
                            prediction_df[column].median()
                        )
                    )


            # Handle missing categorical values without
            # inventing new categories.

            categorical_columns = [
                "Shift",
                "Machine_Type",
                "Material_Grade",
                "Day_of_Week"
            ]

            for column in categorical_columns:

                if column in prediction_df.columns:

                    mode_values = prediction_df[column].mode()

                    if not mode_values.empty:

                        prediction_df[column] = (
                            prediction_df[column]
                            .fillna(mode_values.iloc[0])
                        )


            # ==================================================
            # STEP 4: CHECK REQUIRED FEATURES
            # ==================================================

            missing_columns = [

                column

                for column in REQUIRED_COLUMNS

                if column not in prediction_df.columns

            ]


            if missing_columns:

                st.error(
                    "❌ Required model features are missing."
                )

                st.write(
                    "Missing columns:"
                )

                st.code(
                    "\n".join(missing_columns)
                )

                st.stop()


            # ==================================================
            # STEP 5: SELECT EXACT FEATURE ORDER
            # ==================================================

            prediction_input = prediction_df[
                REQUIRED_COLUMNS
            ].copy()


            # ==================================================
            # STEP 6: GENERATE SELECTED BATCH PREDICTIONS
            # ==================================================

            try:

                if batch_model == "Linear Regression":

                    predictions = linear_model.predict(
                        prediction_input
                    )


                elif batch_model == "Ridge Regression":

                    predictions = ridge_model.predict(
                        prediction_input
                    )


                elif batch_model == "Lasso Regression":

                    predictions = lasso_model.predict(
                        prediction_input
                    )


                else:

                    linear_predictions = (
                        linear_model.predict(
                            prediction_input
                        )
                    )

                    ridge_predictions = (
                        ridge_model.predict(
                            prediction_input
                        )
                    )

                    lasso_predictions = (
                        lasso_model.predict(
                            prediction_input
                        )
                    )

                    predictions = (
                        linear_predictions +
                        ridge_predictions +
                        lasso_predictions
                    ) / 3


                # ==================================================
                # STEP 7: CREATE RESULT DATAFRAME
                # ==================================================

                result_df = uploaded_df.copy()

                result_df[
                    "Predicted_Parts_Per_Hour"
                ] = predictions


                # ==================================================
                # SUCCESS
                # ==================================================

                st.success(
                    f"✅ Batch prediction completed using: {batch_model}"
                )


                # ==================================================
                # ENSEMBLE BATCH DETAILS
                # ==================================================

                if batch_model == "Ensemble":

                    ensemble_batch_df = pd.DataFrame({

                        "Linear Regression":
                            linear_predictions,

                        "Ridge Regression":
                            ridge_predictions,

                        "Lasso Regression":
                            lasso_predictions,

                        "Ensemble Average":
                            predictions

                    })

                    st.subheader(
                        "🔬 Batch Individual Model Predictions"
                    )

                    st.dataframe(
                        ensemble_batch_df,
                        width="stretch"
                    )


                # ==================================================
                # DISPLAY RESULTS
                # ==================================================

                st.subheader(
                    "📊 Prediction Results"
                )

                st.dataframe(
                    result_df.astype(str),
                    width="stretch"
                )


                # ==================================================
                # DOWNLOAD RESULTS
                # ==================================================

                csv_data = result_df.to_csv(
                    index=False
                ).encode("utf-8")


                st.download_button(
                    label="⬇️ Download Prediction Results",
                    data=csv_data,
                    file_name=(
                        "manufacturing_predictions.csv"
                    ),
                    mime="text/csv",
                    width="stretch"
                )


            except Exception as prediction_error:

                st.error(
                    "❌ Model prediction failed."
                )

                st.exception(
                    prediction_error
                )


    except Exception as upload_error:

        st.error(
            "❌ Unable to process the uploaded CSV."
        )

        st.exception(
            upload_error
        )


# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.caption(
    "Manufacturing Equipment Output Prediction | "
    "Linear Regression, Ridge Regression, Lasso Regression "
    "and inference-time Ensemble Averaging"
)
