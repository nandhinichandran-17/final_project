import pandas as pd
import numpy as np
import pickle
from scipy import stats
import streamlit as st
import seaborn as sb
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
import warnings
warnings.filterwarnings('ignore')
from surprise import Dataset, Reader
import base64
import plotly.express as px
import plotly.graph_objects as go

# Load models and encoders
with open("/Users/nandhinichandran/Downloads/Finalproject/rf_cl.pkl", "rb") as f:
    model_rf = pickle.load(f)

with open("/Users/nandhinichandran/Downloads/Finalproject/l_e_credict.pkl", "rb") as f:
    le_credict = pickle.load(f)

with open("/Users/nandhinichandran/Downloads/Finalproject/l_e_loan.pkl", "rb") as f:
    le_loan = pickle.load(f)

with open("/Users/nandhinichandran/Downloads/Finalproject/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open('/Users/nandhinichandran/Downloads/Finalproject/scaler_kmean.pkl', 'rb') as f:
    scaler_kmean = pickle.load(f)

with open('/Users/nandhinichandran/Downloads/Finalproject/kmeans.pkl', 'rb') as f:
    kmeans = pickle.load(f)

with open('/Users/nandhinichandran/Downloads/Finalproject/l_e_t_t.pkl', 'rb') as f:
    l_e_t_t = pickle.load(f)

with open("/Users/nandhinichandran/Downloads/Finalproject/knn_pro.pkl", "rb") as f:
    model_recom = pickle.load(f)

# Load dataset
df = pd.read_csv(r"/Users/nandhinichandran/Downloads/Finalproject/dataset.csv")

# Product mapping
product_mapping = {
    'P01': 'Basic Checking Account',
    'P02': 'Premium Checking Account',
    'P03': 'High-Yield Savings Account',
    'P04': 'Money Market Account',
    'P05': 'Standard Credit Card',
    'P06': 'Gold Credit Card',
    'P07': 'Platinum Credit Card',
    'P08': 'Business Credit Card',
    'P09': 'Personal Loan',
    'P010': 'Home Loan',
    'P011': 'Car Loan',
    'P012': 'Education Loan',
    'P013': 'Mortgage',
    'P014': 'Personal Loan',
    'P015': 'Fixed Deposit',
    'P016': 'Recurring Deposit',
    'P017': 'Investment Fund',
    'P018': 'Car Loan',
    'P019': 'Home Equity Loan',
    'P020': 'Gold Loan',
    'P021': 'Travel Insurance',
    'P022': 'Health Insurance',
    'P023': 'Life Insurance',
    'P024': 'Pet Insurance',
    'P025': 'Business Loan',
    'P026': 'Overdraft Protection',
    'P027': 'Wealth Management Service',
    'P028': 'Retirement Account',
    'P029': 'Savings Account',
    'P030': 'Gold Loan',
    'P031': 'Student Loan',
    'P032': 'Credit Line',
    'P033': 'Investment Advisory',
    'P034': 'Fixed Deposit',
    'P035': 'Trust Services',
    'P036': 'Real Estate Investment',
    'P037': 'Online Savings Account',
    'P038': 'Premium Savings Account',
    'P039': 'Cash Management Account',
    'P040': 'Luxury Credit Card',
    'P041': 'Gold Investment',
    'P042': 'Mutual Fund',
    'P043': 'Bonds',
    'P044': 'Stocks',
    'P045': 'Foreign Exchange Services',
    'P046': 'Financial Planning',
    'P047': 'Estate Planning',
    'P048': 'Long-Term Care Insurance',
    'P049': 'Short-Term Investment',
    'P050': 'Tax Planning'
}

df["Product_Name"] = df["Product_Id"].map(product_mapping)

interaction_type_mapping = {'Viewed': 1, 'Clicked': 2, 'Purchased': 3}
df["Interaction_Type"] = df["Interaction_Type"].map(interaction_type_mapping)

reader = Reader(rating_scale=(1, 3))

def recommend_products(customer_id, model, interaction_data, product_mapping, n=5):
    all_products = set(interaction_data['Product_Id'].unique())
    interacted_products = set(interaction_data[interaction_data['Customer_Id'] == customer_id]['Product_Id'])
    products_to_predict = list(all_products - interacted_products)
    predictions = [model.predict(customer_id, product_id) for product_id in products_to_predict]
    top_n_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)[:n]
    recommended_product_ids = [pred.iid for pred in top_n_predictions]
    recommended_products = pd.DataFrame({
        'Product_Id': recommended_product_ids,
        'Product_Name': [product_mapping.get(pid, 'Unknown') for pid in recommended_product_ids]
    })
    return recommended_products

# Streamlit configuration
st.set_page_config(page_title="🏦 Banking System App", layout="wide")

st.markdown(
    "<h2 style='color: #FFFFFF;'>PREDICTIVE ANALYTICS AND RECOMMENDATION SYSTEM IN BANKING</h2>",
    unsafe_allow_html=True
)
# Custom CSS Styling
st.markdown(
    """
    <style>
    body {
        background-color: #f7f9fc;
        color: #2c3e50;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .sidebar .sidebar-content {
        background-color: #2c3e50;
        color: #ffffff;
    }
    .stButton>button {
        background-color: #4caf50;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .step-title {
        font-size: 28px;
        color: #2c3e50;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    .step-container {
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        background-color: #ffffff;
        margin: 20px;
    }
    .sidebar-title {
        font-size: 22px;
        font-weight: bold;
        text-align: center;
        padding: 10px;
        background-color: #4caf50;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar Navigation
with st.sidebar:
    st.markdown("<div class='sidebar-title'>Navigation</div>", unsafe_allow_html=True)
    selected = option_menu("Steps", ["Loan Default Prediction", "Customer Segmentation", "Product Recommendations","Analysis"],
                          icons=["exclamation-triangle", "people", "star","bar-chart"],
                          default_index=0, orientation="vertical")

# Step 1: Loan Default Prediction
if selected == "Loan Default Prediction":
    st.markdown("<div class='step-title'>Loan Default Prediction</div>", unsafe_allow_html=True)
    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        age=st.number_input("Age",min_value=18,max_value=80,value=25)
        monthly_income=st.number_input("Monthly Income",min_value=10000,max_value=500000,value=15000)
        credit_score=st.number_input("Credit Score",min_value=300,max_value=900,value=350)
        credit_score_band=st.selectbox("Credit Score Band",["Poor","Fair","Good","Excellent"])
        loan_amount=st.number_input("Loan Amount",min_value=100000,max_value=2500000,value=500000)
        interest_rate=st.number_input("Interest Rate",min_value=1.0,max_value=15.0,value=5.0)
        loan_term=st.number_input("Loan Term in months",min_value=12,max_value=360,value=36)
        loan_type=st.selectbox("Loan Type",["Personal","Business","Education","Auto","Mortgage"])
        debt_income=st.number_input("Debt Income",min_value=-1.0,max_value=50.0,value=1.0)

        #debt_income_log=np.log1p(debt_income)


        if st.button("Predict Loan Default", key="loan_default_predict"):
            input_data=pd.DataFrame({
                "Age":[age],
                "Monthly_Income":[monthly_income],
                "Credit_Score":[credit_score],
                "Credit_Score_Band":[credit_score_band],
                "Loan_Amount":[loan_amount],
                "Interest_Rate":[interest_rate],
                "Loan_Term":[loan_term],
                "Loan_Type":[loan_type],
                "Debt_Income":[debt_income]
            })
            input_data["Debt_Income_log"]=np.log1p(input_data["Debt_Income"])

            input_data["Credit_Score_Band"]=le_credict.transform(input_data["Credit_Score_Band"])
            input_data["Loan_Type"]=le_loan.transform(input_data["Loan_Type"])

            input_sample=input_data[["Age","Monthly_Income","Credit_Score","Credit_Score_Band","Loan_Amount","Interest_Rate","Loan_Term","Loan_Type","Debt_Income_log"]]

            input_sample_scaled=scaler.transform(input_sample)
            prediction=model_rf.predict(input_sample_scaled)

            if prediction[0]==1:
                st.markdown(
                f"<h3 style='color: #FF0000;'>⚠️ High probability of Default</h3>",
                unsafe_allow_html=True
                )
            else:
                st.markdown(
                f"<h3 style='color: #00FF00;'>✅ Low probability of Default</h3>",
                unsafe_allow_html=True
                )

# Step 2: Customer Segmentation
elif selected == "Customer Segmentation":
    st.markdown("<div class='step-title'>Customer Segmentation</div>", unsafe_allow_html=True)
    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            try:
                transaction_amount = float(st.text_input("Transaction Amount", "1.0"))
                transaction_frequency = float(st.text_input("Transaction Frequency", "1.0"))
            except ValueError:
                st.error("Please enter valid numbers for transaction amount and frequency.")
                transaction_amount = 1.0
                transaction_frequency = 0.0
            
            transaction_type = st.selectbox("Transaction_Type", ["Deposit", "Withdrawal"])


            if st.button("Predict Customer Segment", key="customer_segmentation_predict"):
                transaction_type_encoded=l_e_t_t.transform([transaction_type])[0]
                input_data={"Transaction_Amount":[transaction_amount],
                        "Transaction_Frequency":[transaction_frequency],
                        "Transaction_Type_Encoded":[transaction_type_encoded]
                        }

                input_df=pd.DataFrame(input_data)

                input_scaled_data=scaler_kmean.transform(input_df)

                predicted_customer=kmeans.predict(input_scaled_data)

                df["Transaction_Type_Encoded"] = l_e_t_t.transform(df["Transaction_Type"])
                df_scaled = scaler_kmean.transform(df[["Transaction_Amount", "Transaction_Frequency", "Transaction_Type_Encoded"]])
                df["Clusters"] = kmeans.fit_predict(df_scaled)


                st.markdown(
                f"<h3 style='color: #FFFFFF;'>The customer belongs to cluster: {predicted_customer[0]}</h3>",
                unsafe_allow_html=True
                )
            
                cluster_summary=df.groupby("Clusters").agg(
                    Average_Transaction_Amount=("Transaction_Amount","mean"),
                    Average_Transaction_Frequency=("Transaction_Frequency","mean"),
                
                ).reset_index()

                st.markdown(
                    f"<h3 style='color: #D3D3D3;'>CLUSTER_SUMMARY</h3>",
                    unsafe_allow_html=True
                )
                fig, ax = plt.subplots(figsize=(12, 8))

                heatmap_data = cluster_summary.set_index('Clusters').T
                sb.heatmap(heatmap_data, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
                plt.title("Cluster Summary")
                plt.xlabel("Cluster")
                st.pyplot(fig)

                

# Step 3: Product Recommendations
elif selected == "Product Recommendations":
    st.markdown("<div class='step-title'>Product Recommendations</div>", unsafe_allow_html=True)
    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            st.subheader("Personalized Product Recommendations")

            customer_id = st.text_input("Enter Customer ID", "")

            if st.button("Get Recommendations", key="product_recommendations_predict"):
                if customer_id:
                    if customer_id in df['Customer_Id'].values:
                        recommendations = recommend_products(customer_id, model_recom, df, product_mapping)

                        if not recommendations.empty:
                            st.markdown(
                                f"<h3 style='color: #FFFFFF;'>Top Recommended Products for {customer_id}:</h3>",
                                unsafe_allow_html=True
                            )
                            st.dataframe(recommendations)
                        else:
                            st.write("No recommendations available.")
                    else:
                        st.error("Invalid Customer ID. Please enter a valid Customer ID.")
                else:
                    st.warning("Please enter a valid Customer ID.")

# Step 4: Analysis Section
if selected == "Analysis":
    st.markdown("<div class='step-title'>Analysis</div>", unsafe_allow_html=True)
    with st.container():
        st.title("Interactive Data Visualization")

        # Upload dataset
        uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type="csv")

        if uploaded_file is not None:  # Check if a file is uploaded
            df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
            
            # Display the first few rows of the dataset
            st.write("Here are the first few rows of your dataset:")
            st.dataframe(df.head())

            # Sidebar options for analysis
            st.sidebar.header("Select Analysis Type")
            analysis_choice = st.sidebar.radio(
                "Choose the type of analysis:",
                ("EDA Plots", "Categorical Analysis", "Numerical Analysis", "Correlation", "Live Dashboard")
            )

            if analysis_choice == "EDA Plots":
                st.subheader("EDA Visualizations")

                # Loan Type Pie Chart
                if "Loan_Type" in df.columns:
                    st.write("### Loan Type Distribution")
                    fig = px.pie(
                        df,
                        names="Loan_Type",
                        title="Loan Type Distribution",
                        color_discrete_sequence=px.colors.sequential.RdBu,
                        hole=0.4
                    )
                    st.plotly_chart(fig)

                # Monthly Income Histogram
                if "Monthly_Income" in df.columns:
                    st.write("### Distribution of Monthly Income")
                    fig = px.histogram(
                        df,
                        x="Monthly_Income",
                        nbins=30,
                        title="Monthly Income Distribution",
                        color_discrete_sequence=["#636EFA"]
                    )
                    fig.update_layout(bargap=0.2)
                    st.plotly_chart(fig)

            elif analysis_choice == "Categorical Analysis":
                st.subheader("Categorical Analysis")
                cat_columns = [col for col in df.select_dtypes(include="object").columns]

                for col in cat_columns:
                    st.write(f"### Distribution of {col}")
                    # Prepare data for Plotly bar chart
                    cat_counts = df[col].value_counts().reset_index()  # Create a new DataFrame with counts
                    cat_counts.columns = [col, "count"]  # Rename columns for clarity

                # Plot using Plotly
                    fig = px.bar(
                        cat_counts,
                        x=col,
                        y="count",
                        text="count",
                        color=col,
                        title=f"{col} Distribution",
                        color_discrete_sequence=px.colors.qualitative.Set2
                        )

                # Enhance visuals
                    fig.update_traces(textposition="outside")
                    fig.update_layout(
                        xaxis_title=col,
                        yaxis_title="Count",
                        showlegend=False,
                        title_x=0.5
                        )

                    st.plotly_chart(fig)


            elif analysis_choice == "Numerical Analysis":
                st.subheader("Numerical Analysis")
                num_columns = [col for col in df.select_dtypes(include="number").columns]

                for col in num_columns:
                    st.write(f"### Distribution of {col}")
                    fig = px.histogram(
                        df,
                        x=col,
                        nbins=30,
                        title=f"{col} Distribution ",
                        color_discrete_sequence=["#AB63FA"],
                        animation_frame="Transaction_Year" if "Transaction_Year" in df.columns else None
                    )
                    st.plotly_chart(fig)

            elif analysis_choice == "Correlation":
                st.subheader("Correlation Heatmap")
                num_columns = [col for col in df.select_dtypes(include="number").columns]
                if len(num_columns) > 1:
                    co_mat = df[num_columns].corr()
                    fig = px.imshow(
                        co_mat,
                        text_auto=True,
                        color_continuous_scale=px.colors.sequential.Viridis,
                        title="Correlation Heatmap"
                    )
                    st.plotly_chart(fig)

            elif analysis_choice == "Live Dashboard":
                st.subheader("Live Dashboard")
                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Total Transactions", df.shape[0])
                    if "Loan_Amount" in df.columns:
                        st.metric("Average Loan Amount", f"${df['Loan_Amount'].mean():,.2f}")

                with col2:
                    if "Default_Flag" in df.columns:
                        default_rate = (df["Default_Flag"].mean() * 100)
                        st.metric("Default Rate", f"{default_rate:.2f}%")

                if "Loan_Amount" in df.columns and "Credit_Score" in df.columns:
                    fig = px.scatter(
                        df,
                        x="Credit_Score",
                        y="Loan_Amount",
                        color="Default_Flag" if "Default_Flag" in df.columns else None,
                        size="Loan_Amount",
                        hover_data=["Credit_Score"],
                        title="Loan Amount vs Credit Score",
                        color_continuous_scale=px.colors.sequential.Plasma
                    )
                    st.plotly_chart(fig)
