import streamlit as st
import pandas as pd
import plotly.express as px

from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

#OpenRouter client setup
#tweaked code to deploy to streamlit cloud. 
#It will first look for the API key in Streamlit secrets, and if not found, 
#it will fall back to the .env file.                                                          
openrouter_api_key = st.secrets.get(
    "OPENROUTER_API_KEY",
    os.getenv("OPENROUTER_API_KEY")
)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_api_key,
)




st.set_page_config(
    page_title="AI-Powered E-Commerce Decision Copilot",
    page_icon="🛒",
    layout="wide"
)

st.title("AI E-Commerce Decision Copilot")

@st.cache_data
def load_data():
    return pd.read_csv("ecommerce_cleaned.csv")

df = load_data()


#Setting up the AI
#Feeding AI a summary of the dataset and the context

#This creates a summary and is sent to the AI instead of the full CSV.

def create_data_context(data):
    
    #If the filtered data has no rows return this message
    if data.empty:
        return "No data available after applying the current filters."

    #Make a copy so we don't accidentally change the original dataframe
    data = data.copy()

    #Basic dataset overview
    context = f"""
Dataset Overview:
- Total rows/orders: {len(data)}
- Total columns: {len(data.columns)}
- Columns available: {list(data.columns)}
"""

    #Overall return summary
    if "Return_Status" in data.columns:
        total_orders = len(data)
        total_returns = int(data["Return_Status"].sum())
        non_returns = total_orders - total_returns
        return_rate = data["Return_Status"].mean() * 100

        context += f"""

Return Summary:
- Total orders: {total_orders}
- Returned orders: {total_returns}
- Non-returned orders: {non_returns}
- Overall return rate: {return_rate:.2f}%
"""

    #Product category summary
    if "Product_Category" in data.columns and "Return_Status" in data.columns:
        category_summary = (
            data.groupby("Product_Category")
            .agg(
                Total_Orders=("Return_Status", "count"),
                Returned_Orders=("Return_Status", "sum"),
                Return_Rate_Percent=("Return_Status", lambda x: x.mean() * 100),
                Average_Total_Spend=("Total_Spend", "mean")
            )
            .sort_values("Return_Rate_Percent", ascending=False)
            .round(2)
            .head(10)
        )

        context += f"""

Product Category Summary:
{category_summary.to_dict()}
"""

    #Customer rating summary
    if "Customer_Rating" in data.columns and "Return_Status" in data.columns:
        rating_summary = (
            data.groupby("Customer_Rating")
            .agg(
                Total_Orders=("Return_Status", "count"),
                Returned_Orders=("Return_Status", "sum"),
                Return_Rate_Percent=("Return_Status", lambda x: x.mean() * 100)
            )
            .sort_index()
            .round(2)
        )

        average_rating = data["Customer_Rating"].mean()

        context += f"""

Customer Rating Summary:
{rating_summary.to_dict()}

Average Customer Rating:
{average_rating:.2f}
"""

    #Payment method summary
    if "Payment_Method" in data.columns and "Return_Status" in data.columns:
        payment_summary = (
            data.groupby("Payment_Method")
            .agg(
                Total_Orders=("Return_Status", "count"),
                Returned_Orders=("Return_Status", "sum"),
                Return_Rate_Percent=("Return_Status", lambda x: x.mean() * 100)
            )
            .sort_values("Return_Rate_Percent", ascending=False)
            .round(2)
        )

        context += f"""

Payment Method Summary:
{payment_summary.to_dict()}
"""

    #Age group summary
    if "Age_Group" in data.columns and "Return_Status" in data.columns:
        age_group_summary = (
            data.groupby("Age_Group")
            .agg(
                Total_Orders=("Return_Status", "count"),
                Returned_Orders=("Return_Status", "sum"),
                Return_Rate_Percent=("Return_Status", lambda x: x.mean() * 100)
            )
            .sort_values("Return_Rate_Percent", ascending=False)
            .round(2)
        )

        context += f"""

Age Group Summary:
{age_group_summary.to_dict()}
"""

    #Discount bucket summary
    if "Discount_Bucket" in data.columns and "Return_Status" in data.columns:
        discount_summary = (
            data.groupby("Discount_Bucket")
            .agg(
                Total_Orders=("Return_Status", "count"),
                Returned_Orders=("Return_Status", "sum"),
                Return_Rate_Percent=("Return_Status", lambda x: x.mean() * 100)
            )
            .sort_values("Return_Rate_Percent", ascending=False)
            .round(2)
        )

        context += f"""

Discount Bucket Summary:
{discount_summary.to_dict()}
"""

    #Unit price summary
    if "Unit_Price" in data.columns and "Return_Status" in data.columns:
        unit_price_summary = (
            data.groupby("Return_Status")["Unit_Price"]
            .agg(["count", "mean", "median"])
            .round(2)
        )

        context += f"""

Unit Price Summary by Return Status:
Return_Status 0 = Not Returned, Return_Status 1 = Returned
{unit_price_summary.to_dict()}
"""

    #Total spend summary
    if "Total_Spend" in data.columns and "Return_Status" in data.columns:
        spend_summary = (
            data.groupby("Return_Status")["Total_Spend"]
            .agg(["count", "mean", "median"])
            .round(2)
        )

        context += f"""

Total Spend Summary by Return Status:
Return_Status 0 = Not Returned, Return_Status 1 = Returned
{spend_summary.to_dict()}
"""

    #Quantity summary
    if "Quantity" in data.columns and "Return_Status" in data.columns:
        quantity_summary = (
            data.groupby("Return_Status")["Quantity"]
            .agg(["count", "mean", "median"])
            .round(2)
        )

        context += f"""

Quantity Summary by Return Status:
Return_Status 0 = Not Returned, Return_Status 1 = Returned
{quantity_summary.to_dict()}
"""

    #Product category and payment method combination
    if (
        "Product_Category" in data.columns 
        and "Payment_Method" in data.columns 
        and "Return_Status" in data.columns
    ):
        combo_summary = (
            data.groupby(["Product_Category", "Payment_Method"])
            .agg(
                Total_Orders=("Return_Status", "count"),
                Returned_Orders=("Return_Status", "sum"),
                Return_Rate_Percent=("Return_Status", lambda x: x.mean() * 100)
            )
            .sort_values("Return_Rate_Percent", ascending=False)
            .round(2)
            .head(10)
        )

        context += f"""

Product Category and Payment Method Combination Summary:
{combo_summary.to_dict()}
"""

    #Final notes for the AI
    context += """

Important Interpretation Notes:
- Return rates should be interpreted together with total order counts.
- A high return rate with very few orders may not be reliable.
- Recommendations should focus on patterns with both meaningful return rate and enough order volume.
- Use the summaries above to explain possible return patterns and suggest practical business actions.
"""

    return context

#Sends the user question and dataset summary to OpenAI
def ask_ai_copilot(user_question, data_context):
    
    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[
            {
                "role": "system",    #instructions for how the AI should behave
                "content": """
You are an AI data analytics copilot for an e-commerce business dashboard.

Rules:
- Use only the data context provided.
- Do not invent figures, columns, or trends.
- If the data is insufficient, clearly say what extra data is needed.
- Explain insights in simple business language.
- When possible, separate your answer into:
  1. Key finding
  2. Possible explanation
  3. Recommended business actions
- Keep the answer concise and practical.
"""
            },
            {
                "role": "user",  #actual question and dataset summary
                "content": f"""
Data context:
{data_context}

User question:
{user_question}
"""
            }
        ]
    )

    return response.choices[0].message.content








st.sidebar.header("Navigation")

page = st.sidebar.radio(
    "Choose a page",
    [
        "Business Insight Generator",
        "Customer Segmentation",
        "AI Copilot",
    ]
)

st.sidebar.header("Filters")

category_filter = st.sidebar.multiselect(
    "Product Category",
    options=sorted(df["Product_Category"].dropna().unique()),
    default=sorted(df["Product_Category"].dropna().unique())
)

filtered_df = df[df["Product_Category"].isin(category_filter)]

if page == "Business Insight Generator":

    st.header(" Business Insight Generator")

    total_orders = len(filtered_df)
    total_returns = filtered_df["Return_Status"].sum()
    return_rate = filtered_df["Return_Status"].mean() * 100

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Orders", f"{total_orders:,}")
    col2.metric("Total Returns", f"{int(total_returns):,}")
    col3.metric("Return Rate", f"{return_rate:.2f}%")

    st.subheader("Return Rate by Product Category")

    category_return = (
        filtered_df.groupby("Product_Category")["Return_Status"]
        .mean()
        .reset_index()
    )

    category_return["Return_Rate"] = category_return["Return_Status"] * 100

    fig = px.bar(
        category_return,
        x="Product_Category",
        y="Return_Rate",
        title="Return Rate by Product Category",
        labels={"Return_Rate": "Return Rate (%)"}
    )

    st.plotly_chart(fig, width='stretch')

    highest_category = category_return.sort_values(
        "Return_Rate", ascending=False
    ).iloc[0]

    st.subheader("Auto-Generated Business Insight")

    st.info(
        f"""
        The product category with the highest return rate is 
        **{highest_category['Product_Category']}** with a return rate of 
        **{highest_category['Return_Rate']:.2f}%**.

        This category should be reviewed for possible issues such as product quality,
        inaccurate product descriptions, sizing issues, delivery problems, or customer expectation mismatch.
        """
    )

elif page == "Customer Segmentation":

    st.header(" Customer Segmentation")

    st.write(
        "This section groups customers/orders based on behavioural patterns such as ratings, payment method, and return behaviour."
    )

    if "Customer_Rating" in filtered_df.columns:

        rating_segment = (
            filtered_df.groupby("Customer_Rating")
            .agg(
                Total_Orders=("Return_Status", "count"),
                Return_Rate=("Return_Status", "mean")
            )
            .reset_index()
        )

        rating_segment["Return_Rate"] = rating_segment["Return_Rate"] * 100

        fig = px.bar(
            rating_segment,
            x="Customer_Rating",
            y="Return_Rate",
            title="Return Rate by Customer Rating",
            labels={"Return_Rate": "Return Rate (%)"}
        )

        st.plotly_chart(fig, width='stretch')

        st.subheader("Customer Segment Insight")

        st.markdown("""
        Customers with lower ratings may indicate dissatisfaction with product quality,
        delivery experience, or expectation mismatch. These segments should be reviewed
        to reduce future returns and improve customer experience.
        """)

    if "Payment_Method" in filtered_df.columns:

        payment_segment = (
            filtered_df.groupby("Payment_Method")
            .agg(
                Total_Orders=("Return_Status", "count"),
                Return_Rate=("Return_Status", "mean")
            )
            .reset_index()
        )

        payment_segment["Return_Rate"] = payment_segment["Return_Rate"] * 100

        st.subheader("Return Rate by Payment Method")

        fig2 = px.bar(
            payment_segment,
            x="Payment_Method",
            y="Return_Rate",
            title="Return Rate by Payment Method",
            labels={"Return_Rate": "Return Rate (%)"}
        )

        st.plotly_chart(fig2, width='stretch')

        st.dataframe(payment_segment)


#AI copilot page
# AI copilot page
elif page == "AI Copilot":
    
    st.header("AI Copilot")

    st.write(
        "Ask business questions about your filtered e-commerce data. The AI will use summary statistics from your dataset to generate insights."
    )

    if filtered_df.empty:
        st.warning("No data available for the selected filters.")

    else:
        st.subheader("Suggested Questions")

        # Create a default empty question in Streamlit session state
        if "copilot_question" not in st.session_state:
            st.session_state.copilot_question = ""

        st.write("Click a preset question or type your own question below.")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Why are product returns high?"):
                st.session_state.copilot_question = "Why are product returns high in this dataset?"

            if st.button("Which category should we focus on?"):
                st.session_state.copilot_question = "Which product category should the business focus on and why?"

            if st.button("Are ratings related to returns?"):
                st.session_state.copilot_question = "Are customer ratings related to return behaviour?"

        with col2:
            if st.button("How can we reduce returns?"):
                st.session_state.copilot_question = "What actions can the business take to reduce product returns?"

            if st.button("Which payment method has highest risk?"):
                st.session_state.copilot_question = "Which payment method has the highest return risk?"

            if st.button("Summarise for management"):
                st.session_state.copilot_question = "Summarise the key dashboard insights for management."

        user_question = st.text_area(
            "Ask your AI copilot a question:",
            key="copilot_question",
            placeholder="Example: Why are returns high and what should the business do?"
        )

        data_context = create_data_context(filtered_df)

        if st.button("Ask AI Copilot"):

            if not user_question:
                st.warning("Please enter a question first.")

            elif os.getenv("OPENROUTER_API_KEY") is None:
                st.error("OpenRouter API key not found. Please check your .env file.")

            else:
                with st.spinner("AI Copilot is analysing your data..."):
                    try:
                        answer = ask_ai_copilot(user_question, data_context)

                        st.subheader("AI Copilot Response")
                        st.write(answer)

                    except Exception as e:
                        st.error("Something went wrong while calling the AI model.")
                        st.write(e)
