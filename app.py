import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import datetime as datetime
import json
from openai import AzureOpenAI
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import  train_test_split
from statsmodels.tsa.arima.model import ARIMA
from transformers import pipeline
import numpy as np


#API_KEY="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"
API_KEY="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"
SLACK_WEBHOOK="https://hooks.slack.com/services/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"

def truncatetxt(txt,max_length=517):
  return txt[:max_length]

def load_competitor_data():
  data=pd.read_csv("competitive.csv")
  #data=pd.read_csv("amazon_products_headphones (3).csv")
  print(data.head())
  return data


def load_reviews_data():
  reviews=pd.read_csv("review.csv")
  return reviews

def analyze_sentiment(reviews):
  sentiment_pipeline=pipeline("sentiment-analysis")
  results = [sentiment_pipeline([review])[0] for review in reviews]
  return results

def trainmodel(data):
  data["discount"]=data["discount"].astype(float)
  data["price"]=data["price"].astype(float)

  data["predicted_discount"]=data["discount"]+(data["price"]*0.05).round(2)

  x=data[["price","discount"]]
  y=data["predicted_discount"]
  print(x)

  xtrain,xtest,ytrain,ytest=train_test_split(
      x,y,test_size=0.2,random_state=42,train_size=0.8
      )
  
  model=RandomForestRegressor(random_state=42)
  model.fit(xtrain,ytrain)
  return model


def forecast_discount(data,future_days=5):
  try:
    data=data.sort_index()
    if data.index.duplicated().any():
      data = data.loc[~data.index.duplicated(keep='first')]

    #print(product_data.index)
    data.index = pd.to_datetime(data.index, errors="coerce")
    data["discount"]=pd.to_numeric(data["discount"],errors="coerce")
    data = data.asfreq('D')  # Ensure a daily frequency
    data=data.dropna(subset=["discount"])
    if data.index.duplicated().any():
      data = data.loc[~data.index.duplicated(keep='first')]
    data=data.dropna(subset=["discount"])
    discount_series=data["discount"]

    if discount_series.empty:
      st.warning("Discount data is empty. Cannot perform forecasting.")
      return None

    if not isinstance(data.index,pd.DatetimeIndex):
      try:
        data.index=pd.to_datetime(data.index)
      except Exception as e:
        raise ValueError(
          "Invalid date format"
        ) from e
      #data.index=pd.to_datetime(data.index,format="%Y-%m-%d")

    model=ARIMA(discount_series,order=(5,1,0))
    model_fit=model.fit()
    forecast=model_fit.forecast(steps=future_days)

    future_dates=pd.date_range(
        start=discount_series.index[-1]+pd.Timedelta(days=1),
        periods=future_days,
        freq="D"
        
    )

    forecast_df=pd.DataFrame({"Date":future_dates,"Predicted_Discount":forecast})
    forecast_df.set_index("Date",inplace=True)
    return forecast_df
  except Exception as e:
        st.error(f"An error occurred in forecasting: {e}")
        return None


def send_to_slack(data):
  payload={"text":data}
  response=requests.post(
      SLACK_WEBHOOK,
      data=json.dumps(payload),
      headers={"Content-Type":"application/json"}
  )
  return response

def generate_strategy_recommendation(product_name,competitor_data,sentiment):
  date=datetime.datetime.now()
  prompt=f"""
  You are a highly skilled business strategist specializing in e-commerce.
  Based on the following details,suggest the following : 
  
  1. **Product Name**:{product_name}

  2. **Competitor Data** (including current prices,discounts, and predicted discounts):
  {competitor_data}

  3. **Sentiment Analysis** :
  {sentiment}

  4.**Today's Date**:{str(date)}.

  ### Task:
  - Analyze the competitor data and identify key pricing trends.
  - Leverage sentiment analysis insights to highlight areas where customer satisfaction can be improved.
  - Use the discount predictions to suggest how pricing strategies can be optimized over the next 5 days.
  - Recommend promotional campaigns or marketing strategies that align with customer sentiments and competitive trends.
  - Ensure the strategies are actionable, realistic, and geared toward increasing customer satisfaction, driving sales, and improving competitive standing.

  Provide your recommendations in a structured format :
  1. **Pricing Strategy**
  2. **Promotional Campaign Ideas**
  3. **Customer Sentiment Analysis**
  """

  headers={"Content-Type":"application/json","Authorization":f"Bearer {API_KEY}"}
  data={
        "messages":[{"role":"user","content":prompt}],
        "model":"llama3-8b-8192",
        "temperature":0
  }

  res=requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        data=json.dumps(data),
        headers=headers,
  )
  print("API Response : ",res.json())
  response_data=res.json()
  if "choices" not in response_data:
    st.error("Invalid response from the API. Check your configuration or API quota.")
    return "Unable to generate recommendations due to an API error."

  response = response_data["choices"][0]["message"]["content"]
  return response


st.set_page_config(page_title="E-Commerce Competitor Strategy Dashboard",layout="wide")
st.title("E-Commerce Competitor Strategy Dashboard")
st.sidebar.header("Select a Product : ")
products=["Apple 15","Apple 15 Pro","Apple 15 Pro Max","Apple 15 Mini"]
#products=["boAt Rockerz 450","boAt Rockerz 430"]
selected_product=st.sidebar.selectbox("Choose a product to analyze: ",products)

competitor_data=load_competitor_data()
reviews=load_reviews_data()
product_data=competitor_data[competitor_data["name"]==selected_product]
review_data=reviews[reviews["name"]==selected_product]

st.header(f"Competitor Analysis for {selected_product}")
st.subheader("Competitor Data")
st.table(product_data.tail(5))

if not review_data.empty:
  review_data["review"]=review_data["review"].apply(
        lambda x:truncatetxt(x,512)          
  ).tolist()
  reviews=review_data["review"].tolist()
  sentiments=analyze_sentiment(reviews)

  st.subheader("Customer Sentiment Analysis")
  sentiment_df=pd.DataFrame(sentiments)
  fig=px.bar(sentiment_df,x="label",title="Sentiment Analysis Results")
  st.plotly_chart(fig)
else:
  sentiments="no reviews"
  st.write("No review available for this product")


product_data["Date"]=pd.to_datetime(product_data["date"])
product_data=product_data.dropna(subset=["Date"])
product_data.set_index("Date",inplace=True)
#product_data=product_data.sort_index(inplace=True)
product_data.sort_index(inplace=True)



product_data["discount"]=pd.to_numeric(product_data["discount"],errors="coerce")
product_data=product_data.dropna(subset=["discount"])

product_data_with_predictions=forecast_discount(product_data)

if product_data_with_predictions is not None:
    st.subheader("Competitor current and predicted discounts")
    st.table(product_data_with_predictions.tail(10))
else:
    st.warning("No predictions available due to data issues or forecasting errors.")

recommendations=generate_strategy_recommendation(
      selected_product,
      product_data_with_predictions,
      sentiments if not review_data.empty else "No reviews"
)
    
st.subheader("Strategic recommendations")
st.write(recommendations)
send_to_slack(recommendations)
