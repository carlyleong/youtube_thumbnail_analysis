# 📸 YouTube Thumbnail Analysis Pipeline

## 🧾 Overview
This project analyzes the relationship between YouTube thumbnail visuals and video performance (specifically view count). Using image and metadata processing combined with machine learning, the pipeline helps content creators and marketing teams optimize thumbnail design to boost engagement.

---

## 🎯 Objective
To **predict** and **understand** which thumbnail features lead to higher YouTube views.

### 📌 Beneficiaries:
- **Marketing Teams**: Identify high-performing visual content.
- **Content Creators**: Choose engaging thumbnails.
- **Product Teams**: Improve video recommendation systems.

---

## 💼 Business Use Case
For media companies managing large YouTube libraries, this pipeline helps:

- Increase click-through rates and viewer retention.
- Design thumbnails that consistently drive engagement.
- A/B test visual themes to improve video performance.

---

## 🧰 Tools & Technologies

- 🔄 **PySpark**: Distributed data analysis & model training  
- 🗃️ **MongoDB**: Storage of video and thumbnail data  
- ⏱️ **Airflow**: Data pipeline scheduling  
- 🐍 **Python + Pandas**: Data wrangling  
- 🧪 **MLflow**: Experiment tracking and model versioning  
- 📊 **Matplotlib & FPDF**: Reporting and visualization  

---

## 🔁 Data Flow and Features

### 🔧 Engineered Features

- `sentiment_score`: Derived from thumbnail image labels.
- `likeCount`, `commentCount`: Raw metadata from videos.
- `like_view_ratio`, `comment_view_ratio`: Engagement indicators.

### 🎯 Target Variable

- `viewCount`: The metric we aim to predict.

---

## ⚙️ Modeling Process

1. **Data Ingestion**: From MongoDB using YouTube API.
2. **Cleaning & Parsing**: Convert nested fields and JSON data.
3. **Feature Engineering**: Sentiment scores and engagement ratios.
4. **Spark ML Pipeline**:
   - `VectorAssembler`
   - `StandardScaler`
5. **Model Training**:
   - Linear Regression
   - Random Forest
   - Gradient Boosted Trees
6. **Evaluation Metrics**: RMSE, MAE, R²
7. **Experiment Tracking**: Logged with MLflow

---

## 📊 Results & Model Comparison

| Model               | MAE       | RMSE      | R²     |
|--------------------|-----------|-----------|--------|
| Gradient Boosting  | 1,677,715 | 3,318,319 | -0.075 |
| Random Forest       | **1,032,889** | **1,942,578** | **0.631** |
| Linear Regression   | 1,777,730 | 2,833,735 | 0.216  |

> ✅ **Best Performer**: Random Forest — offers the best balance of accuracy and generalization.

---

## 💡 Key Insights

- **Engagement Ratios Are Crucial**: Strong predictors of view count.
- **Positive Sentiment Wins**: Labels like `"smile"` or vibrant colors tend to lead to more views.
- **Avoid Low-Affect Visuals**: Static or technical visuals (e.g., `"screenshot"`, `"electronic device"`) reduce engagement.

---

## 🧠 Strategic Takeaways

- Run **A/B tests** comparing high/low sentiment thumbnails.
- Prefer **emotionally charged visuals** over static screenshots.
- Use **like/view** and **comment/view** ratios to guide thumbnail strategy.
- Predict performance **before publishing** using model insights.

---

## ✅ Benefits

- 🚀 **Scalable**: Powered by PySpark for large datasets.
- 📈 **Explainable**: Outputs interpretable metrics and visualizations.
- 🔁 **Repeatable**: Fully trackable with MLflow.
- 🎯 **Actionable**: Can support thumbnail A/B testing and performance prediction.

---

## 🔮 Next Steps

- 📊 Integrate with a **real-time dashboard** for marketing insights.
- 🧠 Extend sentiment analysis using a **deep learning-based vision model**.
- 🎯 Fine-tune models using **thumbnail click-through-rate (CTR)** if available.

---

> Built to empower creators with **data-driven thumbnail decisions**.
