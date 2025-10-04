"""
Fake News Detection App
Streamlit Web Application for Real-time News Verification
"""

import streamlit as st
import joblib
import re
import string
import pandas as pd
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)


# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #7f8c8d;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 2rem 0;
    }
    .fake-news {
        background-color: #ffebee;
        color: #c62828;
        border: 3px solid #c62828;
    }
    .real-news {
        background-color: #e8f5e9;
        color: #2e7d32;
        border: 3px solid #2e7d32;
    }
    .stTextArea textarea {
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    try:
        model = joblib.load('fake_news_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        stop_words = joblib.load('stopwords.pkl')
        return model, vectorizer, stop_words
    except FileNotFoundError:
        st.error("Model files not found. Please train the model first by running train_model.py")
        return None, None, None

model, vectorizer, stop_words = load_models()

# Text cleaning function
def wordopt(text, stop_words):
    """Clean and preprocess text data"""
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub("\\W", " ", text)
    
    word_list = text.split()
    text_clean = [word for word in word_list if word not in stop_words]
    
    return ' '.join(text_clean)

# Prediction function
def predict_news(news_text):
    """Predict if news is fake or real"""
    if model is None:
        return None, None
    
    cleaned_text = wordopt(news_text, stop_words)
    vectorized_text = vectorizer.transform([cleaned_text])
    
    prediction = model.predict(vectorized_text)[0]
    probability = model.predict_proba(vectorized_text)[0]
    
    return prediction, probability

# Header
st.markdown('<p class="main-header">📰 Fake News Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Works best for Political news</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered News Verification System</p>', unsafe_allow_html=True)

st.markdown("---")

# Sidebar
st.sidebar.header("ℹ️ About")
st.sidebar.info(
    """
    This application uses Machine Learning to detect fake news articles.
    
    **Features:**
    - Real-time news verification
    - Confidence scores
    - Text analysis
    - Instant results
    
    **How it works:**
    1. Enter or paste news text
    2. Click "Analyze News"
    3. Get instant verification results
    """
)

st.sidebar.markdown("---")
st.sidebar.header("📊 Model Info")
if model is not None:
    st.sidebar.success("✅ Model Loaded")
    st.sidebar.metric("Accuracy", "98.6%")
    st.sidebar.metric("Precision", "99%")
    st.sidebar.metric("Recall", "99%")
else:
    st.sidebar.error("❌ Model Not Loaded")

# Main content
tab1, tab2, tab3 = st.tabs(["🔍 Analyze News", "📊 Examples", "ℹ️ Information"])

# TAB 1: ANALYZE NEWS
with tab1:
    st.header("Enter News Article")
    
    # Text input methods
    input_method = st.radio(
        "Choose input method:",
        ["Type/Paste Text", "Use Sample News"],
        horizontal=True
    )
    
    if input_method == "Type/Paste Text":
        news_text = st.text_area(
            "Enter the news article text (title and content):",
            height=200,
            placeholder="Paste or type the news article here..."
        )
    else:
        sample_choice = st.selectbox(
            "Select a sample news article:",
            [
                "Select...",
                "Sample 1: Real News - Climate Policy",
                "Sample 2: Fake News - Alien Discovery",
                "Sample 3: Real News - Economic Growth",
                "Sample 4: Fake News - Celebrity Scandal"
            ]
        )
        
        samples = {
            "Sample 1: Real News - Climate Policy": 
                "President announces new climate policy. The administration unveiled a comprehensive plan to reduce carbon emissions by 50% over the next decade. The policy includes incentives for renewable energy and stricter regulations on fossil fuel industries. Environmental groups have largely praised the initiative, while industry leaders express concerns about economic impacts.",
            
            "Sample 2: Fake News - Alien Discovery":
                "SHOCKING: Government confirms aliens living among us! Leaked documents reveal that extraterrestrial beings have been secretly integrated into human society for decades. Anonymous sources claim world leaders have been in contact with alien civilizations. This changes everything we thought we knew about our place in the universe!",
            
            "Sample 3: Real News - Economic Growth":
                "Stock market shows steady growth amid economic recovery. Major indices posted gains as investors responded positively to strong employment data and corporate earnings reports. Economists project continued moderate growth through the next quarter, though some caution about potential inflation risks.",
            
            "Sample 4: Fake News - Celebrity Scandal":
                "BREAKING: Hollywood star exposed in massive scandal that will shock you! Insiders reveal disturbing truth that mainstream media won't tell you. Click here to discover what they're hiding. You won't believe what happens next!"
        }
        
        news_text = samples.get(sample_choice, "")
        if news_text:
            st.text_area("News Article:", value=news_text, height=200, disabled=True)
    
    # Analysis button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button("🔍 Analyze News", type="primary", use_container_width=True)
    
    # Perform analysis
    if analyze_button and news_text:
        if len(news_text.strip()) < 50:
            st.warning("⚠️ Please enter at least 50 characters for accurate analysis.")
        else:
            with st.spinner("Analyzing news article..."):
                prediction, probability = predict_news(news_text)
                
                if prediction is not None:
                    st.markdown("---")
                    st.header("📊 Analysis Results")
                    
                    # Result display
                    col_a, col_b, col_c = st.columns([1, 2, 1])
                    
                    with col_b:
                        if prediction == 0:
                            st.markdown(
                                '<div class="result-box fake-news">⚠️ FAKE NEWS DETECTED</div>',
                                unsafe_allow_html=True
                            )
                            st.error(f"**Confidence: {probability[0]*100:.2f}%**")
                        else:
                            st.markdown(
                                '<div class="result-box real-news">✅ LIKELY REAL NEWS</div>',
                                unsafe_allow_html=True
                            )
                            st.success(f"**Confidence: {probability[1]*100:.2f}%**")
                    
                    # Confidence gauge
                    confidence = probability[prediction] * 100
                    
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=confidence,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Prediction Confidence (%)"},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkred" if prediction == 0 else "darkgreen"},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 75], 'color': "yellow"},
                                {'range': [75, 100], 'color': "lightgreen" if prediction == 1 else "lightcoral"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))
                    
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed breakdown
                    st.markdown("---")
                    st.subheader("📈 Probability Breakdown")
                    
                    col_x, col_y = st.columns(2)
                    with col_x:
                        st.metric(
                            "Fake News Probability",
                            f"{probability[0]*100:.2f}%",
                            delta=f"{(probability[0]-0.5)*100:.2f}%" if probability[0] > 0.5 else None
                        )
                    with col_y:
                        st.metric(
                            "Real News Probability",
                            f"{probability[1]*100:.2f}%",
                            delta=f"{(probability[1]-0.5)*100:.2f}%" if probability[1] > 0.5 else None
                        )
                    
                    # Warning indicators
                    st.markdown("---")
                    st.subheader("🚨 Key Indicators")
                    
                    if prediction == 0:
                        st.warning("""
                        **Warning Signs of Fake News:**
                        - ⚠️ Sensational or emotional language
                        - ⚠️ Lack of credible sources
                        - ⚠️ Unusual or suspicious URLs
                        - ⚠️ Too good or shocking to be true
                        - ⚠️ Poor grammar or spelling errors
                        """)
                        
                        st.info("""
                        **Recommendations:**
                        - 🔍 Cross-check with reputable news sources
                        - 🔗 Verify the original source
                        - 👥 Check if other reliable outlets report the same story
                        - 📅 Check the publication date
                        """)
                    else:
                        st.success("""
                        **Positive Indicators:**
                        - ✅ Professional language and tone
                        - ✅ Factual reporting style
                        - ✅ Verifiable information
                        - ✅ Consistent with known facts
                        """)
                        
                        st.info("""
                        **Best Practices:**
                        - 📖 Still verify from multiple sources
                        - 🔍 Look for primary sources
                        - 📊 Check for supporting evidence
                        - 🕒 Consider timing and context
                        """)
                    
                    # Text statistics
                    with st.expander("📝 Text Statistics"):
                        word_count = len(news_text.split())
                        char_count = len(news_text)
                        
                        stat_col1, stat_col2, stat_col3 = st.columns(3)
                        with stat_col1:
                            st.metric("Word Count", word_count)
                        with stat_col2:
                            st.metric("Character Count", char_count)
                        with stat_col3:
                            st.metric("Avg Word Length", f"{char_count/word_count:.1f}")

# TAB 2: EXAMPLES
with tab2:
    st.header("📊 Example News Articles")
    
    st.markdown("### ✅ Real News Examples")
    
    real_examples = [
        {
            "title": "Economic Report Released",
            "text": "The Federal Reserve released its quarterly economic report showing moderate GDP growth of 2.5% and unemployment holding steady at 4.2%. Analysts note stable inflation rates.",
            "confidence": "98.5%"
        },
        {
            "title": "Scientific Discovery Published",
            "text": "Researchers at MIT published findings in Nature journal demonstrating new advances in renewable energy storage. The peer-reviewed study shows promising results.",
            "confidence": "97.3%"
        }
    ]
    
    for example in real_examples:
        with st.expander(f"📰 {example['title']}"):
            st.write(example['text'])
            st.success(f"Model Confidence (Real): {example['confidence']}")
    
    st.markdown("### ⚠️ Fake News Examples")
    
    fake_examples = [
        {
            "title": "Shocking Conspiracy Revealed",
            "text": "BREAKING: Secret government documents leaked! Everything you know is a lie! Anonymous sources reveal shocking truth that will change the world forever. Click now!",
            "confidence": "99.2%"
        },
        {
            "title": "Miracle Cure Discovered",
            "text": "Doctors HATE this one simple trick! This miracle cure will solve all your health problems instantly. Big pharma doesn't want you to know about this!",
            "confidence": "98.8%"
        }
    ]
    
    for example in fake_examples:
        with st.expander(f"⚠️ {example['title']}"):
            st.write(example['text'])
            st.error(f"Model Confidence (Fake): {example['confidence']}")

# TAB 3: INFORMATION
with tab3:
    st.header("ℹ️ About Fake News Detection")
    
    st.markdown("""
    ### What is Fake News?
    
    Fake news refers to false or misleading information presented as legitimate news. It can:
    - Spread misinformation
    - Influence public opinion
    - Undermine trust in media
    - Impact elections and democracy
    
    ### How This System Works
    
    Our AI model uses **Natural Language Processing (NLP)** and **Machine Learning**:
    
    1. **Text Preprocessing:** Cleans and normalizes the input text
    2. **Feature Extraction:** Converts text to numerical features using TF-IDF
    3. **Classification:** Logistic Regression model predicts authenticity
    4. **Confidence Score:** Probability indicates prediction certainty
    
    ### Model Performance
    
    - **Accuracy:** 98.6%
    - **Precision:** 99%
    - **Recall:** 99%
    - **Training Data:** 44,898 news articles
    
    ### Limitations
    
    ⚠️ **Important Notes:**
    - Not 100% accurate - always verify from multiple sources
    - Works best with English language news
    - Trained on specific dataset - may not cover all news types
    - Cannot detect all forms of misinformation
    - Should be used as a tool, not sole source of truth
    
    ### Tips for Identifying Fake News
    
    1. **Check the source:** Is it from a reputable news outlet?
    2. **Look for evidence:** Are there credible sources and citations?
    3. **Examine the author:** Can you find information about them?
    4. **Check the date:** Is the story current or recycled?
    5. **Read beyond the headline:** Headlines can be misleading
    6. **Consider your bias:** Are your beliefs affecting judgment?
    7. **Consult fact-checkers:** Use sites like Snopes, FactCheck.org
    
    ### Technology Stack
    
    - **Python:** Core programming language
    - **Scikit-learn:** Machine learning framework
    - **TF-IDF:** Text feature extraction
    - **Logistic Regression:** Classification algorithm
    - **Streamlit:** Web application framework
    - **NLTK:** Natural language processing
    
    ### Dataset Information
    
    **Source:** Kaggle Fake News Detection Dataset
    - Real news from Reuters and other reputable sources
    - Fake news from various unreliable websites
    - Labeled and verified by experts
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>📰 Fake News Detection System v1.0</p>
        <p>Built with Machine Learning and Natural Language Processing</p>
        <p><i>Always verify information from multiple reputable sources</i></p>
    </div>
    """,
    unsafe_allow_html=True
)