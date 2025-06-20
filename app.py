import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
from io import StringIO
import base64

# Set page config
st.set_page_config(
    page_title="🎬 Movie Genre Classifier",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4ECDC4;
        margin: 0.5rem 0;
    }
    .genre-tag {
        background: #FF6B6B;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 0.2rem;
        display: inline-block;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4ECDC4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


class MovieGenreClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='linear', random_state=42, probability=True),
            'Naive Bayes': MultinomialNB()
        }
        self.label_encoder = LabelEncoder()
        self.best_model = None
        self.best_model_name = None

    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text

    def parse_data(self, data_text):
        """Parse the custom format data"""
        lines = data_text.strip().split('\n')
        movies = []

        for line in lines:
            if line.strip():
                # Split by ::: to get components
                parts = line.split(' ::: ')
                if len(parts) >= 4:
                    # Extract components
                    movie_id = parts[0].strip()
                    title = parts[1].strip()
                    genre = parts[2].strip()
                    description = ' ::: '.join(parts[3:]).strip()

                    movies.append({
                        'id': movie_id,
                        'title': title,
                        'genre': genre,
                        'description': description
                    })

        return pd.DataFrame(movies)

    def train_models(self, df):
        """Train multiple models and return results"""
        # Preprocess descriptions
        df['clean_description'] = df['description'].apply(self.preprocess_text)

        # Prepare features and labels
        X = df['clean_description']
        y = df['genre']

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)

        # Vectorize text
        X_vectorized = self.vectorizer.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_vectorized, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        results = {}
        best_accuracy = 0

        # Train each model
        for name, model in self.models.items():
            # Train model
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)

            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'predictions': y_pred,
                'y_test': y_test
            }

            # Track best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.best_model = model
                self.best_model_name = name

        return results, X_test, y_test

    def predict_genre(self, description):
        """Predict genre for a new description"""
        if self.best_model is None:
            return None, None

        clean_desc = self.preprocess_text(description)
        vectorized = self.vectorizer.transform([clean_desc])

        prediction = self.best_model.predict(vectorized)[0]
        probabilities = self.best_model.predict_proba(vectorized)[0]

        genre = self.label_encoder.inverse_transform([prediction])[0]

        # Get top 5 predictions with probabilities
        top_indices = np.argsort(probabilities)[-5:][::-1]
        top_predictions = []

        for idx in top_indices:
            pred_genre = self.label_encoder.inverse_transform([idx])[0]
            prob = probabilities[idx]
            top_predictions.append((pred_genre, prob))

        return genre, top_predictions


def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎬 Movie Genre Classification System</h1>
        <p>Advanced ML-powered genre prediction for movie descriptions</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize classifier
    if 'classifier' not in st.session_state:
        st.session_state.classifier = MovieGenreClassifier()

    # Sidebar
    st.sidebar.markdown("### 🛠️ Configuration")

    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Upload & Training",
        "🎯 Model Performance",
        "🔮 Prediction",
        "📈 Analytics",
        "💾 Export Model"
    ])

    with tab1:
        st.markdown("### 📁 Upload Your Movie Dataset")

        # Sample data preview
        with st.expander("📋 View Sample Data Format"):
            st.code("""
1 ::: Movie Title ::: genre ::: Movie description here...
2 ::: Another Movie ::: comedy ::: Another movie description...
            """)

        uploaded_file = st.file_uploader(
            "Choose your movie dataset file",
            type=['txt', 'csv'],
            help="Upload a text file with the format: ID ::: Title ::: Genre ::: Description"
        )

        if uploaded_file is not None:
            try:
                # Read file content
                if uploaded_file.type == "text/plain":
                    content = str(uploaded_file.read(), "utf-8")
                else:
                    content = str(uploaded_file.read(), "utf-8")

                # Parse data
                df = st.session_state.classifier.parse_data(content)

                if not df.empty:
                    st.success(f"✅ Successfully loaded {len(df)} movies!")

                    # Display basic statistics
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>{len(df)}</h3>
                            <p>Total Movies</p>
                        </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        unique_genres = df['genre'].nunique()
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>{unique_genres}</h3>
                            <p>Unique Genres</p>
                        </div>
                        """, unsafe_allow_html=True)

                    with col3:
                        avg_desc_length = df['description'].str.len().mean()
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>{avg_desc_length:.0f}</h3>
                            <p>Avg Description Length</p>
                        </div>
                        """, unsafe_allow_html=True)

                    with col4:
                        most_common_genre = df['genre'].value_counts().index[0]
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>{most_common_genre}</h3>
                            <p>Most Common Genre</p>
                        </div>
                        """, unsafe_allow_html=True)

                    # Genre distribution
                    st.markdown("### 📊 Genre Distribution")
                    genre_counts = df['genre'].value_counts()

                    fig = px.bar(
                        x=genre_counts.index,
                        y=genre_counts.values,
                        title="Distribution of Movie Genres",
                        color=genre_counts.values,
                        color_continuous_scale="viridis"
                    )
                    fig.update_layout(
                        xaxis_title="Genre",
                        yaxis_title="Count",
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Sample data preview
                    st.markdown("### 👀 Sample Data Preview")
                    st.dataframe(df.head(10), use_container_width=True)

                    # Train models button
                    if st.button("🚀 Train Models", type="primary"):
                        with st.spinner("Training multiple ML models... This may take a few minutes."):
                            try:
                                results, X_test, y_test = st.session_state.classifier.train_models(df)
                                st.session_state.results = results
                                st.session_state.X_test = X_test
                                st.session_state.y_test = y_test
                                st.session_state.df = df

                                st.success("🎉 Models trained successfully!")

                                # Display quick results
                                st.markdown("### 🏆 Quick Results")
                                for name, result in results.items():
                                    accuracy = result['accuracy']
                                    if name == st.session_state.classifier.best_model_name:
                                        st.success(f"🏅 **{name}**: {accuracy:.4f} (Best Model)")
                                    else:
                                        st.info(f"📊 **{name}**: {accuracy:.4f}")

                            except Exception as e:
                                st.error(f"❌ Error training models: {str(e)}")

                else:
                    st.error("❌ No valid data found in the uploaded file.")

            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")

    with tab2:
        st.markdown("### 🎯 Model Performance Analysis")

        if 'results' in st.session_state:
            results = st.session_state.results

            # Performance comparison
            st.markdown("#### 📈 Model Accuracy Comparison")

            model_names = list(results.keys())
            accuracies = [results[name]['accuracy'] for name in model_names]

            # Create comparison chart
            fig = go.Figure(data=[
                go.Bar(
                    x=model_names,
                    y=accuracies,
                    text=[f'{acc:.4f}' for acc in accuracies],
                    textposition='auto',
                    marker_color=['#FF6B6B' if name == st.session_state.classifier.best_model_name else '#4ECDC4'
                                  for name in model_names]
                )
            ])

            fig.update_layout(
                title="Model Performance Comparison",
                xaxis_title="Model",
                yaxis_title="Accuracy",
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)

            # Detailed performance for best model
            st.markdown(f"#### 🏆 Detailed Analysis - {st.session_state.classifier.best_model_name}")

            best_result = results[st.session_state.classifier.best_model_name]
            y_pred = best_result['predictions']
            y_test = best_result['y_test']

            # Classification report
            target_names = st.session_state.classifier.label_encoder.classes_
            report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)

            # Convert to DataFrame for better display
            report_df = pd.DataFrame(report).transpose()
            report_df = report_df.round(4)

            st.markdown("##### 📊 Classification Report")
            st.dataframe(report_df, use_container_width=True)

            # Confusion Matrix
            st.markdown("##### 🔄 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)

            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=target_names,
                yticklabels=target_names,
                ax=ax
            )
            ax.set_title('Confusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')

            st.pyplot(fig)

        else:
            st.info("👆 Please upload data and train models first in the 'Data Upload & Training' tab.")

    with tab3:
        st.markdown("### 🔮 Genre Prediction")

        if hasattr(st.session_state.classifier, 'best_model') and st.session_state.classifier.best_model is not None:
            st.markdown(f"**Active Model:** {st.session_state.classifier.best_model_name}")

            # Text input for prediction
            description = st.text_area(
                "Enter a movie description:",
                height=150,
                placeholder="Enter a detailed movie description here..."
            )

            if st.button("🎯 Predict Genre", type="primary"):
                if description.strip():
                    with st.spinner("Analyzing description..."):
                        try:
                            predicted_genre, top_predictions = st.session_state.classifier.predict_genre(description)

                            # Display main prediction
                            st.markdown("#### 🎬 Prediction Results")
                            st.success(f"**Predicted Genre:** {predicted_genre}")

                            # Display top 5 predictions with probabilities
                            st.markdown("#### 📊 Top 5 Predictions")

                            genres = [pred[0] for pred in top_predictions]
                            probabilities = [pred[1] for pred in top_predictions]

                            # Create horizontal bar chart
                            fig = go.Figure(go.Bar(
                                x=probabilities,
                                y=genres,
                                orientation='h',
                                marker_color=['#FF6B6B' if i == 0 else '#4ECDC4' for i in range(len(genres))]
                            ))

                            fig.update_layout(
                                title="Prediction Confidence",
                                xaxis_title="Probability",
                                yaxis_title="Genre",
                                height=400
                            )

                            st.plotly_chart(fig, use_container_width=True)

                            # Display as table
                            pred_df = pd.DataFrame(top_predictions, columns=['Genre', 'Probability'])
                            pred_df['Probability'] = pred_df['Probability'].round(4)
                            st.dataframe(pred_df, use_container_width=True)

                        except Exception as e:
                            st.error(f"❌ Error making prediction: {str(e)}")
                else:
                    st.warning("⚠️ Please enter a movie description.")

            # Quick test examples
            st.markdown("#### 🎲 Try These Examples")
            examples = [
                "A young wizard discovers he has magical powers and must attend a school for wizards to learn magic and fight against dark forces.",
                "Two detectives investigate a series of murders in a dark, gritty city while uncovering a conspiracy that goes to the top.",
                "A romantic story about two people who meet by chance and fall in love despite obstacles keeping them apart.",
                "An action-packed adventure where a group of heroes must save the world from an alien invasion using advanced technology.",
                "A documentary exploring the life and career of a famous musician through interviews and rare footage."
            ]

            selected_example = st.selectbox("Choose an example:", [""] + examples)

            if selected_example and st.button("🔄 Use This Example"):
                st.experimental_rerun()

        else:
            st.info("👆 Please train models first in the 'Data Upload & Training' tab.")

    with tab4:
        st.markdown("### 📈 Dataset Analytics")

        if 'df' in st.session_state:
            df = st.session_state.df

            # Genre analysis
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 🎭 Genre Statistics")
                genre_stats = df['genre'].value_counts()

                fig = px.pie(
                    values=genre_stats.values,
                    names=genre_stats.index,
                    title="Genre Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("#### 📝 Description Length Analysis")
                df['desc_length'] = df['description'].str.len()

                fig = px.histogram(
                    df,
                    x='desc_length',
                    nbins=50,
                    title="Distribution of Description Lengths"
                )
                st.plotly_chart(fig, use_container_width=True)

            # Word cloud data preparation
            st.markdown("#### 📊 Most Common Words by Genre")

            selected_genre = st.selectbox(
                "Select a genre to analyze:",
                df['genre'].unique()
            )

            if selected_genre:
                genre_descriptions = df[df['genre'] == selected_genre]['description'].str.cat(sep=' ')

                # Simple word frequency (since we can't use wordcloud)
                words = genre_descriptions.lower().split()
                word_freq = pd.Series(words).value_counts().head(20)

                fig = px.bar(
                    x=word_freq.values,
                    y=word_freq.index,
                    orientation='h',
                    title=f"Top 20 Words in {selected_genre} Movies"
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("👆 Please upload and process data first.")

    with tab5:
        st.markdown("### 💾 Export Trained Model")

        if hasattr(st.session_state.classifier, 'best_model') and st.session_state.classifier.best_model is not None:
            st.success(f"✅ Model ready for export: {st.session_state.classifier.best_model_name}")

            if st.button("📥 Download Model", type="primary"):
                try:
                    # Create model package
                    model_package = {
                        'model': st.session_state.classifier.best_model,
                        'vectorizer': st.session_state.classifier.vectorizer,
                        'label_encoder': st.session_state.classifier.label_encoder,
                        'model_name': st.session_state.classifier.best_model_name,
                        'performance': st.session_state.results[st.session_state.classifier.best_model_name]['accuracy']
                    }

                    # Serialize model
                    model_bytes = pickle.dumps(model_package)

                    # Create download button
                    st.download_button(
                        label="💾 Download Trained Model",
                        data=model_bytes,
                        file_name=f"movie_genre_classifier_{st.session_state.classifier.best_model_name.lower().replace(' ', '_')}.pkl",
                        mime="application/octet-stream"
                    )

                    st.info("📋 To use the downloaded model, load it with pickle and use the predict_genre method.")

                except Exception as e:
                    st.error(f"❌ Error creating model package: {str(e)}")
        else:
            st.info("👆 Please train a model first before exporting.")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        🎬 Movie Genre Classification System | Built with Streamlit & Scikit-learn
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()