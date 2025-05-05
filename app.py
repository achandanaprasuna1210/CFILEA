from flask import Flask, request, render_template
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend, which doesn't require Tkinter
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

# Initialize Flask app
app = Flask(__name__)

# Load the trained model (update the paths with the actual paths to your model)
model = joblib.load("models/random_forest_model.pkl")  # Update with your model path
scaler = joblib.load("models/scaler.pkl")  # Update if scaling was used

# Define the features for the input form
FEATURES = ['Schooling', 'GDP', 'BMI', 'thinness 1-19 years', 'HIV/AIDS', 'Diphtheria', 'Total expenditure', 'Hepatitis B']

# Load and preprocess the dataset
dataset = pd.read_csv('data/life_expectancy_cleaned.csv')  # Ensure correct file path

# Preprocess the data for insights
dataset['Life expectancy'] = pd.to_numeric(dataset['Life expectancy'], errors='coerce')
dataset['GDP'] = pd.to_numeric(dataset['GDP'], errors='coerce')

# Calculate average life expectancy and GDP for each country
country_stats = dataset.groupby('Country').agg({
    'Life expectancy': 'mean',
    'GDP': 'mean',
    'HIV/AIDS': 'mean',
    'Total expenditure': 'mean'
}).reset_index()

# Add Key Challenges based on certain thresholds
def assign_key_challenges(row):
    challenges = []
    if row['GDP'] < 10000:
        challenges.append("Low GDP")
    if row['HIV/AIDS'] > 0.5:
        challenges.append("High HIV/AIDS rates")
    if row['Total expenditure'] < 5:
        challenges.append("Low health expenditure")
    return ", ".join(challenges) if challenges else "Not Available"

country_stats['Key Challenges'] = country_stats.apply(assign_key_challenges, axis=1)

# Function to create a chart for multiple countries
def create_comparison_chart(selected_data):
    plt.style.use('seaborn-v0_8-dark-palette')  # Set the style

    fig, ax = plt.subplots(1, 2, figsize=(14, 7))

    # Bar chart for Life Expectancy comparison
    sns.barplot(x='Country', y='Life expectancy', data=selected_data, ax=ax[0], palette="coolwarm")
    ax[0].set_title('Life Expectancy by Country', fontsize=14)
    ax[0].tick_params(axis='x', rotation=45)
    ax[0].set_xlabel('Country', fontsize=12)
    ax[0].set_ylabel('Life Expectancy', fontsize=12)

    # Scatter plot for GDP vs Life Expectancy
    sns.scatterplot(x='GDP', y='Life expectancy', data=selected_data, hue='Country', ax=ax[1], palette="tab10")
    ax[1].set_title('GDP vs Life Expectancy', fontsize=14)
    ax[1].set_xlabel('GDP (USD)', fontsize=12)
    ax[1].set_ylabel('Life Expectancy', fontsize=12)

    img_io = io.BytesIO()
    plt.tight_layout()
    plt.savefig(img_io, format='png')
    img_io.seek(0)

    img_base64 = base64.b64encode(img_io.getvalue()).decode('utf8')
    plt.close(fig)
    return img_base64

# Route for the form input page
@app.route('/')
def home():
    return render_template('form.html')

# Route for making predictions based on user input
@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = {feature: float(request.form[feature]) for feature in FEATURES}
        df = pd.DataFrame([input_data])
        scaled_data = scaler.transform(df) if scaler else df
        prediction = model.predict(scaled_data)[0]

        if hasattr(model, 'feature_importances_'):
            feature_importances = model.feature_importances_
            importance_df = pd.DataFrame({
                'Feature': FEATURES,
                'Importance': feature_importances
            }).sort_values(by="Importance", ascending=False).head(3)
        else:
            importance_df = pd.DataFrame()

        return render_template(
            'results.html',
            prediction=round(prediction, 2),
            top_factors=importance_df.to_dict(orient='records')
        )
    except Exception as e:
        return render_template('error.html', error=str(e))

# Route for insights page with multiple country support
@app.route('/insights', methods=['GET', 'POST'])
def insights():
    countries = request.form.get('countries') if request.method == 'POST' else None
    if countries:
        country_list = [country.strip() for country in countries.split(',')]
        selected_data = country_stats[country_stats['Country'].isin(country_list)]

        if selected_data.empty:
            country_stats_html = "<p class='text-danger'>No matching countries found. Please try again with valid country names.</p>"
            img_base64 = None
        else:
            country_stats_html = selected_data.to_html(index=False, classes="table table-striped")
            img_base64 = create_comparison_chart(selected_data)
    else:
        country_stats_html = country_stats.to_html(index=False, classes="table table-striped")
        img_base64 = None

    return render_template('insights.html', country_stats=country_stats_html, chart_img=img_base64)

if __name__ == '__main__':
    app.run(debug=True)
