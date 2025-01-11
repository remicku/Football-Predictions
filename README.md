# Football Match Predictions

This project predicts football match outcomes using data scraped from [FBref](https://fbref.com/). It employs machine learning techniques, specifically a Random Forest classifier, to analyze match statistics and forecast results.

## Features

-   **Web Scraping**: Automatically collects match and team statistics using BeautifulSoup and pandas.
-   **Rolling Averages**: Computes rolling averages for key performance metrics to enhance prediction accuracy.
-   **Machine Learning Model**: Uses a Random Forest classifier to predict match outcomes based on historical data.
-   **Data Analysis**: Includes tools for analyzing prediction accuracy and exploring model performance.

## Installation

1.  Clone this repository:
    
	```bash
	git clone https://github.com/yourusername/football-predictions.git
	```
    
2.  Install required dependencies:

	```bash
	pip install -r requirements.txt
	```
    
3.  Ensure you have Python 3.8+ installed.

## Usage

1.  **Scrape Data**: Run the `scraping.py` script to gather data from FBref:
    
	```bash
	python scraping.py
	```
	This will create a `matches.csv` file containing match statistics.
    
2.  **Train and Test Model**: Use `predictions.py` to train the Random Forest model and evaluate its performance:
    
    ```bash
    python predictions.py
    ```
    
    Key metrics, such as accuracy and precision, will be displayed in the console.
    
3.  **Test Specific Predictions**: Modify and execute `test.py` to explore the prediction outputs and model insights:
    
    ```bash
    python test.py
    ```
    

## Data

-   **Input Data**: The scraped data contains match details, including:
    -   Venue
    -   Opponent
    -   Goals scored and conceded
    -   Shots, shots on target, free kicks, and penalties
    -   Match results (Win/Draw/Loss)
-   **Processed Features**: Extracted features include:
    -   Encoded venue and opponent
    -   Match hour and day of the week
    -   Rolling averages of match performance metrics

## Model

-   **Algorithm**: Random Forest Classifier
-   **Hyperparameters**:
    -   Number of estimators: 50
    -   Minimum samples split: 10
    -   Random state: 1
-   **Evaluation**:
    -   Accuracy and precision scores
    -   Detailed cross-tab analysis of predictions vs. actual outcomes

## File Overview

-   `scraping.py`: Collects and preprocesses football match data.
-   `predictions.py`: Builds and evaluates the prediction model.
-   `test.py`: Provides tools for analyzing and visualizing predictions.
-   `matches.csv`: The dataset used for training and testing the model.

## Future Work

-   Improve feature engineering by incorporating more advanced statistics.
-   Experiment with other machine learning algorithms.
-   Extend predictions to other leagues and competitions.