from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load your scikit-learn models
try:
    meal_model = pickle.load(open('meal_plan.pkl', 'rb'))
    calories_model = pickle.load(open('calories.pkl', 'rb'))
    workout_model = pickle.load(open('workout_plan.pkl', 'rb'))
except Exception as e:
    raise ValueError(f"Error loading models: {str(e)}")


@app.route("/")
def index():
    return render_template("index.html", user_input=None, messages=None, error=None)


@app.route("/predict_all", methods=["POST"])
def predict_all():
    try:
        # Parse form data from the request
        input_data = {
            "Age": int(request.form['Age']),
            "Gender": request.form['Gender'],
            "Height": float(request.form['Height']),
            "Weight": float(request.form['Weight']),
            "Fitness_Level": request.form['Fitness_Level'],
            "Fitness_Goal": request.form['Fitness_Goal'],
            "Medical_History": request.form['Medical_History'],
        }

        # Create a pandas DataFrame with named columns
        input_df = pd.DataFrame(input_data, index=[0])

        # Make predictions using the loaded models
        meal_plan = meal_model.predict(input_df)
        calories = calories_model.predict(input_df)
        workout_plan = workout_model.predict(input_df)

        # Prepare messages for the user
        messages = {
            "meal_plan": f"Recommended Meal Plan: {meal_plan[0]}",
            "calories": f"Estimated Daily Calories: {calories[0]}",
            "workout_plan": f"Recommended Workout Plan: {workout_plan[0]}"
        }

        return render_template("index.html", user_input=input_data, messages=messages, error=None)
    except Exception as e:
        return render_template("index.html", user_input=None, messages=None, error=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)

