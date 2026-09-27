from flask import Flask, render_template, jsonify, request
import requests
import pandas as pd
import joblib
import os
from datetime import datetime
from zoneinfo import ZoneInfo


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# PROJECT PATHS
# ============================================================

APP_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_DIR = os.path.dirname(
    APP_DIR
)


# Final model folder
MODEL_DIR = os.path.join(
    PROJECT_DIR,
    "model"
)


MODEL_PATH = os.path.join(
    MODEL_DIR,
    "saltpan_random_forest.pkl"
)


CONFIG_PATH = os.path.join(
    MODEL_DIR,
    "model_config.pkl"
)


# ============================================================
# LOAD FINAL MODEL
# ============================================================

try:

    model = joblib.load(
        MODEL_PATH
    )

    model_config = joblib.load(
        CONFIG_PATH
    )

    feature_columns = model_config[
        "feature_columns"
    ]

    print("======================================")
    print("FINAL RANDOM FOREST MODEL LOADED")
    print("======================================")

    print(
        "Model:",
        model_config["model_name"]
    )

    print(
        "Number of trees:",
        model_config["n_estimators"]
    )

    print(
        "Features:",
        feature_columns
    )

    print(
        "Classification rule:",
        model_config["classification_rule"]
    )

    print("======================================")


except Exception as e:

    print("======================================")
    print("MODEL LOADING FAILED")
    print("======================================")
    print(e)

    raise


# ============================================================
# TIMEZONE
# ============================================================

TIMEZONE = ZoneInfo(
    "Asia/Kolkata"
)


# ============================================================
# LOCATIONS
# ============================================================

LOCATIONS = {

    0: {
        "name": "Thoothukudi Coastal Area",
        "latitude": 8.7522,
        "longitude": 78.1293,
        "elevation": 6
    },

    1: {
        "name": "Tharuvaikulam Coastal Area",
        "latitude": 8.8928,
        "longitude": 78.1088,
        "elevation": 14
    },

    2: {
        "name": "Vembar",
        "latitude": 9.1037,
        "longitude": 78.3896,
        "elevation": 6
    },

    3: {
        "name": "Thoothukudi District Coastal Area - East",
        "latitude": 9.0334,
        "longitude": 78.2439,
        "elevation": 4
    },

    4: {
        "name": "Kayalpattinam Coastal Area",
        "latitude": 8.5413,
        "longitude": 78.0826,
        "elevation": 3
    }
}


# ============================================================
# OPEN-METEO
# ============================================================

OPEN_METEO_URL = (
    "https://api.open-meteo.com/v1/forecast"
)


# ============================================================
# GET LOCATION
# ============================================================

def get_location(location_id):

    try:

        location_id = int(
            location_id
        )

    except:

        location_id = 0


    if location_id not in LOCATIONS:

        location_id = 0


    return LOCATIONS[
        location_id
    ]


# ============================================================
# GET WEATHER FORECAST
# ============================================================

def get_forecast(location_id):

    location = get_location(
        location_id
    )


    params = {

        "latitude":
            location["latitude"],

        "longitude":
            location["longitude"],

        "hourly": [

            "temperature_2m",

            "relative_humidity_2m",

            "precipitation",

            "wind_speed_10m"

        ],

        "wind_speed_unit":
            "ms",

        "forecast_days":
            7,

        "timezone":
            "Asia/Kolkata"
    }


    response = requests.get(

        OPEN_METEO_URL,

        params=params,

        timeout=30

    )


    response.raise_for_status()


    data = response.json()


    hourly = data.get(
        "hourly",
        {}
    )


    df = pd.DataFrame({

        "Date_Time":

            pd.to_datetime(
                hourly.get(
                    "time",
                    []
                )
            ),

        "temperature_2m (°C)":

            hourly.get(
                "temperature_2m",
                []
            ),

        "relative_humidity_2m (%)":

            hourly.get(
                "relative_humidity_2m",
                []
            ),

        "precipitation (mm)":

            hourly.get(
                "precipitation",
                []
            ),

        "wind_speed_10m (m/s)":

            hourly.get(
                "wind_speed_10m",
                []
            )

    })


    return df


# ============================================================
# RANDOM FOREST PREDICTION
# ============================================================

def predict_forecast(df):

    if df.empty:

        return df


    # Make sure all required features exist

    missing_features = [

        column

        for column in feature_columns

        if column not in df.columns

    ]


    if missing_features:

        raise ValueError(

            "Missing model features: "
            +
            ", ".join(
                missing_features
            )

        )


    # Exact feature order used during training

    X = df[
        feature_columns
    ].copy()


    # Prediction

    predictions = model.predict(
        X
    )


    # Random Forest probabilities

    probabilities = model.predict_proba(
        X
    )


    df["Prediction"] = (
        predictions
    )


    df[
        "Suitable_Probability"
    ] = (

        probabilities[:, 1]
        *
        100

    )


    df[
        "Not_Suitable_Probability"
    ] = (

        probabilities[:, 0]
        *
        100

    )


    df["Status"] = (

        df["Prediction"]

        .map({

            0: "Not Suitable",

            1: "Suitable"

        })

    )


    return df


# ============================================================
# MANUAL PARAMETER PREDICTION
# ============================================================

def predict_manual_parameters(
    temperature,
    humidity,
    wind,
    rainfall
):

    values = {

        "temperature_2m (°C)":
            temperature,

        "relative_humidity_2m (%)":
            humidity,

        "wind_speed_10m (m/s)":
            wind,

        "precipitation (mm)":
            rainfall

    }


    input_df = pd.DataFrame(
        [values]
    )


    # Exact feature order

    input_df = input_df[
        feature_columns
    ]


    # Prediction

    prediction = model.predict(
        input_df
    )[0]


    # Probability

    probabilities = model.predict_proba(
        input_df
    )[0]


    not_suitable_probability = (

        probabilities[0]
        *
        100

    )


    suitable_probability = (

        probabilities[1]
        *
        100

    )


    status = (

        "Suitable"

        if prediction == 1

        else

        "Not Suitable"

    )


    return {

        "prediction":
            int(
                prediction
            ),

        "status":
            status,

        "suitable_probability":
            round(
                float(
                    suitable_probability
                ),
                2
            ),

        "not_suitable_probability":
            round(
                float(
                    not_suitable_probability
                ),
                2
            ),

        "temperature":
            temperature,

        "humidity":
            humidity,

        "wind":
            wind,

        "rainfall":
            rainfall

    }


# ============================================================
# CREATE TIME RANGES
# ============================================================

def create_time_ranges(hours):

    if not hours:

        return []


    hours = sorted(
        hours
    )


    ranges = []


    start = hours[0]

    previous = hours[0]


    for current in hours[1:]:

        if current == previous + 1:

            previous = current

        else:

            ranges.append(

                f"{start}:00 - "
                f"{previous + 1}:00"

            )

            start = current

            previous = current


    ranges.append(

        f"{start}:00 - "
        f"{previous + 1}:00"

    )


    return ranges


# ============================================================
# DAILY SUMMARY
# ============================================================

def create_daily_summary(df):

    if df.empty:

        return []


    # Daytime period:
    # 6 AM to 6 PM

    daytime = df[

        (df["Date_Time"].dt.hour >= 6)

        &

        (df["Date_Time"].dt.hour <= 18)

    ].copy()


    summaries = []


    for date, group in daytime.groupby(

        daytime[
            "Date_Time"
        ].dt.date

    ):


        suitable_rows = group[

            group["Prediction"] == 1

        ]


        not_suitable_rows = group[

            group["Prediction"] == 0

        ]


        suitable_hours = [

            int(
                value.hour
            )

            for value in suitable_rows[
                "Date_Time"
            ]

        ]


        suitable_count = len(
            suitable_rows
        )


        total_count = len(
            group
        )


        # If at least one daytime hour
        # is suitable, classify the day
        # as Suitable.

        if suitable_count > 0:

            status = "Suitable"

        else:

            status = "Not Suitable"


        if total_count > 0:

            suitable_percentage = (

                suitable_count
                /
                total_count
                *
                100

            )

        else:

            suitable_percentage = 0


        average_probability = (

            group[
                "Suitable_Probability"
            ].mean()

            if total_count > 0

            else 0

        )


        summaries.append({

            "date":

                date.strftime(
                    "%Y-%m-%d"
                ),

            "display_date":

                date.strftime(
                    "%d %b"
                ),

            "day":

                date.strftime(
                    "%A"
                ),

            "status":
                status,

            "suitable_hours":
                suitable_count,

            "not_suitable_hours":

                len(
                    not_suitable_rows
                ),

            "total_hours":
                total_count,

            "suitable_percentage":

                round(
                    suitable_percentage,
                    1
                ),

            "average_probability":

                round(
                    average_probability,
                    1
                ),

            "suitable_times":

                create_time_ranges(
                    suitable_hours
                )

        })


    return summaries


# ============================================================
# CURRENT PREDICTION FROM OPEN-METEO
# ============================================================

def get_current_prediction(
    location_id
):

    location = get_location(
        location_id
    )


    df = get_forecast(
        location_id
    )


    df = predict_forecast(
        df
    )


    if df.empty:

        return None


    now = datetime.now(
        TIMEZONE
    )


    current_hour = pd.Timestamp(

        now.strftime(
            "%Y-%m-%d %H:00:00"
        )

    )


    df["difference"] = abs(

        df["Date_Time"]
        -
        current_hour

    )


    current_row = df.loc[

        df[
            "difference"
        ].idxmin()

    ]


    return {

        "current_time":

            now.strftime(
                "%d %B %Y, %I:%M:%S %p"
            ),

        "forecast_time":

            current_row[
                "Date_Time"
            ].strftime(
                "%d %B %Y, %I:%M %p"
            ),

        "temperature":

            round(
                float(
                    current_row[
                        "temperature_2m (°C)"
                    ]
                ),
                1
            ),

        "humidity":

            round(
                float(
                    current_row[
                        "relative_humidity_2m (%)"
                    ]
                ),
                1
            ),

        "rainfall":

            round(
                float(
                    current_row[
                        "precipitation (mm)"
                    ]
                ),
                2
            ),

        "wind":

            round(
                float(
                    current_row[
                        "wind_speed_10m (m/s)"
                    ]
                ),
                2
            ),

        "status":

            current_row[
                "Status"
            ],

        "suitable_probability":

            round(
                float(
                    current_row[
                        "Suitable_Probability"
                    ]
                ),
                1
            ),

        "not_suitable_probability":

            round(
                float(
                    current_row[
                        "Not_Suitable_Probability"
                    ]
                ),
                1
            ),

        "latitude":

            location[
                "latitude"
            ],

        "longitude":

            location[
                "longitude"
            ],

        "elevation":

            location[
                "elevation"
            ],

        "location_name":

            location[
                "name"
            ]

    }


# ============================================================
# SERIALIZE HOURLY DATA
# ============================================================

def serialize_hourly(df):

    result = []


    for _, row in df.iterrows():

        result.append({

            "datetime":

                row[
                    "Date_Time"
                ].strftime(
                    "%Y-%m-%d %H:%M"
                ),

            "date":

                row[
                    "Date_Time"
                ].strftime(
                    "%Y-%m-%d"
                ),

            "hour":

                int(
                    row[
                        "Date_Time"
                    ].hour
                ),

            "display_time":

                row[
                    "Date_Time"
                ].strftime(
                    "%I:%M %p"
                ),

            "temperature":

                round(
                    float(
                        row[
                            "temperature_2m (°C)"
                        ]
                    ),
                    1
                ),

            "humidity":

                round(
                    float(
                        row[
                            "relative_humidity_2m (%)"
                        ]
                    ),
                    1
                ),

            "rainfall":

                round(
                    float(
                        row[
                            "precipitation (mm)"
                        ]
                    ),
                    2
                ),

            "wind":

                round(
                    float(
                        row[
                            "wind_speed_10m (m/s)"
                        ]
                    ),
                    2
                ),

            "status":

                row[
                    "Status"
                ],

            "suitable_probability":

                round(
                    float(
                        row[
                            "Suitable_Probability"
                        ]
                    ),
                    1
                ),

            "not_suitable_probability":

                round(
                    float(
                        row[
                            "Not_Suitable_Probability"
                        ]
                    ),
                    1
                )

        })


    return result


# ============================================================
# HOME
# ============================================================

@app.route("/")
def index():

    return render_template(

        "index.html",

        locations=LOCATIONS

    )


# ============================================================
# CURRENT WEATHER API
# ============================================================

@app.route(
    "/current_prediction"
)
def current_prediction():

    try:

        location_id = request.args.get(

            "location_id",

            default=0,

            type=int

        )


        result = get_current_prediction(

            location_id

        )


        if result is None:

            return jsonify({

                "success":
                    False,

                "message":
                    "Current weather unavailable."

            }), 500


        return jsonify({

            "success":
                True,

            "data":
                result

        })


    except Exception as e:

        print(
            "Current prediction error:",
            e
        )


        return jsonify({

            "success":
                False,

            "message":
                str(e)

        }), 500


# ============================================================
# MANUAL PREDICTION API
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        data = request.get_json()


        if not data:

            return jsonify({

                "success":
                    False,

                "message":
                    "No input data received."

            }), 400


        temperature = float(

            data.get(
                "temperature",
                0
            )

        )


        humidity = float(

            data.get(
                "humidity",
                0
            )

        )


        wind = float(

            data.get(
                "wind",
                0
            )

        )


        rainfall = float(

            data.get(
                "rainfall",
                0
            )

        )


        result = predict_manual_parameters(

            temperature,

            humidity,

            wind,

            rainfall

        )


        return jsonify({

            "success":
                True,

            "data":
                result

        })


    except Exception as e:

        print(
            "Manual prediction error:",
            e
        )


        return jsonify({

            "success":
                False,

            "message":
                str(e)

        }), 500


# ============================================================
# FORECAST API
# ============================================================

@app.route(
    "/forecast"
)
def forecast():

    try:

        location_id = request.args.get(

            "location_id",

            default=0,

            type=int

        )


        location = get_location(
            location_id
        )


        df = get_forecast(
            location_id
        )


        df = predict_forecast(
            df
        )


        daily = create_daily_summary(
            df
        )


        hourly = serialize_hourly(
            df
        )


        return jsonify({

            "success":
                True,

            "location": {

                "name":
                    location["name"],

                "latitude":
                    location["latitude"],

                "longitude":
                    location["longitude"],

                "elevation":
                    location["elevation"]

            },

            "daily":
                daily,

            "hourly":
                hourly

        })


    except Exception as e:

        print(
            "Forecast error:",
            e
        )


        return jsonify({

            "success":
                False,

            "message":
                str(e)

        }), 500


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(

        host="127.0.0.1",

        port=5000,

        debug=True

    )