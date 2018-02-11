# Simple-model-deployment

This is a simple example of deploying a machine learning model using pickling and Flask API.

To test API paste the following example into Postman to get a prediction:

{
	"sl": "5",
	"sw": "5",
	"pl": "5",
	"pw": "4"
}

Or using curl:

curl -d '{"sl": "5","sw": "5","pl": "5","pw": "4"}' -X POST http://localhost:5000/predict
