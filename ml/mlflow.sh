mlflow server \
  --backend-store-uri "$MLFLOW_TRACKING_URI" \
  --default-artifact-root "$MLFLOW_ARTIFACTS_URI" \
  --host 0.0.0.0 \
  --port 5001