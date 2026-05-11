#!/bin/sh
set -e
if [ ! -f /app/models/model.joblib ]; then
  echo "No model found; generating data and training..."
  python -m segmentation_mlops.data.make_dataset
  python -m segmentation_mlops.train.train_model || echo "Training failed; start API anyway for debugging."
fi
exec "$@"
