# OSS Health Predictor: code package (Phases 8 to 19)

Extract into your project root (C:\Users\SANCHITA\Documents\oss-health-predictor).

Run order (always from the project root, venv active):
1. python -m ml.features.build_features
2. python -m ml.models.train_models
3. python -m ml.models.train_lstm            (pip install tensorflow first)
4. python -m ml.explain.shap_analysis
5. python -m ml.evaluation.run_evaluation
6. uvicorn backend.app.main:app --reload --port 8000     (MongoDB service must be running)
7. frontend: see the frontend setup in the chat instructions (npm, port 5173)
8. pytest -q

Design rules: features use only past months; temporal split with 3-month gap;
model chosen by CV on the training period, never by test score; grouped CV by repository.
