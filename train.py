from startup_churn_classifier.training import run_training_pipeline


if __name__ == "__main__":
    summary = run_training_pipeline()
    print(summary)
