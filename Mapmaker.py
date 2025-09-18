# RUN THIS SCRIPT TO FIX YOUR EXISTING SAVED MODELS
# Voeg dit toe aan je script en run het

def create_manual_summary(timestamp="20250918_211758"):
    """
    Create summary file for existing saved models
    """
    import json
    import os
    
    # Check which files exist for this timestamp
    files_exist = {
        'models': os.path.exists(f"enhanced_models_{timestamp}.pkl"),
        'scalers': os.path.exists(f"enhanced_scalers_{timestamp}.pkl"), 
        'label_encoder': os.path.exists(f"label_encoder_{timestamp}.pkl"),
        'shap_weights': os.path.exists(f"shap_weights_{timestamp}.pkl"),
        'performance_tracker': os.path.exists(f"performance_tracker_{timestamp}.pkl"),
        'polynomial_transformer': os.path.exists(f"polynomial_transformer_{timestamp}.pkl"),
        'bayesian_models': os.path.exists(f"bayesian_models_{timestamp}.pkl")
    }
    
    print(f"Found files for timestamp {timestamp}:")
    for file_type, exists in files_exist.items():
        status = "✅" if exists else "❌"
        print(f"  {status} {file_type}")
    
    # Create summary
    summary = {
        'timestamp': timestamp,
        'models': ['xgboost', 'lightgbm', 'catboost', 'random_forest', 'voting_ensemble', 'stacking_ensemble'] if files_exist['models'] else [],
        'scalers': ['StandardScaler'] if files_exist['scalers'] else [],
        'has_shap_weights': files_exist['shap_weights'],
        'has_performance_tracker': files_exist['performance_tracker'],
        'has_polynomial_transformer': files_exist['polynomial_transformer'],
        'has_bayesian_models': files_exist['bayesian_models'],
        'has_lstm_models': False
    }
    
    # Save summary file
    with open(f"model_suite_summary_{timestamp}.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Summary file created: model_suite_summary_{timestamp}.json")
    print("Now you can use option 3 to load your models!")
    
    return summary

# RUN THIS:
if __name__ == "__main__":
    create_manual_summary("20250918_211758")  # Use your timestamp