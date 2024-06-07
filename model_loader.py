import pickle
def load_model():
    try:
        with open('saved_model.pkl', 'rb') as f:  # Or 'saved_model.joblib' for joblib
            model = pickle.load(f)
            print(model)
        return model
    except FileNotFoundError:
        return None
