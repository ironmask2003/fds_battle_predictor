from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import pandas as pd
import argparse

from load_dataset import *
from extract_features import *

def parse_args():
    """
    Funzione per il parsing degli argomenti da riga di comando.

    Returns:
        - args: parsed arguments
    """
    parser = argparse.ArgumentParser(description='Arguments')
    
    parser.add_argument("--show", action="store_true", default=False)
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--n_turns", type=int, default=2)
    parser.add_argument("--show_train", action="store_true", default=False)

    # Kaggle
    parser.add_argument("--kaggle", action="store_true", default=False, help="Check if train is on kaggle")

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    train_path, test_path = gen_path(args.kaggle)

    train_data = load_data(train_path, args.idx, args.n_turns, args.show)
    test_data = load_data(test_path, args.idx, args.n_turns, args.show_train)

    print("Processing training data...")
    train_df = create_simple_features(train_data)

    corr = train_df.corr()
    print(corr['player_won'].sort_values(ascending=False))

    print("\nProcessing test data...")
    test_data = []
    with open(test_path, 'r') as f:
        for line in f:
            test_data.append(json.loads(line))
    test_df = create_simple_features(test_data)

    features = [col for col in train_df.columns if col not in ['battle_id', 'player_won']]
    X_train = train_df[features]
    y_train = train_df['player_won']

    X_test = test_df[features]

    # Suddividiamo i dati train in train + validation
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42
    )

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)

    # Predizione sulla validation
    val_preds = model.predict(X_val)
    train_preds = model.predict(X_train_sub)

    # Calcoliamo accuracy
    accuracy_train = accuracy_score(y_train_sub, train_preds)
    print(f"Train Accuracy: {accuracy_train:.4f}")

    accuracy = accuracy_score(y_val, val_preds)
    print(f"Validation Accuracy: {accuracy:.4f}")

    val_results = pd.DataFrame({
        'y_true': y_val,
        'y_pred': val_preds
    })

    # Filtriamo solo gli errori
    errors = val_results[val_results['y_true'] != val_results['y_pred']]

    print(f"Numero di errori: {len(errors)}")
    print(errors.head())

    cerate_submission(model, X_test, test_df)

def cerate_submission(model, X_test, test_df):
    # Make predictions on the test data
    print("Generating predictions on the test set...")
    test_predictions = model.predict(X_test)

    # Create the submission DataFrame
    submission_df = pd.DataFrame({
        'battle_id': test_df['battle_id'],
        'player_won': test_predictions
    })

    # Save the DataFrame to a .csv file
    submission_df.to_csv('submission.csv', index=False)

    print("\n'submission.csv' file created successfully!")

if __name__ == "__main__":
    main()