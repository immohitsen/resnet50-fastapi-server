# main.py
from data_loader import load_datasets
from train import build_finetuned_model, train_model

if __name__ == "__main__":
    # Step 1: Data load karein
    train_dataset, val_dataset = load_datasets()

    # Step 2: Model build karein
    model = build_finetuned_model()

    # Step 3: Model ko train karein
    trained_model = train_model(model, train_dataset, val_dataset)

    print("🎉🎉🎉 Fine-tuning complete! 🎉🎉🎉")