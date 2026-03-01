from transformers import pipeline

MODEL_NAME = "google/vit-base-patch16-224"

def download_model():
    print(f"Downloading model {MODEL_NAME} (this may take a few minutes)...")
    classifier = pipeline("image-classification", model=MODEL_NAME)
    print("Model downloaded and cached locally.")

if __name__ == "__main__":
    download_model()
