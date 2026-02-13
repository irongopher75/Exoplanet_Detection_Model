import kagglehub
import os
from pathlib import Path

def upload_to_kaggle():
    print("🚀 Preparing for Kaggle migration...")
    
    # Set Kaggle credentials
    # The token you provided is KGAT_ + a 32-character key.
    # Some parts of the Kaggle API prefer KAGGLE_KEY.
    os.environ['KAGGLE_USERNAME'] = "vishnuupanicker"
    os.environ['KAGGLE_KEY'] = "be80a85e16188167b396e8f05a84c682" # Stripped KGAT_ prefix
    os.environ['KAGGLE_API_TOKEN'] = "KGAT_be80a85e16188167b396e8f05a84c682"
    
    # Note: Skipping kagglehub.login()

    # Configuration
    USER_HANDLE = "vishnuupanicker"
    MODEL_SLUG = "exoplanet-detection-pinn" # Descriptive slug
    FRAMEWORK = "pytorch" # Use pytorch instead of keras
    VARIATION = "default"
    
    # Path to your checkpoints folder
    LOCAL_MODEL_DIR = os.path.abspath('outputs/checkpoints')
    
    # Check if directory is empty (we just cleaned it)
    if not os.path.exists(LOCAL_MODEL_DIR) or not os.listdir(LOCAL_MODEL_DIR):
        print(f"⚠️ Warning: {LOCAL_MODEL_DIR} is empty. Training a few epochs locally first is recommended.")
        # Create dummy directory if needed just to setup the handle
        Path(LOCAL_MODEL_DIR).mkdir(parents=True, exist_ok=True)

    handle = f"{USER_HANDLE}/{MODEL_SLUG}/{FRAMEWORK}/{VARIATION}"
    
    print(f"📦 Uploading model to: {handle}")
    
    try:
        kagglehub.model_upload(
            handle = handle,
            local_model_dir = LOCAL_MODEL_DIR,
            version_notes = 'Initial PINN Model Migration 2026-02-14'
        )
        print("✅ Success! Your model is now on Kaggle.")
    except Exception as e:
        print(f"❌ Upload failed: {e}")

if __name__ == "__main__":
    upload_to_kaggle()
