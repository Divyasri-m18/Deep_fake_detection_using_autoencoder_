import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Deepfake Video Detector CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Preprocess
    parser_prep = subparsers.add_parser("preprocess", help="Extract faces from videos")
    
    # Train
    parser_train = subparsers.add_parser("train", help="Train the Autoencoder model")
    
    # Evaluate
    parser_eval = subparsers.add_parser("evaluate", help="Evaluate model on processed data")
    
    # Predict
    parser_pred = subparsers.add_parser("predict", help="Predict if a video is Real or Fake")
    parser_pred.add_argument("--video", type=str, required=True, help="Path to video file")
    
    # UI
    parser_ui = subparsers.add_parser("ui", help="Launch Streamlit UI")
    
    args = parser.parse_args()
    
    if args.command == "preprocess":
        from preprocessing.extract_frames import main as preprocess_main
        print("Starting preprocessing...")
        preprocess_main()
        
    elif args.command == "train":
        from training.train_autoencoder import train
        print("Starting training...")
        train()
        
    elif args.command == "evaluate":
        from training.evaluate import main as eval_main
        print("Starting evaluation...")
        eval_main()
        
    elif args.command == "predict":
        from inference.predict_video import VideoPredictor
        predictor = VideoPredictor()
        results = predictor.predict(args.video)
        print(f"Prediction: {results['prediction']}")
        print(f"Confidence: {results['confidence']:.2f}%")
        print(f"Mean Error: {results['mean_error']:.5f}")
        print(f"Threshold:  {results['threshold']:.5f}")
        
    elif args.command == "ui":
        print("Launching Streamlit App...")
        os.system("streamlit run app/streamlit_app.py")
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
