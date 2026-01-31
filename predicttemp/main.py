import argparse
from pathlib import Path

from predictor import TemperaturePredictor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and run a next-day max temperature predictor for NYC."
    )
    repo_root = Path(__file__).resolve().parents[1]
    default_data = repo_root / "open-meteo-40.70N74.00W51m (2).csv"
    default_model = repo_root / "models" / "temperature_predictor.joblib"
    parser.add_argument(
        "--data",
        type=Path,
        default=default_data,
        help="Path to the Open-Meteo CSV data file",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=default_model,
        help="Path to save/load the trained model artifact",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train a new model and save it to --model-path",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate the model on a held-out split",
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Predict the next day's maximum temperature",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    predictor = TemperaturePredictor(args.data)

    if args.train:
        predictor.train_model()
        predictor.save_model(args.model_path)
    else:
        if args.model_path.exists():
            predictor.load_model(args.model_path)
        else:
            raise FileNotFoundError(
                f"Model file not found at {args.model_path}. Run with --train first."
            )

    if args.evaluate:
        metrics = predictor.evaluate_model()
        print(f"Evaluation MSE={metrics['mse']:.2f}, MAE={metrics['mae']:.2f}")

    if args.predict:
        prediction = predictor.predict_next_day()
        print(f"Predicted next-day max temperature: {prediction:.2f} °C")

    if not any([args.train, args.evaluate, args.predict]):
        parser.print_help()


if __name__ == "__main__":
    main()
