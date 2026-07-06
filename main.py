from __future__ import annotations

import argparse

from data import load_data, split_wet_dry
from train import predict_submission, run_cross_validation, train_models


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_path",
        type=str,
        default="/content/train_dataset.csv",
        help="Path to train_dataset.csv",
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default="/content/test_dataset.csv",
        help="Path to test_dataset.csv",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="submission.csv",
        help="Path where submission CSV will be saved",
    )
    parser.add_argument(
        "--skip_cv",
        action="store_true",
        help="Skip cross-validation and train final models only",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("데이터 로딩 및 전처리 시작")
    train, test = load_data(args.train_path, args.test_path)

    print("wet/dry 데이터 분리 시작")
    split_data = split_wet_dry(train, test)

    if not args.skip_cv:
        print("wet 데이터 CV 시작")
        run_cross_validation(split_data["X_wet"], split_data["y_wet"], name="wet")

        print("dry 데이터 CV 시작")
        run_cross_validation(split_data["X_dry"], split_data["y_dry"], name="dry")

    print("최종 모델 학습 시작")
    model_wet, model_dry = train_models(split_data)

    print("예측 및 submission 저장 시작")
    submission = predict_submission(test, split_data, model_wet, model_dry)
    submission.to_csv(args.output_path, index=False)

    print(f"최종 예측 완료! → '{args.output_path}' 저장")
    print(submission.head())


if __name__ == "__main__":
    main()
