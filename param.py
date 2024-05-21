# 作者: not4ya
# 时间: 2023/10/7 21:00
import argparse


def parameter_parser():
    parser = argparse.ArgumentParser(description="Run myModel")

    parser.add_argument("--epoch",
                        type=int,
                        default=1000,
                        help="Number of training epochs. Default is 1000.")

    parser.add_argument("--k_fold",
                        type=int,
                        default=5,
                        help="Number of cross_validation. Default is 5.")

    parser.add_argument("--knn_nums",
                        type=int,
                        default=30,
                        help="Number of KNN neighbors. Default is 20.")

    parser.add_argument("--herb_number",
                        type=int,
                        default=1497,
                        help="Herb number. Default is 1497.")

    parser.add_argument("--target_number",
                        type=int,
                        default=5219,
                        help="Target number. Default is 5219.")

    parser.add_argument("--hidden_dim",
                        type=int,
                        default=128,
                        help="Hidden Layers feature dimensions. Default is 128.")

    parser.add_argument("--heads",
                        type=int,
                        default=3,
                        help="Heads of GAT. Default is 3.")

    parser.add_argument("--cnn_dim",
                        type=int,
                        default=128,
                        help="Out Channel of cnn combiner. Default is 128.")

    parser.add_argument("--dropout",
                        type=float,
                        default=0.3,
                        help="Dropout of the final embedding features. Default is 0.3.")

    return parser.parse_args()
