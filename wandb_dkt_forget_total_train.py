import argparse
import os

from wandb_train import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="codeforces_strong")
    '''
    parser.add_argument("--code_type", type=str, default="NONE")
    parser.add_argument("--problem_type", type=str, default="NONE")
    '''

    parser.add_argument("--model_name", type=str, default="dkt_forget_total")
    parser.add_argument("--emb_type", type=str, default="qid")
    parser.add_argument("--save_dir", type=str, default="saved_model")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=100)
    # model params

    parser.add_argument("--forget_window", type=float, default=(1/24,1,7,30))
    # parser.add_argument("--forget_window", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=(0.2, 0.2))
    parser.add_argument("--emb_size", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=100)
    parser.add_argument("--size_q_neighbors", type=int, default=5)
    parser.add_argument("--size_s_neighbors", type=int, default=5)
    parser.add_argument("--aggregator", type=str, default="sum")

    parser.add_argument("--use_wandb", type=int, default=1)
    parser.add_argument("--add_uuid", type=int, default=1)
    args = parser.parse_args()

    params = vars(args)
    main(params)
    # os.system('shutdown')
