import argparse

def get_args():
    p = argparse.ArgumentParser(description="Incremental uplift training")
    # IO
    p.add_argument("--train_path", required=True)
    p.add_argument("--test_path",  required=True)
    p.add_argument("--data_dir",   default="data")
    p.add_argument("--model_dir",  default="models")
    p.add_argument("--report_dir", default="reports")

    # Model
    p.add_argument("--repr_layers", type=int,   default=2)
    p.add_argument("--repr_units",  type=int,   default=20)
    p.add_argument("--head_units",  type=int,   nargs=2, default=[20,20])
    p.add_argument("--l2",          type=float, default=1e-2)

    # Training
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--batch_size",  type=int,   default=1000)
    p.add_argument("--epochs",      type=int,   default=20)
    p.add_argument("--lambd",       type=float, default=.1)   # imbalance loss weight
    p.add_argument("--beta",        type=float, default=.1)   # est_t loss weight

    return p.parse_args()