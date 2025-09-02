import os, tensorflow as tf
from src.config import get_args
from src.data_loader import prepare_incremental, build_dataset, read_csv
from src.model import build
from src.train import Trainer

def main():
    cfg = get_args()
    (x_new, y_new, t_new,
     x_pre, y_pre, t_pre,
     x_test, y_test, t_test,
     teacher_path, name_new) = prepare_incremental(cfg)

    # datasets
    train_ds = build_dataset(x_new, y_new, t_new, cfg.batch_size)
    val_ds   = build_dataset(x_test, y_test, t_test, cfg.batch_size, shuffle=False)

    # models
    model  = build(x_new.shape[1], cfg)
    model0 = build(x_new.shape[1], cfg)  # teacher
    if teacher_path:
        model.load_weights(teacher_path)
        model0.load_weights(teacher_path)

    # train
    trainer = Trainer(cfg)
    trainer.compile(model, model0)
    ckpt_path = os.path.join(cfg.model_dir, f"model_{name_new}", "export_model.ckpt")
    val_auc = trainer.fit(train_ds, val_ds,
                          x_pre if x_pre is not None else x_new,
                          t_pre if t_pre is not None else t_new,
                          cfg.epochs, ckpt_path)

    # save meta
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    with open(os.path.join(cfg.data_dir,"txt",f"{name_new}.txt"),"w") as f:
        f.write(ckpt_path+"\n")
        f.write(str(len(x_new)*10 + (len(x_pre)*10 if x_pre is not None else 0))+"\n")

    # report
    from src.eval import evaluate_and_report
    evaluate_and_report(model, val_ds, cfg, name_new, val_auc)

if __name__ == "__main__":
    main()