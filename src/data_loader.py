import os
import pandas as pd
import numpy as np
import tensorflow as tf

TRT_LIST = [0,1,2]

def read_csv(path):
    df = pd.read_csv(path)
    x  = df[['x1','x2']].astype(np.float32).values
    y  = df['y'].astype(np.float32).values.reshape(-1,1)
    t  = df['t'].astype(int)
    t1 = np.eye(len(TRT_LIST))[t].astype(np.float32)
    return x, y, t1

def reservoir_sample(df, k):
    return df.sample(n=k, random_state=42).reset_index(drop=True)

def build_dataset(x, y, t, batch, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices(((x, t), y))
    if shuffle: ds = ds.shuffle(buffer_size=len(x))
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)

def prepare_incremental(cfg):
    os.makedirs(cfg.data_dir, exist_ok=True)
    name_new = os.path.basename(cfg.test_path).split('.')[0].split('_')[-1]
    name_pre = str(int(name_new)-1)

    # read previous round meta
    txt_path = os.path.join(cfg.data_dir, "txt", f"{name_pre}.txt")
    if os.path.exists(txt_path):
        with open(txt_path) as f:
            model_teacher_path = f.readline().strip()
            n_pre              = int(f.readline())
    else:
        model_teacher_path = None
        n_pre = 0

    # read data
    data_new = pd.read_csv(cfg.train_path)
    data_pre = pd.read_csv(os.path.join(cfg.data_dir, "csv", f"{name_pre}.csv")) \
               if n_pre else pd.DataFrame()

    n_new  = len(data_new) * 10
    n_tot  = n_new + n_pre
    k_new  = int(100 * n_new / n_tot) if n_tot else len(data_new)
    k_pre  = int(100 * n_pre / n_tot) if n_tot else 0

    data_new_s = reservoir_sample(data_new, k_new)
    data_pre_s = reservoir_sample(data_pre, k_pre) if k_pre else pd.DataFrame()
    data_s     = pd.concat([data_new_s, data_pre_s], ignore_index=True)

    csv_path = os.path.join(cfg.data_dir, "csv", f"{name_new}.csv")
    data_s.to_csv(csv_path, index=True)

    x_new, y_new, t_new = read_csv(cfg.train_path)
    x_pre, y_pre, t_pre = read_csv(csv_path) if n_pre else (None,None,None)
    x_test, y_test, t_test = read_csv(cfg.test_path)

    return (x_new, y_new, t_new,
            x_pre, y_pre, t_pre,
            x_test, y_test, t_test,
            model_teacher_path, name_new)