import os 
import numpy as np
import pandas as pd
import tensorflow as tf
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from evaluation.report import ReportGenerator

def evaluate_and_report(model, test_ds, cfg, name_new, val_auc):
    x_test, t_test, y_test = [], [], []
    for (x,t),y in test_ds:
        x_test.append(x)
        t_test.append(t)
        y_test.append(y)
    x_test = tf.concat(x_test,0)
    t_test = tf.concat(t_test,0)
    y_test = tf.concat(y_test,0)

    p_pred = model(x_test, training=False)[0].numpy()
    p_exact = pd.read_csv(cfg.test_path)[['p0','p1','p2']].values

    uplift_1 = p_exact[:,1:2]-p_exact[:,0:1]
    uplift_2 = p_exact[:,2:3]-p_exact[:,1:2]
    pehe = np.sqrt(np.mean(
        np.concatenate([p_pred[:,1:2]-p_pred[:,0:1]-uplift_1,
                        p_pred[:,2:3]-p_pred[:,1:2]-uplift_2])**2))

    # report
    os.makedirs(cfg.report_dir, exist_ok=True)
    rg = ReportGenerator([0,1,2], trt_type="deduction")
    df = rg.generate_report(
        pd.DataFrame(np.concatenate([p_pred, y_test.numpy(), t_test.numpy()],1),
                     columns=['0','1','2','y','t']),
        val_auc=val_auc,
        save_path=os.path.join(cfg.report_dir, f"predict_{name_new}"))
    df.insert(0,'date',f"metric{name_new}")
    df.insert(5,'PEHE',pehe)
    df.to_csv("out.csv", mode='a', header=not os.path.exists("out.csv"), index=False)
    return pehe