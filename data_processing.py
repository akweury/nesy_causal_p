# Created by MacBook Pro at 18.06.25
import pandas as pd

import config

prox_path = config.wandb_path / "prox.csv"
simi_path = config.wandb_path / "simi.csv"
closure_path = config.wandb_path / "closure.csv"
cont_path = config.wandb_path / "cont.csv"
symm_path = config.wandb_path / "symm.csv"

prox = pd.read_csv(prox_path)
simi = pd.read_csv(simi_path)
closure = pd.read_csv(closure_path)
cont = pd.read_csv(cont_path)
symm = pd.read_csv(symm_path)

prox_mean =  prox["prox_01 - hard_ogc_acc_avg"].mean()
prox_std =  prox["prox_01 - hard_ogc_acc_avg"].std()
simi_mean  = simi["simi_01 - hard_ogc_acc_avg"].mean()
simi_std  = simi["simi_01 - hard_ogc_acc_avg"].std()
closure_mean\
    = closure["closure_01 - hard_ogc_acc_avg"].mean()
closure_std  = closure["closure_01 - hard_ogc_acc_avg"].std()

cont_mean  = cont["cont_01 - hard_ogc_acc_avg"].mean()
cont_std  = cont["cont_01 - hard_ogc_acc_avg"].std()

prin = "prox"
prox_o_mean = pd.read_csv(config.wandb_path / f"{prin}_o.csv")[f"{prin}_01 - hard_obj_acc_avg"].mean()
prox_o_std = pd.read_csv(config.wandb_path / f"{prin}_o.csv")[f"{prin}_01 - hard_obj_acc_avg"].std()

prox_oc_mean = pd.read_csv(config.wandb_path / f"{prin}_oc.csv")[f"{prin}_01 - hard_obj_calib_acc_avg"].mean()
prox_oc_std = pd.read_csv(config.wandb_path / f"{prin}_oc.csv")[f"{prin}_01 - hard_obj_calib_acc_avg"].std()

prox_og_mean = pd.read_csv(config.wandb_path / f"{prin}_og.csv")[f"{prin}_01 - hard_og_acc_avg"].mean()
prox_og_std = pd.read_csv(config.wandb_path / f"{prin}_og.csv")[f"{prin}_01 - hard_og_acc_avg"].std()



prin = "simi"
simi_o_mean = pd.read_csv(config.wandb_path / f"{prin}_o.csv")[f"{prin}_01 - hard_obj_acc_avg"].mean()
simi_o_std = pd.read_csv(config.wandb_path / f"{prin}_o.csv")[f"{prin}_01 - hard_obj_acc_avg"].std()

simi_oc_mean = pd.read_csv(config.wandb_path / f"{prin}_oc.csv")[f"{prin}_01 - hard_obj_calib_acc_avg"].mean()
simi_oc_std = pd.read_csv(config.wandb_path / f"{prin}_oc.csv")[f"{prin}_01 - hard_obj_calib_acc_avg"].std()

simi_og_mean = pd.read_csv(config.wandb_path / f"{prin}_og.csv")[f"{prin}_01 - hard_og_acc_avg"].mean()
simi_og_std = pd.read_csv(config.wandb_path / f"{prin}_og.csv")[f"{prin}_01 - hard_og_acc_avg"].std()



prin = "cont"
cont_o_mean = pd.read_csv(config.wandb_path / f"{prin}_o.csv")[f"{prin}_01 - hard_obj_acc_avg"].mean()
cont_o_std = pd.read_csv(config.wandb_path / f"{prin}_o.csv")[f"{prin}_01 - hard_obj_acc_avg"].std()

cont_oc_mean = pd.read_csv(config.wandb_path / f"{prin}_oc.csv")[f"{prin}_01 - hard_obj_calib_acc_avg"].mean()
cont_oc_std = pd.read_csv(config.wandb_path / f"{prin}_oc.csv")[f"{prin}_01 - hard_obj_calib_acc_avg"].std()

cont_og_mean = pd.read_csv(config.wandb_path / f"{prin}_og.csv")[f"{prin}_01 - hard_og_acc_avg"].mean()
cont_og_std = pd.read_csv(config.wandb_path / f"{prin}_og.csv")[f"{prin}_01 - hard_og_acc_avg"].std()

prin = "closure"
closure_o_mean = pd.read_csv(config.wandb_path / f"{prin}_o.csv")[f"{prin}_01 - hard_obj_acc_avg"].mean()
closure_o_std = pd.read_csv(config.wandb_path / f"{prin}_o.csv")[f"{prin}_01 - hard_obj_acc_avg"].std()

closure_oc_mean = pd.read_csv(config.wandb_path / f"{prin}_oc.csv")[f"{prin}_01 - hard_obj_calib_acc_avg"].mean()
closure_oc_std = pd.read_csv(config.wandb_path / f"{prin}_oc.csv")[f"{prin}_01 - hard_obj_calib_acc_avg"].std()

closure_og_mean = pd.read_csv(config.wandb_path / f"{prin}_og.csv")[f"{prin}_01 - hard_og_acc_avg"].mean()
closure_og_std = pd.read_csv(config.wandb_path / f"{prin}_og.csv")[f"{prin}_01 - hard_og_acc_avg"].std()


symm_mean  = symm["symm_01 - hard_ogc_acc_avg"].mean()
symm_std  = symm["symm_01 - hard_ogc_acc_avg"].std()

symm_o_mean = pd.read_csv(config.wandb_path / "symm_o.csv")["symm_01 - hard_obj_acc_avg"].mean()
symm_o_std = pd.read_csv(config.wandb_path / "symm_o.csv")["symm_01 - hard_obj_acc_avg"].std()

symm_oc_mean = pd.read_csv(config.wandb_path / "symm_oc.csv")["symm_01 - hard_obj_calib_acc_avg"].mean()
symm_oc_std = pd.read_csv(config.wandb_path / "symm_oc.csv")["symm_01 - hard_obj_calib_acc_avg"].std()

symm_og_mean = pd.read_csv(config.wandb_path / "symm_og.csv")["symm_01 - hard_og_acc_avg"].mean()
symm_og_std = pd.read_csv(config.wandb_path / "symm_og.csv")["symm_01 - hard_og_acc_avg"].std()


print("")