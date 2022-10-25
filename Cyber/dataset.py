import pandas as pd
import numpy  as np

def power(val):
    ret  = "Power_Module_6"
    if val >    0.0 and val <=  2000.0:
        ret  = "Power_Module_1"
    if val > 2000.0 and val <=  4000.0:
        ret  = "Power_Module_2"
    if val > 4000.0 and val <=  6000.0:
        ret  = "Power_Module_3"
    if val > 4000.0 and val <=  8000.0:
        ret  = "Power_Module_4"
    if val > 8000.0 and val <= 10000.0:
        ret  = "Power_Module_5"
    return ret

dat0                            = pd.read_csv("/mnt/data/csv/airconditioning.csv")

dat1                            = dat0.copy()
dat5                            = pd.read_csv("/mnt/data/csv/condenser.csv")
dat6                            = dat5.merge(dat1,how="cross")
dat1                            = dat6.rename(columns={"Condenser_Coil_x":"Part"})
dat1                            = dat1.drop(["TOn","Noise_level(db)","STAR","Ratings","Price","Image_url","Condenser_Coil_y"],axis=1)
dat1["SKU" ]                    = list(range(1,len(dat1)+1))
dat1["Power_Consumption(Watts)"]= list(map(power,dat1["Power_Consumption(Watts)"].astype(np.float64)))
cols                            = list(dat1.columns)
dat2                            = dat1.copy()[[cols[c] for c in range(len(cols)) if c not in [cols.index("Power_Consumption(Watts)"),cols.index("RefrigeranT")]]]
dat2["cost"]                    = np.round(dat2["cost"].astype(np.float64),2)
dat3                            = dat1.copy()[[cols[c] for c in range(len(cols)) if c not in [cols.index("Part"),cols.index("RefrigeranT")]]]
dat3                            = dat3.rename(columns={"Power_Consumption(Watts)":"Part"})
dat3["SKU" ]                    = dat3["SKU"].astype(np.int8) +    len(dat0)
dat3["cost"]                    = np.round(dat3["cost"].astype(np.float64)*np.random.sample(size=dat3["cost"].to_numpy().shape),2)
dat4                            = dat1.copy()[[cols[c] for c in range(len(cols)) if c not in [cols.index("Part"),cols.index("Power_Consumption(Watts)")]]]
dat4                            = dat4.rename(columns={"RefrigeranT":"Part"})
dat4["SKU" ]                    = dat4["SKU"].astype(np.int8) + (2*len(dat0))
dat4["cost"]                    = np.round(dat4["cost"].astype(np.float64)*np.random.sample(size=dat4["cost"].to_numpy().shape),2)
dat7                            = dat1.copy()[[cols[c] for c in range(len(cols)) if c not in [cols.index("Power_Consumption(Watts)"),cols.index("RefrigeranT")]]]
dat7["SKU" ]                    = dat7["SKU"].astype(np.int8) + (3*len(dat0))
dat7["cost"]                    = np.round(dat7["cost"].astype(np.float64)*np.random.sample(size=dat7["cost"].to_numpy().shape),2)
dat7["Part"]                    = list(map(lambda x: x.replace("copper_condenser_coil","copper"),dat7["Part"]))
dat7["Part"]                    = list(map(lambda x: x.replace("aluminum_alloy_condenser_coil","aluminum"),dat7["Part"]))
dat8                            = dat1.copy()[[cols[c] for c in range(len(cols)) if c not in [cols.index("Power_Consumption(Watts)"),cols.index("RefrigeranT")]]]
dat8["SKU" ]                    = dat8["SKU"].astype(np.int8) + (4*len(dat0))
dat8["cost"]                    = np.round(dat8["cost"].astype(np.float64)*np.random.sample(size=dat8["cost"].to_numpy().shape),2)
dat8["Part"]                    = list(map(lambda x: x.replace("copper_condenser_coil","plastic"),dat8["Part"]))
dat8["Part"]                    = list(map(lambda x: x.replace("aluminum_alloy_condenser_coil","glass"),dat8["Part"]))
dat2                            = dat2.append([dat3,dat4,dat7,dat8],ignore_index=True)
dat2.to_csv("/mnt/data/csv/airconditioningcombined.csv",index=False)
