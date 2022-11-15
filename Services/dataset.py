import pandas as pd
import numpy  as np
import datetime

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

def dates(df):
    cdf  = df.copy()
    cols = list(cdf.columns)
    ret  = None
    for row in range(0,len(cdf)):
        dat  = datetime.datetime.strptime(cdf.iloc[row,cols.index("dt")],"%m/%d/%Y")
        while np.int8(dat.strftime("%d")) >= 2:
            cdf.iloc[row,cols.index("dt")] = (dat-pd.Timedelta(days=1)).strftime("%m/%d/%Y")
            ret                            = ret.append(pd.DataFrame(cdf.to_numpy()[row,:].reshape((1,len(cols))),columns=cols)) \
                                             if ret is not None                                                                  \
                                             else       pd.DataFrame(cdf.to_numpy()[row,:].reshape((1,len(cols))),columns=cols)
            dat                            = datetime.datetime.strptime(cdf.iloc[row,cols.index("dt")],"%m/%d/%Y")
    return ret

sku1                            = "ORIGIN_IP"
sku2                            = "DEST_IP"
clicks                          = "CLICKS"

dat0                            = pd.read_csv("/mnt/data/csv/airconditioning.csv")

dat1                            = dat0.copy()
dat1[sku1  ]                    = list(range(1,len(dat1)+1))
dat1[sku2  ]                    = list(range(256,len(dat1)+256))
dat5                            = pd.read_csv("/mnt/data/csv/condenser.csv")
dat6                            = dat5.merge(dat1,how="cross")
dat9                            = dates(dat6)
dat6                            = dat6.append(dat9,ignore_index=True)
dat1                            = dat6.rename(columns={"Condenser_Coil_x":"Part","cost":clicks})
dat1                            = dat1.drop(["TOn","Noise_level(db)","STAR","Ratings","Price","Image_url","Condenser_Coil_y"],axis=1)
dat1["Power_Consumption(Watts)"]= list(map(power,dat1["Power_Consumption(Watts)"].astype(np.float64)))
max_dat1_copper_SKU             = max(dat1[sku2])
vals                            = [c for c,v in enumerate(dat1["Part"]) if not v == "copper_condenser_coil"]
dat1[sku2  ][vals]              = dat1[sku2][vals] + max_dat1_copper_SKU
cols                            = list(dat1.columns)
dat2                            = dat1.copy()[[cols[c] for c in range(len(cols)) if c not in [cols.index("Power_Consumption(Watts)"),cols.index("RefrigeranT")]]]
dat2[clicks]                    = np.int64(dat2[clicks].astype(np.float64)*np.random.sample(size=dat2[clicks].to_numpy().shape))
dat3                            = dat1.copy()[[cols[c] for c in range(len(cols)) if c not in [cols.index("Part"),cols.index("RefrigeranT")]]]
dat3                            = dat3.rename(columns={"Power_Consumption(Watts)":"Part"})
dat3[sku2  ]                    = dat3[sku2] + (4*max_dat1_copper_SKU)
dat3[clicks]                    = np.int64(dat3[clicks].astype(np.float64)*np.random.sample(size=dat3[clicks].to_numpy().shape))
dat4                            = dat1.copy()[[cols[c] for c in range(len(cols)) if c not in [cols.index("Part"),cols.index("Power_Consumption(Watts)")]]]
dat4                            = dat4.rename(columns={"RefrigeranT":"Part"})
dat4[sku2  ]                    = dat4[sku2] + (8*max_dat1_copper_SKU)
dat4[clicks]                    = np.int64(dat4[clicks].astype(np.float64)*np.random.sample(size=dat4[clicks].to_numpy().shape))
dat7                            = dat1.copy()[[cols[c] for c in range(len(cols)) if c not in [cols.index("Power_Consumption(Watts)"),cols.index("RefrigeranT")]]]
dat7[sku2  ]                    = dat7[sku2] + (12*max_dat1_copper_SKU)
dat7[clicks]                    = np.int64(dat7[clicks].astype(np.float64)*np.random.sample(size=dat7[clicks].to_numpy().shape))
dat7["Part"]                    = list(map(lambda x: x.replace("copper_condenser_coil","copper"),dat7["Part"]))
dat7["Part"]                    = list(map(lambda x: x.replace("aluminum_alloy_condenser_coil","aluminum"),dat7["Part"]))
dat8                            = dat1.copy()[[cols[c] for c in range(len(cols)) if c not in [cols.index("Power_Consumption(Watts)"),cols.index("RefrigeranT")]]]
dat8[sku2  ]                    = dat8[sku2] + (16*max_dat1_copper_SKU)
dat8[clicks]                    = np.int64(dat8[clicks].astype(np.float64)*np.random.sample(size=dat8[clicks].to_numpy().shape))
dat8["Part"]                    = list(map(lambda x: x.replace("copper_condenser_coil","plastic"),dat8["Part"]))
dat8["Part"]                    = list(map(lambda x: x.replace("aluminum_alloy_condenser_coil","glass"),dat8["Part"]))
dat2                            = dat2.append([dat3,dat4,dat7,dat8],ignore_index=True)
dat2                            = dat2[[cols[c] for c in range(len(cols)) if c in [cols.index("dt"),cols.index(clicks),cols.index(sku1),cols.index(sku2)]]]
dat2["Fraud"]                   = list(map(lambda x: 1 if x > 0.5 else 0, np.random.sample(size=dat2[clicks].to_numpy().shape)))
dat2.to_csv("/mnt/data/csv/clicks.csv",index=False)
