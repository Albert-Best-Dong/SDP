# -*- coding: utf-8 -*-
"""
Corrected analysis for: Boruta feature selection + tree-based CPDP.
Inputs (all four provided files):
  repository_summary.csv  -> Table3 (features per dataset), Table4 (stability)
  overall_summary.csv     -> 5.2.3 frequent metrics
  final_results.csv       -> 5.2.4 no-FS performance
  final_results_FS.csv    -> 5.2.4 Boruta-FS performance
Outputs: 6 csv tables + 4 png figures
"""
import csv, os, statistics, collections
import numpy as np
import matplotlib
import pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt

A = os.path.join(os.path.dirname(os.path.abspath(__file__)), "")
def read(fn):
    with open(fn, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

repo = read("input/repository_summary.csv")
overall = read("input/overall_summary.csv")
res = read("input/final_results.csv")
res_fs = read("input/final_results_FS.csv")

models = ["RandomForest", "XGBoost", "LightGBM", "CatBoost"]
metrics = ["Accuracy","Precision","Recall","F1","ROC-AUC","MCC","Balanced Accuracy","Specificity"]

# ---------- Table 3: features selected per (dataset, model), stats per dataset ----------
combo_feats = collections.defaultdict(set)     # (Dataset,Model) -> features
for r in repo:
    combo_feats[(r["Dataset"], r["Model"])].add(r["Feature"])

datasets = sorted({k[0] for k in combo_feats})
rows3 = []
for ds in datasets:
    counts = [len(combo_feats[(ds, m)]) for m in models if (ds, m) in combo_feats]
    rows3.append({"Dataset": ds, "Mean": round(statistics.mean(counts),2),
                  "Std": round(statistics.stdev(counts),2) if len(counts)>1 else 0.0,
                  "Min": min(counts), "Max": max(counts), "Runs/combos": len(counts)})
with open("RQ2_results/table3_features_per_dataset.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=["Dataset","Mean","Std","Min","Max","Runs/combos"]); w.writeheader(); w.writerows(rows3)
print("Table3 (features selected per dataset):")
for x in rows3: print(x)

# ---------- Table 4: stability ----------
# 1) always-selected across 10 runs (Selected_Rate == 100)
always = sum(1 for r in repo if float(r["Selected_Rate"])>=100)
total  = len(repo)
# 2) within-combo stable core (avg number of always-selected features per combo)
core=[]
for k,v in combo_feats.items():
    freqs=[r for r in repo if (r["Dataset"],r["Model"])==k]
    core.append(sum(1 for r in freqs if float(r["Selected_Rate"])>=100))
# 3) cross-model Jaccard within dataset
def jac(a,b): return len(a&b)/len(a|b) if (a|b) else 0.0
jacs=[]
for ds in datasets:
    ss={m:combo_feats[(ds,m)] for m in models if (ds,m) in combo_feats}
    prs=[(ss[models[i]],ss[models[j]]) for i in range(4) for j in range(i+1,4) if models[i] in ss and models[j] in ss]
    jacs += [jac(a,b) for a,b in prs]
# 4) cross-dataset sharing
ds_of = collections.defaultdict(set)
for r in repo: ds_of[r["Feature"]].add(r["Dataset"])
n_sh = {n: sum(1 for v in ds_of.values() if len(v)>=n) for n in [1,5,10,20]}
rows4 = [
    ("Always-selected ratio (100% across 10 runs)", f"{always/total*100:.1f}%"),
    ("Avg stable core features per (dataset,model)", f"{statistics.mean(core):.2f}"),
    ("Mean cross-model Jaccard (same dataset)", f"{statistics.mean(jacs):.2f}"),
    ("Features in >=5 datasets", str(n_sh[5])),
    ("Features in >=10 datasets", str(n_sh[10])),
    ("Features in all 20 datasets", str(n_sh[20])),
]
with open("RQ2_results/table4_stability.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["Metric","Value"]); w.writerows(rows4)
print("\nTable4 (stability):"); [print(x) for x in rows4]

# ---------- 5.2.3 frequent metrics ----------
ov = sorted(overall, key=lambda x: int(x["Frequency"]), reverse=True)
with open("RQ2_results/frequent_metrics_top15.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["Feature","Frequency","Share%","Rank"])
    for r in ov[:15]: w.writerow([r["Feature"], int(r["Frequency"]), round(int(r["Frequency"])/800*100,1), int(r["Rank"])])

# ---------- 5.2.4 performance impact ----------
def mean(v): return float(v.split("+")[0])
def load(fn):
    d={}
    for r in fn: d[(r["Dataset"],r["Model"])] = {mm: mean(r[mm]) for mm in metrics}
    return d
base=load(res); fs=load(res_fs)
rows_overall=[]
for mm in metrics:
    b=[base[k][mm] for k in base]; f=[fs[k][mm] for k in fs]
    d=[f[i]-b[i] for i in range(len(b))]
    rows_overall.append([mm, round(statistics.mean(b),4), round(statistics.mean(f),4),
                         round(statistics.mean(d),4), sum(1 for x in d if x>0.0001), sum(1 for x in d if x<-0.0001)])
with open("RQ2_results/perf_overall_FS_vs_noFS.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["Metric","No-FS mean","FS mean","Delta","Improved/80","Degraded/80"]); w.writerows(rows_overall)
print("\nTable (5.2.4 overall):")
for x in rows_overall: print(x)

rows_perm=[]
for mod in models:
    dss=sorted({k[0] for k in base})
    for mm in metrics:
        b=[base[(ds,mod)][mm] for ds in dss]; f=[fs[(ds,mod)][mm] for ds in dss]
        d=[f[i]-b[i] for i in range(len(b))]
        rows_perm.append([mod,mm,round(statistics.mean(b),4),round(statistics.mean(f),4),
                          round(statistics.mean(d),4),sum(1 for x in d if x>0.0001),sum(1 for x in d if x<-0.0001)])
with open("RQ2_results/perf_per_model_FS_vs_noFS.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["Model","Metric","No-FS mean","FS mean","Delta","Improved/20","Degraded/20"]); w.writerows(rows_perm)

print("\nFiles written: table3_features_per_dataset.csv, table4_stability.csv, frequent_metrics_top15.csv, perf_overall_FS_vs_noFS.csv, perf_per_model_FS_vs_noFS.csv")

# ==================== FIGURES ====================
plt.rcParams.update({"font.family":"Arial","figure.dpi":300,"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
plt.rcParams["axes.unicode_minus"]=False

# Fig1: features per dataset
# ds=[x["Dataset"] for x in rows3]; val=[x["Mean"] for x in rows3]
# o=np.argsort(-np.array(val))
# fig,ax=plt.subplots(figsize=(9,4.5))
# ax.bar([ds[i] for i in o],[val[i] for i in o],color="#4472C4")
# ax.set_ylabel("Mean # features selected (Boruta)"); ax.set_xlabel("Dataset")
# plt.xticks(rotation=45,ha="right")
# for i,v in enumerate(val): ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
# plt.tight_layout(); plt.savefig("fig_features_per_dataset.png",bbox_inches="tight"); plt.close()

plt.rcParams["font.family"]="Arial"; plt.rcParams["axes.unicode_minus"]=False
repo = pd.read_csv("input/repository_summary.csv")
feat = repo.groupby(["Dataset","Model"]).size().reset_index(name="n_selected")
ds = feat.groupby("Dataset")["n_selected"].mean().reset_index().sort_values("n_selected", ascending=False)
fig, ax = plt.subplots(figsize=(12,6), dpi=300)
bars = ax.bar(ds["Dataset"], ds["n_selected"], color="#4472C4")
for b in bars:
    h=b.get_height(); ax.text(b.get_x()+b.get_width()/2, h+0.25, f"{int(h)}", ha="center", va="bottom", fontsize=9)
ax.set_title("Number of Boruta-selected features per dataset", fontsize=14, pad=15)
ax.set_ylabel("Count of selected features", fontsize=11)
ax.set_ylim(0,25); ax.set_yticks([0,5,10,15,20,25])
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.xticks(rotation=45, ha="right", fontsize=9); plt.tight_layout()
plt.savefig("RQ2_results/fig_features_per_dataset.png", dpi=300, bbox_inches="tight")

# Fig2: stability - cross-dataset sharing
fig,ax=plt.subplots(figsize=(6,4))
lbl=[">=1",">=5",">=10","all 20"]; vv=[len(ds_of),n_sh[5],n_sh[10],n_sh[20]]
ax.bar(lbl,vv,color="#ED7D31")
for i,v in enumerate(vv): ax.text(i,v,str(v),ha="center",va="bottom",fontsize=9)
ax.set_ylabel("Number of features"); ax.set_xlabel("Number of datasets a feature appears in")
plt.tight_layout(); plt.savefig("RQ2_results/fig_cross_dataset_sharing.png",bbox_inches="tight"); plt.close()

# Fig3: frequent metrics
top=ov[:12][::-1]
fig,ax=plt.subplots(figsize=(7,4.5))
ax.barh([r["Feature"] for r in top],[int(r["Frequency"]) for r in top],color="#70AD47")
ax.set_xlabel("Selection frequency (of 800 runs)")
for r in top: ax.text(int(r["Frequency"]), top.index(r)+1, str(int(r["Frequency"])), va="center", fontsize=8)
plt.tight_layout(); plt.savefig("RQ2_results/fig_frequent_metrics.png",bbox_inches="tight"); plt.close()

# Fig4: performance delta overall per metric
mm=[x[0] for x in rows_overall]; dv=[x[3] for x in rows_overall]
fig,ax=plt.subplots(figsize=(8,4))
cols=["#C00000" if v<0 else "#2E75B6" for v in dv]
ax.bar(mm,dv,color=cols)
ax.axhline(0,color="k",linewidth=0.8)
ax.set_ylabel("Delta (FS - No-FS)"); ax.set_title("Overall performance change after Boruta FS")
plt.xticks(rotation=45,ha="right")
for i,v in enumerate(dv): ax.text(i,v,f"{v:.3f}",ha="center",va="bottom" if v>=0 else "top",fontsize=8)
plt.tight_layout(); plt.savefig("RQ2_results/fig_perf_delta_overall.png",bbox_inches="tight"); plt.close()

print("\nFigures written: fig_features_per_dataset.png, fig_cross_dataset_sharing.png, fig_frequent_metrics.png, fig_perf_delta_overall.png")
