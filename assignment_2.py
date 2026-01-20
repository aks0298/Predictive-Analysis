import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from imblearn.over_sampling import RandomOverSampler


dataset = pd.read_csv("C:/Users/Acer/Downloads/CreditCard_data.csv")

A = dataset.drop("Class", axis=1)
B = dataset["Class"]

balancer = RandomOverSampler(random_state=21)
A_res, B_res = balancer.fit_resample(A, B)

merged = pd.DataFrame(A_res, columns=A.columns)
merged["target"] = B_res


def rnd_sample(df, portion=0.6):
    return df.sample(frac=portion, random_state=21)

def sys_sample(df, gap=2):
    return df.iloc[::gap]

def strat_sample(df, portion=0.6):
    return df.groupby("target", group_keys=False).apply(
        lambda d: d.sample(frac=portion, random_state=21)
    )

def clust_sample(df, parts=4):
    temp = df.copy()
    temp["grp"] = temp.index % parts
    pick = np.random.choice(temp["grp"].unique())
    return temp[temp["grp"] == pick].drop("grp", axis=1)

def boot_sample(df):
    return df.sample(n=len(df), replace=True, random_state=21)


data_variants = {
    "RANDOM": rnd_sample(merged),
    "SYSTEMATIC": sys_sample(merged),
    "STRATIFIED": strat_sample(merged),
    "CLUSTER": clust_sample(merged),
    "BOOTSTRAP": boot_sample(merged)
}


algo_bank = {
    "LR": LogisticRegression(max_iter=1200),
    "DT": DecisionTreeClassifier(random_state=21),
    "RF": RandomForestClassifier(n_estimators=120, random_state=21),
    "NB": GaussianNB(),
    "SVM": SVC()
}


scores = []

for tag, block in data_variants.items():
    Xv = block.drop("target", axis=1)
    yv = block["target"]

    normalizer = StandardScaler()
    Xn = normalizer.fit_transform(Xv)

    Xtr, Xts, ytr, yts = train_test_split(
        Xn, yv, test_size=0.2, random_state=21, stratify=yv
    )

    for name, algo in algo_bank.items():
        algo.fit(Xtr, ytr)
        out = algo.predict(Xts)

        scores.append({
            "Algo": name,
            "Sampling": tag,
            "Score": accuracy_score(yts, out) * 100
        })


table = pd.DataFrame(scores)

final_view = (
    table
    .pivot(index="Algo", columns="Sampling", values="Score")
    .round(2)
)

print(final_view)
print("\nBest Sampling Per Model\n")
print(final_view.idxmax(axis=1))
