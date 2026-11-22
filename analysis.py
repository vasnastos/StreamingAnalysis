import os,re,sys,pandas as pd
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,VotingClassifier,RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (απαιτείται για το IterativeImputer)
from sklearn.impute import KNNImputer,IterativeImputer
from sklearn.linear_model import LinearRegression,LassoCV
from sklearn.feature_selection import SelectFromModel,RFECV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")  # τα διαγράμματα αποθηκεύονται ως png, δεν χρειάζεται οθόνη (π.χ. στον server)
import matplotlib.pyplot as plt
import matplotlib.ticker
import math,numpy as np
import seaborn as sns
from scipy.stats import ks_2samp,mannwhitneyu

# Στόχος της πρόβλεψης: τα συνολικά byte της ροής
TARGET="Bytes"
# Μετρούμενα μεγέθη κάθε ροής. Οι θύρες και το Stream ID είναι αναγνωριστικά (όχι μετρήσεις),
# γι' αυτό δεν μπαίνουν στις συσχετίσεις.
MEASURED=["Packets","Bytes","Packets A → B","Bytes A → B","Packets B → A","Bytes B → A",
          "Duration","Bits/s A → B","Bits/s B → A"]

# Ομάδες πρωτοκόλλων για την ανάλυση (το ICMPv6 είναι το ICMP του IPv6)
PROTOCOL_GROUPS={"TCP":["TCP"],"UDP":["UDP"],"ICMP":["ICMP","ICMPv6"]}
GROUP_COLORS={"TCP":"#2a78d6","UDP":"#eb6834","ICMP":"#1baf7a","Other":"#a3a29c"}
MIN_FLOWS=50  # κάτω από τόσες ροές μια ομάδα πρωτοκόλλων δεν αναλύεται (δεν βγάζει αξιόπιστα αποτελέσματα)

def fmt(v):
    """Αριθμός για τίτλους και εκτυπώσεις: χωρίς επιστημονική μορφή για μεγάλες τιμές."""
    if isinstance(v,(int,float,np.integer,np.floating)) and not isinstance(v,bool):
        return f"{v:,.0f}" if abs(v)>=1000 else f"{v:,.4g}"
    return str(v)

class Dataset:
    def __init__(self,filename,out_dir="Results",protocol="TCP",name=None,verbose=True,label=None) -> None:
        self.filepath=filename
        self.name=name or os.path.splitext(os.path.basename(filename))[0]  # π.χ. TCP_MAWI ή 20260609
        self.out_dir=out_dir
        self.verbose=verbose
        os.makedirs(self.out_dir,exist_ok=True)
        self.results={}
        self.loader=pd.read_csv(self.filepath)
        # Τα CSV του pcap_first_flows.py έχουν ροές όλων των πρωτοκόλλων (στήλη Protocol).
        # Η ανάλυση γίνεται στο πρωτόκολλο του LSTM (προεπιλογή TCP).
        self.all_flows=len(self.loader)
        self.protocol_share=np.nan
        if protocol and "Protocol" in self.loader.columns:
            members=[protocol] if isinstance(protocol,str) else list(protocol)
            self.loader=self.loader[self.loader["Protocol"].isin(members)]
            self.protocol_share=100*len(self.loader)/self.all_flows if self.all_flows else np.nan
            if name is None:
                self.name+=f"_{label or '_'.join(members)}"
        # Το Wireshark γράφει τις γραμμές ταξινομημένες κατά διεύθυνση IP, όχι κατά χρόνο.
        # Για χρονοσειρά και LSTM οι ροές μπαίνουν με σειρά έναρξης (Rel Start, και Stream ID όταν ισοβαθμούν).
        self.loader=self.loader.sort_values(["Rel Start","Stream ID"],kind="mergesort").reset_index(drop=True)
        self.loader.describe().to_csv(self.path("describe.csv"))

    def path(self,suffix):
        return os.path.join(self.out_dir,f"{self.name}_{suffix}")

    def save(self,fig,suffix):
        fig.savefig(self.path(suffix),dpi=300,bbox_inches="tight")
        plt.close(fig)
        if self.verbose:
            print(f"[SAVED] {self.path(suffix)}")

    @staticmethod
    def acf(x,max_lag):
        """Αυτοσυσχέτιση: πόσο μοιάζει η σειρά με τον εαυτό της μετατοπισμένη κατά k θέσεις."""
        x=np.asarray(x,dtype=float)
        x=x-x.mean()
        den=np.dot(x,x)
        max_lag=min(max_lag,len(x)-1)
        if den==0:  # σταθερή σειρά: η αυτοσυσχέτιση δεν ορίζεται
            return np.array([1.0]+[np.nan]*max_lag)
        return np.array([1.0]+[np.dot(x[:-k],x[k:])/den for k in range(1,max_lag+1)])

    def aggregate(self,bin_seconds=0.1):
        """Συγκεντρωτική χρονοσειρά: πλήθος ροών και byte των ροών που ξεκινούν σε κάθε διάστημα bin_seconds."""
        d=self.loader
        b=np.floor(d["Rel Start"]/bin_seconds).astype(int)
        g=d.groupby(b)[TARGET].agg(Flows="size",Bytes="sum")
        g=g.reindex(range(int(b.max())+1),fill_value=0)
        g.index=np.round(g.index*bin_seconds,6)
        g.index.name="Time (s)"
        return g

    def plot_column(self,column_names=["Packets","Bytes","Packets A → B","Bytes A → B","Packets B → A","Bytes B → A","Rel Start","Duration"]):
        """Πραγματική κατανομή κάθε στήλης (ιστόγραμμα).
        Πακέτα και byte σε λογαριθμική κλίμακα, γιατί λίγες πολύ μεγάλες ροές
        θα συμπίεζαν όλες τις υπόλοιπες κοντά στο μηδέν."""
        num_columns=len(column_names)
        num_rows=math.ceil(num_columns/2)

        fig,axs=plt.subplots(num_rows,2,figsize=(15,num_rows*4.5))
        fig.tight_layout(pad=4.0)

        for i,column in enumerate(column_names):
            ax=axs[i//2,i%2]
            if column not in self.loader.columns:
                print(f"Column {column} not found in the data")
                ax.axis('off')
                continue
            x=self.loader[column].dropna()
            if "Packets" in column or "Bytes" in column:
                ax.hist(np.log10(1+x),bins=60,color='red',alpha=.8)
                ax.set_xlabel(f"log10(1 + {column})")
                ax.set_yscale('log')
            elif column=="Rel Start":
                ax.hist(x,bins=60,color='red',alpha=.8)
                ax.set_xlabel("Rel Start (s)")
            else:
                ax.hist(x,bins=60,color='red',alpha=.8)
                ax.set_xlabel(f"{column} (s)")
                ax.set_yscale('log')
            ax.set_ylabel("Number of flows")
            ax.set_title(f"Distribution of {column} (median = {fmt(x.median())})")

        if num_columns%2!=0:
            axs[-1,-1].axis('off')

        self.save(fig,"DistributionOfDatasetColumns.png")

    def statistics(self,verbose=None):
        verbose=self.verbose if verbose is None else verbose
        stats={
            "Column":[],
            "Mean":[],
            "Median":[],
            "Std":[],
            "Min":[],
            "Max":[],
            "Variance":[],
            "Skewness":[],
            "Kurtosis":[],
            "P90":[],
            "P99":[],
        }

        for i,column in enumerate(["Packets","Bytes","Packets A → B","Bytes A → B","Packets B → A","Bytes B → A","Rel Start","Duration"]):
            col=self.loader[column]
            if verbose:
                print(f'[INFO{i}]COL:{column}')
                print(f'Mean:{col.mean()}')
                print(f'Median:{col.median()}')
                print(f'Std:{col.std()}')
                print(f'Min:{col.min()}')
                print(f'Max:{col.max()}')
                print(f'Variance:{col.var()}')
                print(f'Skewness:{col.skew()}')
                print(f'Kurtosis:{col.kurt()}')
                print('\n\n')

            stats["Column"].append(column)
            stats["Mean"].append(col.mean())
            stats["Median"].append(col.median())
            stats["Std"].append(col.std())
            stats["Min"].append(col.min())
            stats["Max"].append(col.max())
            stats["Variance"].append(col.var())
            stats["Skewness"].append(col.skew())
            stats["Kurtosis"].append(col.kurt())
            stats["P90"].append(col.quantile(.90))
            stats["P99"].append(col.quantile(.99))
        pd.DataFrame.from_dict(stats).to_csv(self.path("statistics.csv"))

    def correlation_analysis(self,columns=MEASURED):
        """Συσχέτιση Spearman μεταξύ των μεγεθών της ΙΔΙΑΣ ροής.
        Ο Spearman βασίζεται στη σειρά κατάταξης των τιμών, άρα δεν τον καθορίζουν οι λίγες τεράστιες ροές
        όπως τον Pearson. Δείχνει ποια μεγέθη δίνουν την ίδια πληροφορία (περιττές είσοδοι)."""
        with np.errstate(invalid="ignore",divide="ignore"):  # σταθερές στήλες (π.χ. όλες οι ροές 1 πακέτου) -> κενό
            corr=self.loader[columns].rank().corr()
        corr.to_csv(self.path("correlation.csv"))
        fig, ax = plt.subplots(figsize=(11, 9))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(corr, cmap=cmap, vmin=-1, vmax=1, center=0, fmt=".2f",
                    square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, ax=ax)
        ax.set_title('Correlation Matrix (Spearman, same flow)')
        self.save(fig,"correlation_analysis.png")

    def lag_correlation(self,columns=MEASURED,lags=(0,1,10,100)):
        """Αυτό που χρειάζεται η πρόβλεψη: συσχέτιση (Spearman) κάθε μεγέθους μιας ροής
        με τα byte της ροής που ξεκινά k θέσεις αργότερα. Το lag 0 είναι η ίδια ροή (όπως ο πίνακας συσχέτισης)·
        τα lag 1, 10, 100 δείχνουν πόση πληροφορία δίνουν οι προηγούμενες ροές για την επόμενη."""
        r=self.loader[columns].rank()
        y=r[TARGET]
        with np.errstate(invalid="ignore",divide="ignore"):
            out=pd.DataFrame({f"lag {k}":[r[c].corr(y.shift(-k)) for c in columns] for k in lags},index=columns)
        out.index.name=f"Feature of flow t vs {TARGET} of flow t+k"
        out.to_csv(self.path("lag_correlation.csv"))
        fig,ax=plt.subplots(figsize=(8,7))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(out, cmap=cmap, vmin=-1, vmax=1, center=0, annot=True, fmt=".2f", linewidths=.5, ax=ax)
        ax.set_title(f"Spearman: feature of flow t → {TARGET} of flow t+k")
        ax.set_ylabel("")
        self.save(fig,"lag_correlation.png")
        self.results["lag_correlation"]=out
        return out

    def time_series(self,bin_seconds=0.1,max_points=200000):
        """Η ακολουθία που βλέπει το LSTM: οι ροές με σειρά έναρξης.
        Πάνω: byte κάθε ροής. Κάτω: byte και πλήθος ροών που ξεκινούν ανά bin_seconds."""
        d=self.loader
        step=max(1,len(d)//max_points)
        s=d.iloc[::step]
        series=self.aggregate(bin_seconds)
        series.to_csv(self.path(f"per_{bin_seconds:g}s.csv"))

        fig,axs=plt.subplots(3,1,figsize=(15,13),sharex=True)
        axs[0].plot(s["Rel Start"],s[TARGET],linestyle='none',marker='.',markersize=2,color='red',rasterized=True)
        axs[0].set_yscale('log')
        axs[0].set_ylabel(f"{TARGET} per flow")
        sample=f" (every {step}th flow)" if step>1 else ""
        axs[0].set_title(f"{TARGET} of each flow in start-time order{sample}")
        axs[1].plot(series.index,series["Bytes"],color='red',linewidth=1)
        axs[1].set_ylabel(f"Bytes per {bin_seconds:g} s")
        axs[1].set_title(f"Bytes of flows starting in each {bin_seconds:g} s interval")
        axs[2].plot(series.index,series["Flows"],color='red',linewidth=1)
        axs[2].set_ylabel(f"Flows per {bin_seconds:g} s")
        axs[2].set_title(f"New flows in each {bin_seconds:g} s interval")
        axs[2].set_xlabel("Rel Start (s)")
        self.save(fig,"TimeSeries.png")

    def autocorrelation(self,max_lag=200,bin_seconds=0.1,window=100):
        """Υπάρχει χρονική εξάρτηση για να τη μάθει το LSTM;
        ACF του log(1+Bytes) με σειρά ροών, και ACF της συγκεντρωτικής σειράς byte ανά bin_seconds.
        Τιμές μέσα στη γκρι ζώνη (±1.96/√N) δεν διαφέρουν στατιστικά από το μηδέν."""
        flow=self.acf(np.log1p(self.loader[TARGET]),max_lag)
        series=self.aggregate(bin_seconds)
        if len(series)<50:
            # Πολύ μικρό χρονικό διάστημα (π.χ. 5.000 ροές ≈ 0,1 δευτ.): δεν υπάρχει συγκεντρωτική χρονοσειρά
            if self.verbose:
                print(f"[NOTE] Only {len(series)} intervals of {bin_seconds:g} s: the aggregated series is too short "
                      f"for autocorrelation; only the per-flow ACF is computed.")
            agg=np.array([1.0])
        else:
            agg=self.acf(np.log1p(series["Bytes"]),max_lag)
        out=pd.DataFrame({"ACF per flow log(1+Bytes)":pd.Series(flow),
                          f"ACF per {bin_seconds:g} s log(1+Bytes)":pd.Series(agg)})
        out.index.name="Lag"
        out.to_csv(self.path("autocorrelation.csv"))

        fig,axs=plt.subplots(2,1,figsize=(13,9))
        for ax,vals,n,unit in [(axs[0],flow,len(self.loader),"flows"),(axs[1],agg,len(series),f"intervals of {bin_seconds:g} s")]:
            if len(vals)<=1:
                ax.text(.5,.5,f"Not enough {unit} for an autocorrelation",ha="center",va="center",
                        transform=ax.transAxes,color="grey")
                ax.set_xticks([]); ax.set_yticks([])
                continue
            lags=np.arange(len(vals))
            ax.vlines(lags[1:],0,vals[1:],color='red')
            band=1.96/np.sqrt(n)
            ax.axhspan(-band,band,color='grey',alpha=.3)
            ax.axhline(0,color='black',linewidth=.8)
            ax.set_xlabel(f"Lag ({unit})")
            ax.set_ylabel("Autocorrelation")
        axs[0].axvline(window,color='blue',linestyle='--',label=f"LSTM window = {window} flows")
        axs[0].legend()
        axs[0].set_title("Autocorrelation of log(1 + Bytes), flows in start-time order")
        axs[1].set_title(f"Autocorrelation of log(1 + Bytes per {bin_seconds:g} s)")
        fig.tight_layout()
        self.save(fig,"autocorrelation.png")
        self.results["acf_flow"]=flow
        self.results["acf_agg"]=agg
        return out

    def baselines(self,window=100,test_size=0.2):
        """Τα απλά μοντέλα που πρέπει να ξεπεράσει το LSTM, στο ίδιο χρονολογικό 80/20 χωρίς ανακάτεμα."""
        y=self.loader[TARGET].to_numpy(dtype=float)
        n_test=int(len(y)*test_size)
        n_tr=len(y)-n_test
        window=min(window,max(1,n_tr//2))  # λίγες ροές (π.χ. ICMP): μικρότερο παράθυρο
        true=y[n_tr:]
        s=pd.Series(y)
        preds={
            "Previous flow":y[n_tr-1:-1],
            "Train median":np.full(n_test,np.median(y[:n_tr])),
            f"Mean of previous {window} flows":s.rolling(window).mean().shift(1).to_numpy()[n_tr:],
            f"Median of previous {window} flows":s.rolling(window).median().shift(1).to_numpy()[n_tr:],
        }
        rows=[]
        for name,p in preds.items():
            rows.append({"Model":name,
                         "MAE":np.mean(np.abs(true-p)),
                         "RMSE":np.sqrt(np.mean((true-p)**2))})
        out=pd.DataFrame(rows).set_index("Model")
        out.to_csv(self.path("baselines.csv"))
        if self.verbose:
            print(f"\nBaselines on the last {n_test:,} flows (train: first {n_tr:,} flows, up to Rel Start "
                  f"{self.loader['Rel Start'].iloc[n_tr-1]:.1f} s)")
            print(out.to_string(float_format=lambda v:f"{v:,.0f}"))
        self.results["baselines"]=out
        return out

    def summary(self,bin_seconds=0.1):
        """Τα βασικά ευρήματα σε έναν πίνακα, για το κείμενο της πτυχιακής."""
        d=self.loader
        b=d[TARGET].to_numpy(dtype=float)
        n=len(d)
        top=np.sort(b)[::-1]
        flow=self.results.get("acf_flow")
        agg=self.results.get("acf_agg")
        if flow is None:
            self.autocorrelation(bin_seconds=bin_seconds)
            flow,agg=self.results["acf_flow"],self.results["acf_agg"]
        base=self.results.get("baselines")
        if base is None:
            base=self.baselines()
        lagc=self.results.get("lag_correlation")
        if lagc is None:
            lagc=self.lag_correlation()
        best=base["MAE"].idxmin()
        span=d["Rel Start"].max()-d["Rel Start"].min()
        gaps=np.diff(d["Rel Start"].to_numpy(dtype=float))
        end=d["Rel Start"]+d["Duration"]
        facts={
            "Flows":n,
            "Share of all flows in the file (%)":self.protocol_share,
            "First to last flow start (s)":span,
            "New flows per second":n/span if span>0 else np.nan,
            # CV = τυπική απόκλιση / μέσος όρος των χρόνων ανάμεσα σε διαδοχικές ροές (1 = τυχαίες αφίξεις Poisson)
            "Inter-arrival time CV":gaps.std()/gaps.mean() if len(gaps)>1 and gaps.mean()>0 else np.nan,
            "Flows active until the last second of the file (%)":100*(end>=end.max()-1.0).mean(),
            "Flows with 1 packet (%)":100*(d["Packets"]==1).mean(),
            "One-direction flows (%)":100*((d["Packets A → B"]==0)|(d["Packets B → A"]==0)).mean(),
            "Bytes carried by top 1% of flows (%)":100*top[:max(1,n//100)].sum()/b.sum(),
            "Bytes carried by top 10% of flows (%)":100*top[:max(1,n//10)].sum()/b.sum(),
            "Median bytes per flow":np.median(b),
            "Mean bytes per flow":b.mean(),
            "Missing Bits/s values (%)":100*d["Bits/s A → B"].isna().mean(),
            "Spearman Packets~Bytes, same flow":lagc.loc["Packets","lag 0"],
            "Spearman Bytes(t)~Bytes(t+1)":lagc.loc[TARGET,"lag 1"],
            "ACF per flow, lag 1":flow[1],
            "ACF per flow, lag 10":flow[10] if len(flow)>10 else np.nan,
            "ACF per flow, lag 100":flow[100] if len(flow)>100 else np.nan,
            f"ACF per {bin_seconds:g} s, lag 1":agg[1] if len(agg)>1 else np.nan,
            f"ACF per {bin_seconds:g} s, lag 10":agg[10] if len(agg)>10 else np.nan,
            "Best baseline (MAE)":best,
            "Best baseline MAE":base.loc[best,"MAE"],
        }
        out=pd.Series(facts,name="Value")
        out.to_csv(self.path("summary.csv"))
        self.results["summary"]=out
        if self.verbose:
            print("\nSummary")
            for k,v in facts.items():
                print(f"  {k:45s} {fmt(v)}")
        return out

    def run_all(self):
        """Όλη η ανάλυση ενός αρχείου (μίας μέρας)."""
        self.plot_column()
        self.statistics()
        self.correlation_analysis()
        self.lag_correlation()
        self.time_series()
        self.autocorrelation()
        self.baselines()
        return self.summary()

    def missing_values(self,x_train,x_test,non_categorical_columns:list,method='mean'):
        if method=='mean':
            for column in non_categorical_columns:
                x_train[column].fillna(x_train[column].mean(),inplace=True)
                x_test[column].fillna(x_test[column].mean(),inplace=True)
        elif method=='median':
            for column in non_categorical_columns:
                x_train[column].fillna(x_train[column].median(),inplace=True)
                x_test[column].fillna(x_test[column].median(),inplace=True)
        elif method=='KNN':
            imputer=KNNImputer(n_neighbors=5)
            for column in x_train.columns.to_list():
                x_train[column]=imputer.fit_transform(x_train[column])
                x_test[column]=imputer.fit_transform(x_test[column])
        elif method=='forwardfill':
            x_train=x_train[non_categorical_columns]
            x_test=x_test[non_categorical_columns]
            x_train.fillna(method='ffill',inplace=True)
            x_test.fillna(method='ffill',inplace=True)
        elif method=='backwardfill':
            x_train=x_train[non_categorical_columns]
            x_test=x_test[non_categorical_columns]
            x_train.fillna(method='bfill',inplace=True)
            x_test.fillna(method='bfill',inplace=True)
        elif method=='iterativeImputer':
            x_train=x_train[non_categorical_columns]
            x_test=x_test[non_categorical_columns]
            imputer=IterativeImputer()
            x_train=pd.DataFrame(imputer.fit_transform(x_train),columns=x_train.columns.to_list())
            x_test=pd.DataFrame(imputer.fit_transform(x_test),columns=x_test.columns.to_list())
        elif method=="regressionImputer":
            imputer=IterativeImputer(estimator=LinearRegression(),random_state=1234,max_iter=20)
            x_train=pd.DataFrame(imputer.fit_transform(x_train),columns=x_train.columns.to_list())
            x_test=pd.DataFrame(imputer.fit_transform(x_test),columns=x_test.columns.to_list())
        return x_train,x_test

    def feature_selection(x_train:pd.DataFrame,y_train:pd.Series,x_test:pd.DataFrame,method='feature_importance'):
        # Features should be normalized
        if method=='feature_importance':
            model=RandomForestRegressor(max_depth=8,verbose=True,n_jobs=-1)
            model=model.fit(x_train,y_train)
            # importances=model.feature_importances_
            sfm=SelectFromModel(model,threshold='mean',prefit=True)
            x_train=pd.DataFrame(sfm.transform(x_train),columns=x_train.columns.tolist())
            x_test=pd.DataFrame(sfm.transform(x_test),columns=x_test.columns.tolist())
        elif method=='rfe':
            tscv=TimeSeriesSplit(n_splits=5)
            model=LinearRegression()
            selector=RFECV(model,cv=tscv,step=1)
            selector=selector.fit(x_train,y_train)
            selected_features=selector.support_
            x_train=x_train.loc[:,selected_features]
            x_test=x_test.loc[:,selected_features]
        elif method=='lasso':
            lasso=make_pipeline(StandardScaler(),LassoCV(cv=5,random_state=1234,max_iter=10000))
            lasso=lasso.fit(x_train,y_train)
            lasso_model=lasso.named_steps['lassocv']
            selected_features=lasso_model.coef_!=0
            x_train=x_train[selected_features]
            x_test=x_test[selected_features]
        elif method=='pca':
            pca_model=PCA(n_components=0.95) # 95% of variance in the columns
            x_train=pd.DataFrame(pca_model.fit_transform(x_train),columns=x_train.columns.tolist())
            x_test=pd.DataFrame(pca_model.fit_transform(x_test),columns=x_test.columns.tolist())
        return x_train,x_test

# ---------------------------------------------------------------------------
# Σύγκριση πολλών ημερών (π.χ. 9-15 Ιουνίου του MAWI)
# ---------------------------------------------------------------------------
WEEKDAY_COLOR="#2a78d6"   # καθημερινές
WEEKEND_COLOR="#eb6834"   # Σαββατοκύριακο
INK,INK2,GRID="#0b0b0b","#52514e","#e4e3df"


def day_of(path):
    """Η ημερομηνία από το όνομα του αρχείου ή του φακέλου, π.χ. first_5000_flows_20260609.csv -> 20260609."""
    m=re.search(r"(20\d{6})",os.path.basename(path)) or re.search(r"(20\d{6})",path)
    return m.group(1) if m else os.path.splitext(os.path.basename(path))[0]


def style(ax):
    for s in ("top","right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2)
    ax.grid(axis="y",color=GRID,linewidth=.8)
    ax.set_axisbelow(True)


class Week:
    """Η ίδια ανάλυση για κάθε μέρα και πίνακες/διαγράμματα που συγκρίνουν τις μέρες μεταξύ τους.
    Απαντά σε ερωτήματα που χρειάζονται πριν το LSTM:
      - Είναι οι μέρες ίδιες; (αν ναι, το LSTM μπορεί να εκπαιδευτεί σε κάποιες μέρες και να ελεγχθεί σε άλλες)
      - Διαφέρουν οι καθημερινές από το Σαββατοκύριακο;
      - Πόσο κυριαρχούν οι μεγάλες ροές; (καθορίζουν το σφάλμα του μοντέλου)
      - Πόσο καλά προβλέπουν τα απλά μοντέλα, με εκπαίδευση στην ίδια ή σε άλλες μέρες;"""

    def __init__(self,paths,out_dir="Results",protocol="TCP",label=None):
        self.out_dir=out_dir
        self.protocol=label or (protocol if isinstance(protocol,str) else "_".join(protocol))
        os.makedirs(out_dir,exist_ok=True)
        self.days=[]
        for p in sorted(paths,key=day_of):
            d=day_of(p)
            ds=Dataset(p,out_dir=os.path.join(out_dir,d),protocol=protocol,name=d,verbose=False)
            if len(ds.loader)<MIN_FLOWS:
                print(f"[SKIP] {d}: only {len(ds.loader)} {self.protocol} flows")
                continue
            ds.date=datetime.strptime(d,"%Y%m%d") if re.fullmatch(r"20\d{6}",d) else None
            ds.label=ds.date.strftime("%a %d/%m") if ds.date else d
            ds.weekend=bool(ds.date and ds.date.weekday()>=5)
            self.days.append(ds)

    def path(self,suffix):
        return os.path.join(self.out_dir,f"week_{suffix}")

    def save(self,fig,suffix):
        fig.savefig(self.path(suffix),dpi=300,bbox_inches="tight")
        plt.close(fig)
        print(f"[SAVED] {self.path(suffix)}")

    def color(self,ds):
        return WEEKEND_COLOR if ds.weekend else WEEKDAY_COLOR

    def line(self,ax,x,y,ds,shown,step=False):
        """Μία γραμμή ανά μέρα: καθημερινές μπλε (ένα στοιχείο στο υπόμνημα), Σάββατο/Κυριακή πορτοκαλί."""
        if ds.weekend:
            lab=ds.label
        else:
            lab="Weekdays" if "wd" not in shown else "_nolegend_"
            shown.add("wd")
        kw=dict(color=self.color(ds),linewidth=1.8 if ds.weekend else 1.1,
                linestyle="--" if (ds.weekend and ds.date and ds.date.weekday()==6) else "-",
                alpha=1 if ds.weekend else .7,label=lab)
        (ax.step(x,y,where="post",**kw) if step else ax.plot(x,y,**kw))

    def run(self):
        if len(self.days)<2:
            print(f"[SKIP] {self.protocol}: fewer than 2 days with at least {MIN_FLOWS} flows")
            return
        print(f"\n===== {self.protocol} =====")
        for ds in self.days:
            print(f"Day {ds.label}: {len(ds.loader):,} {self.protocol} flows of {ds.all_flows:,} -> {ds.out_dir}/")
            ds.run_all()
        self.summary_table()
        self.by_day()
        self.distributions()
        self.similarity()
        self.heavy_hitters()
        self.baselines()

    # 1. Πίνακας με τα βασικά νούμερα κάθε μέρας --------------------------------------------
    def summary_table(self):
        t=pd.concat([ds.results["summary"].rename(ds.label) for ds in self.days],axis=1)
        num=t.drop(index=["Best baseline (MAE)"],errors="ignore").apply(pd.to_numeric,errors="coerce")
        wd=[ds.label for ds in self.days if not ds.weekend]
        we=[ds.label for ds in self.days if ds.weekend]
        if wd:
            t["Weekday mean"]=num[wd].mean(axis=1)
        if we:
            t["Weekend mean"]=num[we].mean(axis=1)
        t.to_csv(self.path("summary.csv"))
        self.table=t
        print("\nWeek summary (one column per day)")
        print(t.map(fmt).to_string() if hasattr(t,"map") else t.applymap(fmt).to_string())
        return t

    # 2. Μικρά διαγράμματα: ένα μέγεθος ανά μέρα ------------------------------------------------
    def by_day(self):
        metrics=[("Share of all flows in the file (%)",f"{self.protocol} share of the flows (%)"),
                 ("New flows per second",f"New {self.protocol} flows per second"),
                 ("Median bytes per flow","Median bytes per flow"),
                 ("Flows with 1 packet (%)","Flows with 1 packet (%)"),
                 ("Bytes carried by top 1% of flows (%)","Bytes carried by the top 1% of flows (%)"),
                 ("ACF per flow, lag 1","Autocorrelation per flow, lag 1")]
        fig,axs=plt.subplots(2,3,figsize=(15,8))
        labels=[ds.label for ds in self.days]
        colors=[self.color(ds) for ds in self.days]
        for ax,(key,title) in zip(axs.flat,metrics):
            vals=[pd.to_numeric(ds.results["summary"].get(key),errors="coerce") for ds in self.days]
            vals=[0 if pd.isna(v) else float(v) for v in vals]
            ax.bar(labels,vals,color=colors,width=.62,edgecolor="white",linewidth=2)
            for x,v in enumerate(vals):
                ax.text(x,v,fmt(round(v,3) if abs(v)<10 else v),ha="center",va="bottom",fontsize=8,color=INK2)
            ax.set_title(title,loc="left",fontsize=10.5,color=INK,fontweight="bold")
            ax.tick_params(axis="x",labelrotation=45,labelsize=8)
            style(ax)
        handles=[plt.Rectangle((0,0),1,1,color=WEEKDAY_COLOR),plt.Rectangle((0,0),1,1,color=WEEKEND_COLOR)]
        fig.legend(handles,["Weekday","Weekend"],loc="upper right",ncol=2,frameon=False)
        fig.suptitle(f"{self.protocol} flows per day",x=.01,ha="left",fontsize=13,fontweight="bold")
        fig.tight_layout(rect=(0,0,1,.95))
        self.save(fig,"by_day.png")

    # 3. Κατανομή του στόχου ανά μέρα ---------------------------------------------------------
    def distributions(self):
        fig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,5.5))
        shown=set()
        for ds in self.days:
            x=np.sort(np.log10(ds.loader[TARGET].to_numpy(dtype=float)))
            y=np.arange(1,len(x)+1)/len(x)
            self.line(ax1,x,y,ds,shown,step=True)
        ax1.set_xlabel(f"log10({TARGET} per flow)")
        ax1.set_ylabel("Share of flows (ECDF)")
        ax1.set_title(f"Distribution of {TARGET} per flow, one line per day",loc="left",fontweight="bold")
        ax1.legend(frameon=False,fontsize=9,loc="lower right")
        style(ax1)
        data=[ds.loader[TARGET].to_numpy(dtype=float) for ds in self.days]
        bp=ax2.boxplot(data,whis=(1,99),showfliers=False,patch_artist=True,widths=.55,
                       medianprops=dict(color=INK,linewidth=1.5))
        for patch,ds in zip(bp["boxes"],self.days):
            patch.set_facecolor(self.color(ds)); patch.set_alpha(.85); patch.set_edgecolor("white")
        ax2.set_xticks(range(1,len(self.days)+1),[ds.label for ds in self.days],rotation=45,fontsize=8)
        ax2.set_yscale("log")
        ax2.set_ylabel(f"{TARGET} per flow (log scale)")
        ax2.set_title("Box plot per day (whiskers: 1st–99th percentile)",loc="left",fontweight="bold")
        style(ax2)
        fig.tight_layout()
        self.save(fig,"distributions.png")

    # 4. Πόσο μοιάζουν οι μέρες μεταξύ τους -------------------------------------------------------
    def similarity(self):
        """Απόσταση Kolmogorov-Smirnov (D) ανάμεσα στις κατανομές byte κάθε ζεύγους ημερών:
        0 = ίδια κατανομή, 1 = εντελώς διαφορετική. Μικρές τιμές σημαίνουν ότι ένα μοντέλο που εκπαιδεύεται
        σε κάποιες μέρες βλέπει στον έλεγχο δεδομένα παρόμοια με αυτά που έμαθε."""
        labels=[ds.label for ds in self.days]
        vals=[ds.loader[TARGET].to_numpy(dtype=float) for ds in self.days]
        n=len(self.days)
        D=np.zeros((n,n))
        for i in range(n):
            for j in range(i+1,n):
                D[i,j]=D[j,i]=ks_2samp(vals[i],vals[j]).statistic
        Dm=pd.DataFrame(D,index=labels,columns=labels)
        Dm.to_csv(self.path("similarity_ks.csv"))
        fig,ax=plt.subplots(figsize=(8,6.5))
        sns.heatmap(Dm,cmap="Blues",vmin=0,vmax=max(.3,D.max()),annot=True,fmt=".2f",square=True,
                    linewidths=1,linecolor="white",cbar_kws={"shrink":.7,"label":"KS distance D"},ax=ax)
        ax.set_title(f"How different is each pair of days? ({TARGET} per flow)",loc="left",fontweight="bold")
        self.save(fig,"similarity.png")

        # Καθημερινές και Σαββατοκύριακο
        wd=[i for i,ds in enumerate(self.days) if not ds.weekend]
        we=[i for i,ds in enumerate(self.days) if ds.weekend]
        if not wd or not we:
            return Dm
        a=np.concatenate([vals[i] for i in wd]); b=np.concatenate([vals[i] for i in we])
        ks=ks_2samp(a,b); mw=mannwhitneyu(a,b)
        within=[D[i,j] for i in wd for j in wd if i<j]
        cross=[D[i,j] for i in wd for j in we]
        rows={
            "Flows":(len(a),len(b)),
            "Median bytes per flow":(np.median(a),np.median(b)),
            "Mean bytes per flow":(a.mean(),b.mean()),
            "Flows with 1 packet (%)":(100*np.mean(np.concatenate([self.days[i].loader["Packets"].to_numpy()==1 for i in wd])),
                                       100*np.mean(np.concatenate([self.days[i].loader["Packets"].to_numpy()==1 for i in we]))),
        }
        t=pd.DataFrame(rows,index=["Weekdays","Weekend"]).T
        t.loc["KS distance D, weekdays vs weekend (pooled)"]=[ks.statistic,np.nan]
        t.loc["KS p-value"]=[ks.pvalue,np.nan]
        t.loc["Mann-Whitney p-value"]=[mw.pvalue,np.nan]
        t.loc["Mean KS D between two weekdays"]=[np.mean(within) if within else np.nan,np.nan]
        t.loc["Mean KS D weekday vs weekend day"]=[np.mean(cross),np.nan]
        t.to_csv(self.path("weekday_vs_weekend.csv"))
        print("\nWeekdays vs weekend")
        print(t.map(fmt).to_string() if hasattr(t,"map") else t.applymap(fmt).to_string())
        print("  (With thousands of flows even small differences give tiny p-values; compare the KS distances:\n"
              "   if weekday-vs-weekend D is close to weekday-vs-weekday D, the weekend is not different.)")
        return Dm

    # 5. Μεγάλες ροές ----------------------------------------------------------------------------
    def heavy_hitters(self,top_pairs=10):
        """Πόσες ροές μεταφέρουν τα περισσότερα byte, ποια ζεύγη διευθύνσεων, και πόσο από το σφάλμα
        του απλού μοντέλου οφείλεται σε αυτές."""
        fig,ax=plt.subplots(figsize=(10,6))
        rows=[]; pairs=[]; shown=set()
        for ds in self.days:
            d=ds.loader
            b=np.sort(d[TARGET].to_numpy(dtype=float))[::-1]
            n=len(b); cum=np.cumsum(b)/b.sum()
            x=100*np.arange(1,n+1)/n
            self.line(ax,x,100*cum,ds,shown)
            need=lambda q:int(np.searchsorted(cum,q)+1)
            # Μεγάλες ροές = το 1% με τα περισσότερα byte
            thr=np.quantile(d[TARGET],.99)
            big=d[d[TARGET]>=thr]; small=d[d[TARGET]<thr]
            # Σφάλμα του απλού μοντέλου «διάμεσος εκπαίδευσης» στο τελευταίο 20% της μέρας
            y=d[TARGET].to_numpy(dtype=float); n_test=int(n*.2); n_tr=n-n_test
            err=np.abs(y[n_tr:]-np.median(y[:n_tr]))
            big_test=y[n_tr:]>=thr
            rows.append({"Day":ds.label,"Flows":n,
                         "Flows carrying 50% of bytes":need(.5),
                         "Flows carrying 80% of bytes":need(.8),
                         "Flows carrying 90% of bytes":need(.9),
                         "Flows carrying 80% of bytes (%)":100*need(.8)/n,
                         "Bytes carried by top 1% (%)":100*b[:max(1,n//100)].sum()/b.sum(),
                         "Top-1% flows: median bytes":big[TARGET].median(),
                         "Top-1% flows: median packets":big["Packets"].median(),
                         "Top-1% flows: median duration (s)":big["Duration"].median(),
                         "Other flows: median duration (s)":small["Duration"].median(),
                         "Top-1% flows: one-direction (%)":100*((big["Packets A → B"]==0)|(big["Packets B → A"]==0)).mean(),
                         "Share of baseline absolute error from top-1% flows (%)":100*err[big_test].sum()/err.sum() if err.sum()>0 else np.nan})
            # Ζεύγη διευθύνσεων με τα περισσότερα byte (ανεξάρτητα από την κατεύθυνση)
            a_=d[["Address A","Address B"]].min(axis=1); b_=d[["Address A","Address B"]].max(axis=1)
            g=d.assign(_a=a_,_b=b_).groupby(["_a","_b"]).agg(Flows=(TARGET,"size"),Bytes=(TARGET,"sum"))
            g=g.sort_values("Bytes",ascending=False).head(top_pairs)
            g["Share of day's bytes (%)"]=100*g["Bytes"]/b.sum()
            g=g.reset_index().rename(columns={"_a":"Address 1","_b":"Address 2"})
            g.insert(0,"Rank",range(1,len(g)+1)); g.insert(0,"Day",ds.label)
            pairs.append(g)
        ax.set_xscale("log")
        ax.set_xlabel(f"Top % of {self.protocol} flows (largest first, log scale)")
        ax.set_ylabel("Share of bytes (%)")
        ax.axhline(80,color=GRID,linewidth=1)
        ax.set_title("How many of the largest flows carry the bytes? One line per day",loc="left",fontweight="bold")
        ax.legend(frameon=False,fontsize=9,loc="lower right")
        style(ax)
        self.save(fig,"heavy_hitters.png")
        t=pd.DataFrame(rows).set_index("Day").T
        t.to_csv(self.path("heavy_hitters.csv"))
        pd.concat(pairs).to_csv(self.path("top_pairs.csv"),index=False)
        print(f"[SAVED] {self.path('heavy_hitters.csv')}\n[SAVED] {self.path('top_pairs.csv')}")
        print("\nLarge flows per day")
        print(t.map(fmt).to_string() if hasattr(t,"map") else t.applymap(fmt).to_string())
        return t

    # 6. Απλά μοντέλα με εκπαίδευση στην ίδια ή σε άλλες μέρες -----------------------------------
    def baselines(self,window=100,test_size=0.2,test_days=2):
        """Όλα τα μοντέλα ελέγχονται στο ίδιο κομμάτι κάθε μέρας (το τελευταίο 20%), ώστε να συγκρίνονται.
        Η «διάμεσος» υπολογίζεται με τρεις τρόπους, όσοι και οι πιθανοί διαχωρισμοί για το LSTM:
          - από το πρώτο 80% της ίδιας μέρας            (80/20 μέσα σε κάθε μέρα)
          - από όλες τις άλλες μέρες                   (μία μέρα έλεγχος κάθε φορά)
          - από τις πρώτες μέρες (όλες εκτός των 2 τελευταίων), μόνο για τις 2 τελευταίες (εκπαίδευση/έλεγχος ανά μέρες)
        Αν τα σφάλματα είναι κοντά, οι μέρες είναι εναλλάξιμες και το LSTM μπορεί να εκπαιδευτεί σε άλλες μέρες."""
        ys=[ds.loader[TARGET].to_numpy(dtype=float) for ds in self.days]
        window=min(window,max(1,min(len(y)-int(len(y)*test_size) for y in ys)//2))
        train_days=list(range(max(1,len(ys)-test_days)))
        rows=[]
        for k,(ds,y) in enumerate(zip(self.days,ys)):
            n_test=int(len(y)*test_size); n_tr=len(y)-n_test
            true=y[n_tr:]; s=pd.Series(y)
            preds={
                "Previous flow":y[n_tr-1:-1],
                f"Mean of previous {window} flows":s.rolling(window).mean().shift(1).to_numpy()[n_tr:],
                f"Median of previous {window} flows":s.rolling(window).median().shift(1).to_numpy()[n_tr:],
                "Median: first 80% of the same day":np.full(n_test,np.median(y[:n_tr])),
                "Median: all other days":np.full(n_test,np.median(np.concatenate([ys[j] for j in range(len(ys)) if j!=k]))),
            }
            if k not in train_days:
                preds[f"Median: training days ({self.days[0].label}–{self.days[train_days[-1]].label})"]=\
                    np.full(n_test,np.median(np.concatenate([ys[j] for j in train_days])))
            for name,p in preds.items():
                rows.append({"Day":ds.label,"Model":name,"Test flows":n_test,
                             "MAE":np.mean(np.abs(true-p)),"RMSE":np.sqrt(np.mean((true-p)**2))})
        t=pd.DataFrame(rows)
        t.to_csv(self.path("baselines.csv"),index=False)
        mae=t.pivot(index="Model",columns="Day",values="MAE")[[ds.label for ds in self.days]]
        print(f"[SAVED] {self.path('baselines.csv')}")
        print("\nBaseline MAE (bytes) on the last 20% of each day")
        print(mae.map(fmt).to_string() if hasattr(mae,"map") else mae.applymap(fmt).to_string())
        return t


# ---------------------------------------------------------------------------
# Σύγκριση πρωτοκόλλων (TCP, UDP, ICMP) σε κάθε μέρα
# ---------------------------------------------------------------------------
def has_protocol_column(path):
    return "Protocol" in pd.read_csv(path,nrows=0).columns


def protocol_overview(paths,out_dir="Results",prefix="week"):
    """Για κάθε μέρα και κάθε ομάδα πρωτοκόλλων: μερίδιο σε ροές, πακέτα και byte, και πώς μοιάζουν
    οι ροές τους (μέγεθος, ροές ενός πακέτου, ροές μίας κατεύθυνσης, διάρκεια, IPv6).
    Δείχνει τι είδους κίνηση είναι το TCP, το UDP και το ICMP και πόσο σταθερά είναι αυτά από μέρα σε μέρα."""
    os.makedirs(out_dir,exist_ok=True)
    known=[m for ms in PROTOCOL_GROUPS.values() for m in ms]
    rows=[]
    for p in sorted(paths,key=day_of):
        d=pd.read_csv(p)
        day=day_of(p)
        date=datetime.strptime(day,"%Y%m%d") if re.fullmatch(r"20\d{6}",day) else None
        label=date.strftime("%a %d/%m") if date else day
        tot_f,tot_p,tot_b=len(d),d["Packets"].sum(),d[TARGET].sum()
        groups=list(PROTOCOL_GROUPS.items())+[("Other",None)]
        for g,members in groups:
            x=d[d["Protocol"].isin(members)] if members else d[~d["Protocol"].isin(known)]
            n=len(x)
            b=np.sort(x[TARGET].to_numpy(dtype=float))[::-1]
            rows.append({
                "Day":label,"Protocol":g,"Flows":n,
                "Share of flows (%)":100*n/tot_f if tot_f else np.nan,
                "Share of packets (%)":100*x["Packets"].sum()/tot_p if tot_p else np.nan,
                "Share of bytes (%)":100*x[TARGET].sum()/tot_b if tot_b else np.nan,
                "Median bytes per flow":x[TARGET].median() if n else np.nan,
                "Mean bytes per flow":x[TARGET].mean() if n else np.nan,
                "Mean packets per flow":x["Packets"].mean() if n else np.nan,
                "Flows with 1 packet (%)":100*(x["Packets"]==1).mean() if n else np.nan,
                "One-direction flows (%)":100*((x["Packets A → B"]==0)|(x["Packets B → A"]==0)).mean() if n else np.nan,
                "Median duration (s)":x["Duration"].median() if n else np.nan,
                "IPv6 flows (%)":100*(x["Network"]=="IPv6").mean() if n and "Network" in x else np.nan,
                "Bytes carried by top 1% of flows (%)":100*b[:max(1,n//100)].sum()/b.sum() if n and b.sum()>0 else np.nan,
                "Other protocols in group":", ".join(sorted(x["Protocol"].unique())) if g=="Other" and n else "",
            })
    t=pd.DataFrame(rows)
    t.to_csv(os.path.join(out_dir,f"{prefix}_protocols.csv"),index=False)
    print(f"\n[SAVED] {os.path.join(out_dir,f'{prefix}_protocols.csv')}")

    # Εκτύπωση: μέσος όρος των ημερών ανά πρωτόκολλο
    cols=["Share of flows (%)","Share of bytes (%)","Median bytes per flow","Flows with 1 packet (%)",
          "One-direction flows (%)","Median duration (s)","IPv6 flows (%)"]
    m=t.groupby("Protocol",sort=False)[cols].mean()
    print("\nProtocols (mean over the days)")
    print(m.map(fmt).to_string() if hasattr(m,"map") else m.applymap(fmt).to_string())

    # Διάγραμμα: ένα μέγεθος ανά πλαίσιο, μπάρες ανά μέρα και πρωτόκολλο
    panels=[("Share of flows (%)",False),("Share of bytes (%)",False),("Median bytes per flow",True),
            ("Flows with 1 packet (%)",False),("One-direction flows (%)",False),("IPv6 flows (%)",False)]
    days=list(dict.fromkeys(t["Day"]))
    groups=[g for g in list(PROTOCOL_GROUPS)+["Other"] if t.loc[t["Protocol"]==g,"Flows"].sum()>0]
    fig,axs=plt.subplots(2,3,figsize=(16,8.5))
    width=.8/len(groups)
    for ax,(col,logy) in zip(axs.flat,panels):
        for k,g in enumerate(groups):
            v=t[t["Protocol"]==g].set_index("Day").reindex(days)[col].to_numpy(dtype=float)
            ax.bar(np.arange(len(days))+(k-(len(groups)-1)/2)*width,np.nan_to_num(v),width=width*.92,
                   color=GROUP_COLORS[g],label=g,edgecolor="white",linewidth=.8)
        if logy:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v,_:f"{v:,.0f}"))
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xticks(range(len(days)),days,rotation=45 if len(days)>1 else 0,fontsize=8)
        ax.set_title(col,loc="left",fontsize=10.5,fontweight="bold",color=INK)
        style(ax)
    h,l=axs.flat[0].get_legend_handles_labels()
    fig.legend(h,l,loc="upper right",ncol=len(groups),frameon=False)
    fig.suptitle("TCP, UDP and ICMP flows" + (" per day" if len(days)>1 else f" ({days[0]})"),
                 x=.01,ha="left",fontsize=13,fontweight="bold")
    fig.tight_layout(rect=(0,0,1,.95))
    png=os.path.join(out_dir,f"{prefix}_protocols.png")
    fig.savefig(png,dpi=300,bbox_inches="tight"); plt.close(fig)
    print(f"[SAVED] {png}")
    return t


if __name__=='__main__':
    # Χρήση:
    #   python analysis.py                                     -> Conversations/TCP.csv
    #   python analysis.py first_5000_flows_20260609.csv       -> μία μέρα
    #   python analysis.py */first_5000_flows_*.csv            -> πολλές μέρες: ανάλυση κάθε μέρας + σύγκριση
    # Τα αποτελέσματα γράφονται στον φάκελο Results/. Αν το CSV έχει στήλη Protocol (αρχεία του
    # pcap_first_flows.py), γίνεται χωριστή ανάλυση για TCP, UDP και ICMP (Results/TCP, Results/UDP,
    # Results/ICMP) και μια σύγκριση των πρωτοκόλλων (Results/..._protocols).
    paths=sys.argv[1:] or [os.path.join('','Conversations','TCP.csv')]
    if not has_protocol_column(paths[0]):
        # Αρχεία μόνο με ροές TCP (π.χ. TCP_MAWI.csv): όπως πριν
        if len(paths)==1:
            Dataset(paths[0]).run_all()
        else:
            Week(paths,protocol=None,label="TCP").run()
    elif len(paths)==1:
        stem=os.path.splitext(os.path.basename(paths[0]))[0]
        for g,members in PROTOCOL_GROUPS.items():
            ds=Dataset(paths[0],out_dir=os.path.join("Results",g),protocol=members,label=g)
            if len(ds.loader)<MIN_FLOWS:
                print(f"[SKIP] {g}: only {len(ds.loader)} flows")
                continue
            print(f"\n===== {g}: {len(ds.loader):,} of {ds.all_flows:,} flows =====")
            ds.run_all()
        protocol_overview(paths,"Results",prefix=stem)
    else:
        # Για κάθε πρωτόκολλο: κάθε μέρα στον φάκελό της (Results/TCP/20260609/ ...) και η σύγκριση
        # των ημερών στο Results/TCP/week_* (το ίδιο για UDP και ICMP)
        for g,members in PROTOCOL_GROUPS.items():
            Week(paths,out_dir=os.path.join("Results",g),protocol=members,label=g).run()
        protocol_overview(paths,"Results",prefix="week")
