import os,pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,VotingClassifier,RandomForestRegressor
from sklearn.impute import KNNImputer,IterativeImputer
from sklearn.linear_model import LinearRegression,LassoCV
from sklearn.feature_selection import SelectFromModel,RFECV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import math,numpy as np
import seaborn as sns

class Dataset:
    def __init__(self,filename) -> None:
        self.filepath=filename
        self.loader=pd.read_csv(self.filepath)
        file_components=self.filepath.split(os.sep)[-2:]
        self.loader.describe().to_csv(f'{file_components[0]}_{file_components[1].replace(".csv","")}_describe.csv')

    def plot_column(self,column_names=["Packets A → B","Bytes A → B","Packets B → A","Bytes B → A","Rel Start","Duration"]):
        num_columns=len(column_names)
        num_rows=math.ceil(num_columns/2)

        fig,axs=plt.subplots(num_rows,2,figsize=(15,num_rows*5))
        fig.tight_layout(pad=3.0)

        for i,column in enumerate(column_names):
            if column in self.loader.columns:
                row=i//2
                col=i%2
                axs[row,col].plot(self.loader[column],color='red',linewidth=2,marker='s')
                axs[row, col].set_title(f"Distribution of {column}")
            else:
                print(f"Column {column} not found in the data")

        if num_columns%2!=0:
            axs[-1,-1].axis('off')
        
        fig.savefig("DistributionOfDatasetColumns.png",dpi=300)
        plt.show()
    
    def statistics(self):
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
        }

        for i,column in enumerate(["Packets A → B","Bytes A → B","Packets B → A","Bytes B → A","Rel Start","Duration"]):
            print(f'[INFO{i}]COL:{column}')
            print(f'Mean:{self.loader[column].mean()}')
            print(f'Median:{self.loader[column].median()}')
            print(f'Std:{self.loader[column].std()}')
            print(f'Min:{self.loader[column].min()}')
            print(f'Max:{self.loader[column].max()}')
            print(f'Variance:{self.loader[column].var()}')
            print(f'Skewness:{self.loader[column].skew()}')
            print(f'Kurtosis:{self.loader[column].kurt()}')
            print('\n\n')

            stats["Column"].append(column)
            stats["Mean"].append(self.loader[column].mean())
            stats["Median"].append(self.loader[column].median())
            stats["Std"].append(self.loader[column].std())
            stats["Min"].append(self.loader[column].min())
            stats["Max"].append(self.loader[column].max())
            stats["Variance"].append(self.loader[column].var())
            stats["Skewness"].append(self.loader[column].skew())
            stats["Kurtosis"].append(self.loader[column].kurt())
        pd.DataFrame.from_dict(stats).to_csv("TCP_Packets_PAOKPANS.csv")

    def correlation_analysis(self):
        corr=self.loader.corr()
        fig, ax = plt.subplots(figsize=(11, 9))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
        plt.title('Correlation Matrix')
        plt.show()
        fig.savefig("TCP_correlation_analysis.png",dpi=300)
    
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
    
if __name__=='__main__':
    analysis_tool=Dataset(os.path.join('','Conversations','TCP.csv'))
    # analysis_tool.plot_column()
    analysis_tool.statistics()
    analysis_tool.correlation_analysis()