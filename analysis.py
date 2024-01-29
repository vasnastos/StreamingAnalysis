import os,pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
import matplotlib.pyplot as plt
import math,numpy as np
import seaborn as sns

class PacketAnalysis:
    def __init__(self,filename) -> None:
        self.filepath=filename
        self.loader=pd.read_csv(self.filepath)
        file_components=self.filepath.split(os.sep)[-2:]
        self.loader.describe().to_csv(f'{file_components[0]}_{file_components[1].replace(".csv","")}_describe.csv')

        print(f'[INFO]{self.loader.dtypes=}')

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
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
        plt.title('Correlation Matrix')
        plt.show()
        fig.savefig("TCP_correlation_analysis.png",dpi=300)
    
    def traffic_volume(self):
        pass
    
    def performance_analysis(self):
        pass

    def spark_exporter(self):
        pass
    
if __name__=='__main__':
    analysis_tool=PacketAnalysis(os.path.join('','Conversations','TCP.csv'))
    # analysis_tool.plot_column()
    analysis_tool.statistics()
    analysis_tool.correlation_analysis()