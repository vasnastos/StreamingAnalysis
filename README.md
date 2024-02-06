# StreamingAnalysis

- In order to run the data (and create the plots)
`python analysis.py` 


## Pipelines

- Traffic Volume Analysis:
    Time Series Analysis: Examine traffic volume over time to identify patterns, trends, peaks, and troughs.
    Volume by IP/Port: Analyze the data volume by source and destination IP, as well as by ports, to identify which hosts and services are the most active.
- Performance Analysis:
    Latency Analysis: If you have timestamp information, you can analyze the latency between packets.
    Packet Size Analysis: Understand the distribution of packet sizes to optimize network performance.


## Datasets
**External datasets**
- **UWF-ZeekData22**: This is a comprehensive network traffic dataset based on the MITRE ATT&CK Framework1. The dataset was collected using Zeek and labelled using the MITRE ATT&CK framework. It can be used to develop user profiles of groups intending to perform attacks. The dataset is publicly available at datasets.uwf.edu1.
- **Traffic analysis for 5G network slice:** This research paper discusses a home traffic analysis system combined with the Internet of Things2. Although it’s not a dataset, it might provide useful insights for your work.
- **Stanford Large Network Dataset Collection:** This collection includes several large network datasets, including social networks, networks with ground-truth communities, and communication networks3.

## Network traffic prediction pipeline
1. Data Collection
   - **Capture Data:** Use Wireshark to capture TCP packets on your network. Ensure you have sufficient data to capture various patterns and anomalies in network traffic.
2. Data Preprocessing
     - **Extract Features:** Convert the raw pcap (packet capture) files into a structured format like CSV or a database. Extract relevant features from each packet, such as source IP, destination IP, source port, destination port, timestamp, TCP flags, packet size, and any payloads if they're not encrypted.
     - **Data Cleaning:** Clean the data by removing duplicates, handling missing values, or filtering out irrelevant packets to focus on the traffic of interest.
     - **Feature Engineering:** Create new features that might be helpful for prediction, such as the flow duration, total bytes transferred in a flow, packet arrival times, and inter-arrival times.
3. Exploratory Data Analysis (EDA)
    - **Analyze the Data:** Perform statistical analysis and visualization to understand traffic patterns, identify outliers, and grasp the relationships between features.
    - **Correlation Analysis:** Determine which features are most relevant to network traffic behavior, which can help in reducing the feature space.
4. Model Selection
    - **Choose a Model:** Based on the problem at hand (e.g., traffic volume prediction, anomaly detection), select appropriate machine learning algorithms. Time series forecasting models (like ARIMA, SARIMA, LSTM networks) can be suitable for predicting future traffic volumes. For classification tasks (e.g., identifying types of traffic or detecting anomalies), algorithms like Random Forest, SVM, or neural networks might be more appropriate.

5. Feature Selection and Dimensionality Reduction
    - **Reduce Feature Space:** Use techniques like Principal Component Analysis (PCA) or feature importance scores from machine learning models to reduce the number of features, focusing on the most informative ones.

6. Model Training
   - **Train the Model:** Split the data into training and test sets. Train your model on the training set. Use cross-validation to fine-tune hyperparameters and avoid overfitting.

7. Evaluation
      - **Test the Model:** Evaluate the model's performance on the test set using appropriate metrics (e.g., accuracy, precision, recall, F1 score for classification; MSE, RMSE, MAE for regression).

10. Monitoring and Updating
    - **Monitor Performance:** Continuously monitor the model's performance and update it as necessary to adapt to new patterns in network traffic.

#### Tools and Technologies
 - Data Processing: Python (pandas, NumPy), SQL.
 - Machine Learning: scikit-learn, TensorFlow, Keras, PyTorch.
 - Visualization: Matplotlib, seaborn, Plotly.
