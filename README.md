# AutoML Project

This project is designed to automate the machine learning pipeline, from data loading to model training. With a simple interface, it allows users to upload datasets, visualize key statistics, and automatically train machine learning models.

## Example Usage
![Example Usage](https://github.com/imfurkaann/autoML/blob/main/gif/video.gif)

## Features
- **Data Loading**: Upload datasets (e.g., CSV, Excel) easily.
- **Data Visualization**: Automatically generate insightful graphs, such as histograms, correlation matrices, and boxplots.
- **Statistical Analysis**: Get important statistical information about the dataset (mean, standard deviation, missing values, etc.).
- **Model Training**: Automatically split the data, select algorithms, and train models.
- **Evaluation**: Model performance is evaluated using various metrics such as accuracy, precision, recall, and F1-score.
- **Easy Setup**: Simple configuration for seamless data input and model execution.
- **MongoDB** : Preferred because it is fast as database. [Install MongoDB](https://www.mongodb.com/docs/manual/installation/)

## Requirements
- Python 3.xx

## Installation
Clone the repository: 
```bash
git clone https://github.com/imfurkaann/autoML.git
cd autoML
```

Install the required libraries using:
```bash
pip install -r requirements.txt
````

## Open The Web Interface
```bash
cd web
python run.py
```
## Without Using Web Interface
- Open app.py
- Write the required variables and then
- ```bash
  python app.py
  ```

 ## Contributing
 If you'd like to contribute to the project, feel free to fork the repository and submit pull requests. Your contributions are welcome!



