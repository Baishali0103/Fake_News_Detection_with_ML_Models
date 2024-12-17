# Fake_News_Detection_with_ML_Models
This project is about Fake News Detection using a Random Forest ML model. 
# How to run
This can be run locally but I ran it on VSCode using virtual environment. 
1. First, run the .ipynb file to train the model. I suggest to run it on Google Collab or Jupyter Notebook (jupyter lab - download from anaconda)
2. Then, save the .ipynb files, .pkl files (which you will get after the training, these are the saved model and vectorizer), and app.py in a same folder.
3. Open the above folder on VSCode.
4. Create a **python virtual environment** on VSCode using Ctrl+Shift+P --> click python virtual environment --> choose whichever python version you have on your system.
5. Install all the libraries in the app.py files in this virtual environment using pip install command (streamlit, joblib, nltk, scikit-learn). Use the VSCode terminal you get after opening virtual environment to do this.
6. Then run the app.py file using **streamlit run app.py**

Note: 1. Make sure you add your dataset in the .ipynb file. 2. This can't be deployed via Github as the .pkl file of the model is bigger than 100MB. 
