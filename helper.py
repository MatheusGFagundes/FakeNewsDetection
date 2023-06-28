from datasets import load_dataset
from simpletransformers.classification import ClassificationModel
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import pandas as pd
import os   
import copy

def load_corpus_factbr():
    path = "corpura/factckbr/corpus.tsv"
    df = pd.read_csv(path, sep='\t', header=0)
    
    true = df[(df["alternativeName"] == "Verdadeiro") | (df["alternativeName"] == "verdadeiro")]["claimReviewed"]
    false = df[(df["alternativeName"] == "falso") | (df["alternativeName"] == "Falso")]["claimReviewed"]
    
    true_news = list(zip(true, [1] * len(true)))
    fake_news = list(zip(false, [0] * len(false)))
    
    all_news = true_news + fake_news
    
    return pd.DataFrame(all_news)

def load_corpus_fakeBR():
    base_path = "corpura/fakeBR/size_normalized_texts/"
    
    fake_news = []
    true_news = []
    for root, dirs, files in os.walk(base_path + "fake"):
        for file in files:
            if file.endswith('.txt'):
                with open(os.path.join(root, file), 'r') as f:
                    text = f.read()
                    fake_news.append([text, 0])

    for root, dirs, files in os.walk(base_path + "true"):
        for file in files:
            if file.endswith('.txt'):
                with open(os.path.join(root, file), 'r') as f:
                    text = f.read()
                    true_news.append([text, 1])                


    all_news = fake_news + true_news
    return pd.DataFrame(all_news)

def evaluate_model(model, test_data):
    result, _, _  = model.eval_model(test_data)
    return {"tp":result["tp"], "tn":result["tn"], "fp":result["fp"], "fn":result["fn"]}  


def run_model(model, train_data, test_data):
    model.train_model(train_data, args={"overwrite_output_dir":True}, manual_seed = 0)
    return evaluate_model(model, test_data)


def run_model_with_cv(model, all_data, k_fold = 5):
    kf = KFold(n_splits=k_fold, random_state=2, shuffle=True)
    k_confusion_metrix = []
    
    for train_index, val_index in kf.split(all_data):
        model_i = copy.deepcopy(model)
        train_df = all_data.iloc[train_index]
        val_df = all_data.iloc[val_index]
        result =  run_model(model_i, train_df, val_df)   
        confusion_metrix.append(result)

    return k_confusion_metrix
