import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from tqdm import tqdm

def extract_features(dataloader, flatten=True):
    """
    Extracts features and labels from a PyTorch dataloader.
    If flatten is True, flattens the image tensors to 1D arrays per sample,
    suitable for simple machine learning models.
    """
    X_list = []
    y_list = []
    
    # We use tqdm if possible, if not just iterate
    for images, labels in dataloader:
        if flatten:
            # Flatten all dimensions except batch size
            images = images.view(images.size(0), -1)
        X_list.append(images.cpu().numpy())
        y_list.append(labels.cpu().numpy())
        
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    
    return X, y

def train_evaluate_knn(X_train, y_train, X_test, y_test, n_neighbors=5, **kwargs):
    """
    Trains a k-Nearest Neighbors classifier and evaluates it on test data.
    """
    print(f"Training kNN with n_neighbors={n_neighbors}...")
    model = KNeighborsClassifier(n_neighbors=n_neighbors, **kwargs)
    model.fit(X_train, y_train)
    
    print("Evaluating kNN...")
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"kNN Results -> Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    print("\nClassification Report (kNN):")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    return model, y_pred

def train_evaluate_random_forest(X_train, y_train, X_test, y_test, n_estimators=100, **kwargs):
    """
    Trains a Random Forest classifier and evaluates it on test data.
    """
    print(f"Training Random Forest with n_estimators={n_estimators}...")
    model = RandomForestClassifier(n_estimators=n_estimators, **kwargs)
    model.fit(X_train, y_train)
    
    print("Evaluating Random Forest...")
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"Random Forest Results -> Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    print("\nClassification Report (Random Forest):")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    return model, y_pred
