
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, precision_recall_curve, auc
)
from architecture import Architecture
from trainer import Trainer

class Predictor:
    def __init__(self, arch, data_path='clean_data.pkl'):
        self.data_path = data_path
        self.X_val = None
        self.y_val = None
        self.arch = arch
        self.model = arch.model

    def load_data_and_preprocess(self):
        trainer = Trainer(data_path=self.data_path)
        X, y = trainer.load_data()
        _, X_val, _, y_val = trainer.process_data(X, y)
        self.X_val = X_val
        self.y_val = y_val

    def evaluate_model(self):
      with torch.no_grad():
          logits = self.model(torch.as_tensor(self.X_val).float()).numpy()
          probs = torch.sigmoid(torch.tensor(logits)).numpy()
          y_pred = (probs >= 0.3).astype(int)

      # Métricas básicas
      acc = accuracy_score(self.y_val, y_pred)
      precision = precision_score(self.y_val, y_pred)
      recall = recall_score(self.y_val, y_pred)
      f1 = f1_score(self.y_val, y_pred)
      roc_auc = roc_auc_score(self.y_val, probs)

      # Curvas
      fpr, tpr, _ = roc_curve(self.y_val, probs)
      prec, rec, _ = precision_recall_curve(self.y_val, probs)
      pr_auc = auc(rec, prec)  # Ordem corrigida!
      cm = confusion_matrix(self.y_val, y_pred)

      # Exibe resultados
      print(f"Acurácia: {acc:.4f}")
      print(f"Precisão: {precision:.4f}")
      print(f"Recall: {recall:.4f}")
      print(f"F1-score: {f1:.4f}")
      print(f"AUC-ROC: {roc_auc:.4f}")
      print(f"AUC-PR: {pr_auc:.4f}")
      print("Matriz de Confusão:")
      print(cm)

      # Gráficos
      plt.figure(figsize=(12, 5))

      plt.subplot(1, 2, 1)
      plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
      plt.plot([0, 1], [0, 1], 'k--')
      plt.xlabel('False Positive Rate')
      plt.ylabel('True Positive Rate')
      plt.title('ROC Curve')
      plt.legend()

      plt.subplot(1, 2, 2)
      plt.plot(rec, prec, label=f'AUC = {pr_auc:.2f}')
      plt.xlabel('Recall')
      plt.ylabel('Precision')
      plt.title('Precision-Recall Curve')
      plt.legend()

      plt.tight_layout()
      plt.show()

    def run(self):
        self.load_data_and_preprocess()
        self.evaluate_model()
