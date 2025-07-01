import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, jaccard_score
from google.colab.patches import cv2_imshow
from google.colab import files
from PIL import Image

# Upload das duas imagens
uploaded = files.upload()

# Carregamento das imagens
image_gt_path = list(uploaded.keys())[0]  # Ground truth
image_pred_path = list(uploaded.keys())[1]  # Resultado do algoritmo

# Leitura em escala de cinza
img_gt = cv2.imread(image_gt_path, cv2.IMREAD_GRAYSCALE)
img_pred = cv2.imread(image_pred_path, cv2.IMREAD_GRAYSCALE)

# Verificação de tamanho
if img_gt.shape != img_pred.shape:
    raise ValueError("As imagens precisam ter o mesmo tamanho!")

# Binarização (garante que os pixels são 0 ou 1)
_, img_gt_bin = cv2.threshold(img_gt, 127, 1, cv2.THRESH_BINARY)
_, img_pred_bin = cv2.threshold(img_pred, 127, 1, cv2.THRESH_BINARY)

# Flatten para métricas
y_true = img_gt_bin.flatten()
y_pred = img_pred_bin.flatten()

# Cálculo das métricas
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
iou = jaccard_score(y_true, y_pred)

# Impressão dos resultados
print(f" Avaliação do algoritmo de contorno:\n")
print(f" Precisão (Precision): {precision:.4f}")
print(f" Revocação (Recall): {recall:.4f}")
print(f" Acurácia: {accuracy:.4f}")
print(f" F1-Score: {f1:.4f}")
print(f" IoU (Jaccard): {iou:.4f}")
