from PIL import Image, ImageFilter, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion, binary_dilation

# Caminhos das imagens
input_image_path = r'C:\Users\Administrador\Documents\imagem_mamografia_2.jpg'
output_image_path = r'C:\Users\Administrador\Documents\nodulos_detectados.jpg'

# Carregar a imagem em escala de cinza
img = Image.open(input_image_path).convert('L')
img_blur = img.filter(ImageFilter.GaussianBlur(2))  # Reduzir ruído com desfoque
img_array = np.array(img_blur)

# Detecção de bordas com Sobel
def sobel_edge_detection(image):
    dx = np.gradient(image.astype(float), axis=1)
    dy = np.gradient(image.astype(float), axis=0)
    return np.sqrt(dx**2 + dy**2)

edges = sobel_edge_detection(img_array)
edges = np.uint8(edges / np.max(edges) * 255)
edges[edges < 100] = 0
edges[edges >= 100] = 255

# Morfologia: erosão e dilatação
def dilatacao(binary_image):
    return binary_dilation(binary_image, structure=np.ones((3, 3)))

def erosao(binary_image):
    return binary_erosion(binary_image, structure=np.ones((3, 3)))

edges_erosion = erosao(edges)
edges_dilated = dilatacao(edges_erosion)

# Encontrar contornos
def find_contours(edges, min_area=50):
    contours = []
    visited = np.zeros_like(edges, dtype=bool)

    def bfs(x, y, contour):
        queue = [(x, y)]
        while queue:
            cx, cy = queue.pop(0)
            if cx < 0 or cx >= edges.shape[1] or cy < 0 or cy >= edges.shape[0] or visited[cy, cx] or edges[cy, cx] == 0:
                continue
            visited[cy, cx] = True
            contour.append((cx, cy))
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]:
                queue.append((cx + dx, cy + dy))

    for y in range(edges.shape[0]):
        for x in range(edges.shape[1]):
            if edges[y, x] == 255 and not visited[y, x]:
                contour = []
                bfs(x, y, contour)
                if len(contour) >= min_area:
                    contours.append(contour)
    return contours

contours = find_contours(edges_dilated)

# Sobrepor contornos na imagem original
overlay = img.copy()
draw = ImageDraw.Draw(overlay)
for contour in contours:
    draw.polygon(contour, outline="red")

# Visualizar as imagens em diferentes etapas
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(img, cmap='gray')
plt.title("Imagem Original")

plt.subplot(2, 3, 2)
plt.imshow(img_blur, cmap='gray')
plt.title("Após Desfoque")

plt.subplot(2, 3, 3)
plt.imshow(edges, cmap='gray')
plt.title("Bordas Detectadas")

plt.subplot(2, 3, 4)
plt.imshow(edges_erosion, cmap='gray')
plt.title("Após Erosão")

plt.subplot(2, 3, 5)
plt.imshow(edges_dilated, cmap='gray')
plt.title("Após Dilatação")

plt.subplot(2, 3, 6)
plt.imshow(overlay, cmap='gray')
plt.title("Contornos Detectados")

plt.tight_layout()
plt.show()

# Salvar imagem com contornos detectados
overlay.save(output_image_path)
print("Imagem com nódulos destacados salva em: " + output_image_path)
