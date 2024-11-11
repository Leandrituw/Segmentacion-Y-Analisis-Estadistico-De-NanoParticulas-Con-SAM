import cv2  # Importar la biblioteca OpenCV para procesamiento de imágenes
import numpy as np  # Importar NumPy para cálculos numéricos
import os  # Importar os para interactuar con el sistema de archivos
import time  # Importar time para calcular el tiempo de ejecución
import matplotlib.pyplot as plt  # Importar Matplotlib para graficar imágenes

# Función para calcular las propiedades de la máscara
def calculate_mask_properties(mask_image_path, pixels_per_nm):
    # Cargar la imagen de la máscara en escala de grises
    mask_image = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)

    # Asegurarse de que la máscara esté en formato binario (blanco y negro)
    _, binary_mask = cv2.threshold(mask_image, 127, 255, cv2.THRESH_BINARY)

    # Calcular el área de la máscara contando los píxeles blancos
    area_pixels = np.sum(binary_mask > 0)

    # Encontrar contornos en la máscara
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print(f"No Se Encontraron Contornos En La Máscara: {mask_image_path}.")
        return None, None, None

    largest_contour = max(contours, key=cv2.contourArea)
    max_diagonal = 0
    point1, point2 = None, None

    for i in range(len(largest_contour)):
        for j in range(i + 1, len(largest_contour)):
            distance = np.linalg.norm(largest_contour[i] - largest_contour[j])
            if distance > max_diagonal:
                max_diagonal = distance
                point1 = largest_contour[i][0]
                point2 = largest_contour[j][0]

    mid_point = (point1 + point2) / 2
    distances_to_mid = [np.linalg.norm(pt[0] - mid_point) for pt in largest_contour]
    half_diagonal_width = np.mean(distances_to_mid) if distances_to_mid else 0

    area_nm2 = area_pixels * (pixels_per_nm ** 2)
    max_diagonal_nm = max_diagonal * pixels_per_nm
    half_diagonal_width_nm = half_diagonal_width * pixels_per_nm

    return area_nm2, max_diagonal_nm, half_diagonal_width_nm

# Función para graficar la máscara y mostrar las propiedades calculadas
def plot_mask_with_properties(mask_image_path, area, max_diagonal, half_diagonal_width):
    mask_image = cv2.imread(mask_image_path)
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_image)
    plt.title(f'Área: {area:.2f} nm² - Diagonal Mayor: {max_diagonal:.2f} nm - Ancho a Mitad: {half_diagonal_width:.2f} nm')
    plt.axis('off')

    output_path = os.path.splitext(mask_image_path)[0] + '_properties.png'
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

# Función principal para procesar todas las máscaras y guardar propiedades en un archivo .txt
def process_masks_in_folder(folder_path, pixels_per_nm):
    areas, max_diagonals, half_diagonal_widths = [], [], []
    output_txt_path = os.path.join(folder_path, 'mascara_propiedades.txt')

    with open(output_txt_path, 'w') as file:
        for filename in os.listdir(folder_path):
            if filename.endswith('.png') or filename.endswith('.jpg'):
                mask_image_path = os.path.join(folder_path, filename)
                area, max_diagonal, half_diagonal_width = calculate_mask_properties(mask_image_path, pixels_per_nm)

                if area is not None:
                    plot_mask_with_properties(mask_image_path, area, max_diagonal, half_diagonal_width)
                    areas.append(area)
                    max_diagonals.append(max_diagonal)
                    half_diagonal_widths.append(half_diagonal_width)

                    # Guardar las propiedades en el archivo .txt
                    file.write(f"{filename} - Área: {area:.2f} nm², Diagonal Mayor: {max_diagonal:.2f} nm, Ancho a Mitad: {half_diagonal_width:.2f} nm\n")

    plot_histograms(areas, max_diagonals, half_diagonal_widths, len(areas))

def plot_histograms(areas, max_diagonals, half_diagonal_widths, num_masks):
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.hist(areas, bins=30, color='blue', alpha=0.7)
    plt.title('Histograma de Áreas')
    plt.xlabel('Área (nm²)')
    plt.ylabel('Frecuencia')

    plt.subplot(1, 3, 2)
    plt.hist(max_diagonals, bins=30, color='green', alpha=0.7)
    plt.title('Histograma de Diagonales Mayores')
    plt.xlabel('Diagonal Mayor (nm)')
    plt.ylabel('Frecuencia')

    plt.subplot(1, 3, 3)
    plt.hist(half_diagonal_widths, bins=30, color='red', alpha=0.7)
    plt.title('Histograma de Ancho a Mitad de Diagonal Mayor')
    plt.xlabel('Ancho a Mitad (nm)')
    plt.ylabel('Frecuencia')

    plt.suptitle(f'Cantidad De Partículas Analizadas: {num_masks}')
    plt.tight_layout()
    plt.show()

folder_path = input("Introduce La Ruta De La Carpeta Con Las Máscaras: ")
nm_per_pixel = float(input("Introduce La Relación De Nanómetros Por Píxel: "))

print("Analizando Partículas...")
tic = time.perf_counter()
process_masks_in_folder(folder_path, nm_per_pixel)
toc = time.perf_counter()
print(f"El Código Tomó {toc - tic:0.2f} Segundos En Ejecutarse.")
