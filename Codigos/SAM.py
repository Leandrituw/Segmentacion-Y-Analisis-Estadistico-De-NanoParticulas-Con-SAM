import numpy as np # Importa la biblioteca NumPy para operaciones con arreglos.
import torch # Importa PyTorch para el manejo de tensores y modelos de aprendizaje profundo.
import matplotlib.pyplot as plt # Importa Matplotlib para visualización de gráficos.
import cv2 # Importa OpenCV para procesamiento de imágenes.
import os # Importa la biblioteca OS para operaciones del sistema operativo.
import time # Importa la biblioteca time para medir el tiempo de ejecución.
from PyQt5 import QtWidgets # Importa QtWidgets para crear la interfaz gráfica.
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton # Importa componentes específicos de PyQt5.
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas # Importa el lienzo para gráficos de Matplotlib en PyQt5.
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator # Importa funciones y clases para el modelo SAM.

# Variable global para almacenar la imagen
image = None

# Función para mostrar la imagen con las máscaras superpuestas en un gráfico lineal
def show_colored_masks(image, anns):
    if len(anns) == 0: # Si no hay máscaras generadas, no hace nada.
        return
    
    overlay_image = image.copy() # Crea una copia de la imagen original para superponer las máscaras.
    for ann in anns: # Itera sobre cada máscara generada.
        mask = ann['segmentation'] # Obtiene la máscara de la anotación.
        color = np.random.randint(0, 255, size=(3,), dtype=np.uint8) # Genera un color aleatorio para la máscara.
        overlay_image[mask] = color # Aplica el color a la región de la imagen correspondiente a la máscara.
    
    plt.figure(figsize=(10, 10)) # Crea una figura de Matplotlib con un tamaño específico.
    plt.imshow(overlay_image) # Muestra la imagen con las máscaras superpuestas.
    plt.title('Máscaras Superpuestas\nCantidad De Máscaras Generadas: {}'.format(len(anns))) # Título del gráfico.
    plt.axis('off') # Desactiva los ejes del gráfico.
    plt.show() # Muestra el gráfico.

# Clase para la ventana interactiva de PyQt5 con actualización de gráfico y botón "Finalizar"
class MaskDiscardWindow(QMainWindow):  # Define la clase MaskDiscardWindow que hereda de QMainWindow.
    def __init__(self, image, masks, mask_filenames, mask_folder):  # Constructor que recibe la imagen, máscaras, nombres de archivos de máscaras y carpeta de máscaras.
        super().__init__()  # Llama al constructor de la clase base QMainWindow.
        self.setWindowTitle("Descartar Máscaras")  # Establece el título de la ventana.
        self.setGeometry(100, 100, 500, 500)  # Define la posición y tamaño de la ventana (x, y, ancho, alto).

        central_widget = QWidget(self)  # Crea un widget central para la ventana.
        self.setCentralWidget(central_widget)  # Establece el widget central en la ventana.
        layout = QVBoxLayout(central_widget)  # Crea un diseño vertical para el widget central.

        self.canvas = FigureCanvas(plt.figure())  # Crea un lienzo para mostrar imágenes utilizando Matplotlib.
        layout.addWidget(self.canvas)  # Añade el lienzo al diseño vertical.

        self.finish_button = QPushButton("Finalizar")  # Crea un botón para finalizar la operación.
        self.finish_button.clicked.connect(self.close_window)  # Conecta el evento de clic del botón a la función close_window.
        layout.addWidget(self.finish_button)  # Añade el botón al diseño.

        self.image = image  # Almacena la imagen proporcionada.
        self.masks = masks  # Almacena las máscaras proporcionadas.
        self.mask_filenames = mask_filenames  # Almacena los nombres de archivo de las máscaras.
        self.mask_folder = mask_folder  # Almacena la carpeta donde se guardan las máscaras.

        self.update_display()  # Llama a la función update_display para mostrar las máscaras en el lienzo.

        self.canvas.mpl_connect("button_press_event", self.on_click)  # Conecta el evento de clic en el lienzo a la función on_click.

    def update_display(self):  # Método para actualizar la visualización de las máscaras.
        self.canvas.figure.clf()  # Limpia la figura actual en el lienzo.
        ax = self.canvas.figure.add_subplot(111)  # Crea un nuevo subplot en la figura.

        overlay_image = self.image.copy()  # Crea una copia de la imagen original para superponer las máscaras.
        for ann in self.masks:  # Itera sobre las máscaras.
            mask = ann['segmentation']  # Obtiene la segmentación de la máscara actual.
            color = np.random.randint(0, 255, size=(3,), dtype=np.uint8)  # Genera un color aleatorio para la máscara.
            overlay_image[mask] = color  # Superpone la máscara de color sobre la imagen.

        ax.imshow(overlay_image)  # Muestra la imagen con las máscaras superpuestas.
        ax.set_title('Máscaras Superpuestas\nCantidad De Máscaras Restantes: {}'.format(len(self.masks)))  # Establece el título del gráfico con la cantidad de máscaras restantes.
        ax.axis('off')  # Desactiva los ejes del gráfico.
        self.canvas.draw()  # Dibuja el lienzo para mostrar la actualización.

    def on_click(self, event):  # Método que maneja el evento de clic en el lienzo.
        if event.inaxes:  # Verifica si el clic fue dentro de los ejes del gráfico.
            x, y = int(event.xdata), int(event.ydata)  # Obtiene las coordenadas x e y del clic.
            for i, mask in enumerate(self.masks):  # Itera sobre las máscaras.
                if np.any(mask['segmentation'][y, x]):  # Verifica si la máscara cubre las coordenadas clicadas.
                    print(f"Máscara {self.mask_filenames[i]} eliminada.")  # Imprime un mensaje indicando que se eliminará la máscara.
                    self.masks.pop(i)  # Elimina la máscara de la lista de máscaras.
                    os.remove(os.path.join(self.mask_folder, self.mask_filenames[i]))  # Elimina el archivo de la máscara de la carpeta.
                    self.mask_filenames.pop(i)  # Elimina el nombre del archivo de la lista de nombres.
                    break  # Sale del bucle una vez que se elimina una máscara.
            self.update_display()  # Actualiza la visualización de las máscaras después de eliminar una.

    def close_window(self):  # Método que se llama para cerrar la ventana.
        print("Ejecución Finalizada.")  # Imprime un mensaje indicando que la ejecución ha terminado.
        self.clean_up()  # Llama al método clean_up para limpiar la ventana.
        self.close()  # Cierra la ventana.

    def clean_up(self):  # Método para limpiar la ventana.
        """Limpia la ventana y elimina el gráfico para evitar errores de actualización."""  # Comentario sobre la función de limpieza.
        self.canvas.close()  # Cierra el lienzo para prevenir errores de objeto eliminado.
        plt.close(self.canvas.figure)  # Cierra la figura de Matplotlib.

# Función principal para ejecutar la segmentación y mostrar las máscaras
def run_segmentation(image_path=None, sam_checkpoint=None, output_folder=None):  # Define la función run_segmentation que acepta una ruta de imagen, un punto de control de SAM y una carpeta de salida.
    if image_path is None:  # Verifica si no se proporcionó una ruta de imagen.
        image_path = input("Introduce La Ruta De La Imagen: ")  # Solicita al usuario que ingrese la ruta de la imagen.
        sam_checkpoint = 'D:/Users/Leandro/Downloads/sam_vit_h_4b8939.pth'  # Define el punto de control del modelo SAM.
        output_folder = input("Introduce La Ruta Donde Guardar Las Máscaras: ")  # Solicita al usuario que ingrese la carpeta donde se guardarán las máscaras.

    image = cv2.imread(image_path)  # Lee la imagen desde la ruta proporcionada usando OpenCV.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convierte la imagen de BGR a RGB.

    # Visualización inicial lineal
    plt.figure(figsize=(10, 10))  # Crea una nueva figura de Matplotlib con un tamaño específico.
    plt.imshow(image)  # Muestra la imagen original en la figura.
    plt.title('Imagen Original')  # Establece el título de la figura.
    plt.axis('off')  # Desactiva los ejes del gráfico.
    plt.show()  # Muestra la figura.

    if 'nm_per_pixel' not in run_segmentation.__dict__:  # Verifica si la variable nm_per_pixel no está definida en el contexto de la función.
        nm_per_pixel = float(input("Introduce La Relación De Nanómetros Por Píxel: "))  # Solicita la relación de nanómetros por píxel al usuario.
        run_segmentation.nm_per_pixel = nm_per_pixel  # Almacena el valor en la función para uso futuro.
    else:
        nm_per_pixel = run_segmentation.nm_per_pixel  # Si ya está definido, usa el valor almacenado.

    filter_area = input("¿Deseas Filtrar Por Área? (si/no): ").lower()  # Pregunta al usuario si desea filtrar por área y convierte la respuesta a minúsculas.
    if filter_area == 'si':  # Si el usuario responde que sí:
        min_area_nm = float(input("Introduce El Área Mínima En Nanómetros Cuadrados: "))  # Solicita el área mínima en nanómetros cuadrados.
        max_area_nm = float(input("Introduce El Área Máxima En Nanómetros Cuadrados: "))  # Solicita el área máxima en nanómetros cuadrados.
        min_area_px = min_area_nm / (nm_per_pixel**2)  # Convierte el área mínima a píxeles cuadrados.
        max_area_px = max_area_nm / (nm_per_pixel**2)  # Convierte el área máxima a píxeles cuadrados.

    filter_diag = input("¿Deseas Filtrar Por Diagonal Mayor? (si/no): ").lower()  # Pregunta al usuario si desea filtrar por diagonal mayor.
    if filter_diag == 'si':  # Si el usuario responde que sí:
        min_diag_nm = float(input("Introduce La Diagonal Mayor Mínima En Nanómetros: "))  # Solicita la diagonal mayor mínima en nanómetros.
        max_diag_nm = float(input("Introduce La Diagonal Mayor Máxima En Nanómetros: "))  # Solicita la diagonal mayor máxima en nanómetros.
        min_diag_px = min_diag_nm / nm_per_pixel  # Convierte la diagonal mínima a píxeles.
        max_diag_px = max_diag_nm / nm_per_pixel  # Convierte la diagonal máxima a píxeles.

    print("Generando Máscaras...")  # Imprime un mensaje indicando que se están generando máscaras.
    tic = time.perf_counter()  # Inicia un temporizador para medir el tiempo de ejecución.

    def resize_image(image, target_size=1024):  # Define una función interna para redimensionar la imagen a un tamaño objetivo.
        long_side = max(image.shape[:2])  # Obtiene la longitud del lado más largo de la imagen.
        scale_factor = target_size / long_side  # Calcula el factor de escala para redimensionar.
        new_size = (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor))  # Calcula el nuevo tamaño de la imagen.
        resized_image = cv2.resize(image, new_size)  # Redimensiona la imagen usando OpenCV.
        return resized_image  # Devuelve la imagen redimensionada.

    image = resize_image(image, target_size=1024)  # Redimensiona la imagen a 1024 píxeles en su lado más largo.
    image_tensor = torch.as_tensor(image).permute(2, 0, 1).float()  # Convierte la imagen a un tensor de PyTorch y cambia el orden de las dimensiones a (canal, alto, ancho).

    model_type = "vit_h"  # Define el tipo de modelo a usar.
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)  # Carga el modelo SAM usando el punto de control especificado.
    mask_generator = SamAutomaticMaskGenerator(sam)  # Crea un generador de máscaras automáticas SAM.
    masks = mask_generator.generate(image_tensor.cpu().numpy().transpose(1, 2, 0))  # Genera las máscaras a partir del tensor de imagen.

    filtered_masks = masks  # Inicializa la lista de máscaras filtradas con todas las máscaras generadas.
    if filter_area == 'si':  # Si el usuario eligió filtrar por área:
        filtered_masks = [mask for mask in filtered_masks if min_area_px <= mask['area'] <= max_area_px]  # Filtra las máscaras por el área especificada.
    if filter_diag == 'si':  # Si el usuario eligió filtrar por diagonal:
        filtered_masks = [mask for mask in filtered_masks if min_diag_px <= mask['bbox'][2] <= max_diag_px]  # Filtra las máscaras por la diagonal especificada.

    show_colored_masks(image, filtered_masks)  # Muestra las máscaras filtradas superpuestas en la imagen original.

    toc = time.perf_counter()  # Detiene el temporizador.
    print(f"El Código Tomó {toc - tic:0.2f} Segundos En Ejecutarse.")  # Imprime el tiempo total de ejecución.

    rerun = input("¿Deseas Volver a Correr El Código Con Nueva Área/Diagonal Mayor Mínima y Máxima? (si/no): ").lower()  # Pregunta si se desea volver a correr el código.
    if rerun == 'si':  # Si el usuario responde que sí:
        run_segmentation(image_path=image_path, sam_checkpoint=sam_checkpoint, output_folder=output_folder)  # Llama a la función de nuevo con los parámetros actuales.
    else:  # Si el usuario no desea volver a correr el código:
        save_masks = input("¿Deseas Guardar Las Máscaras Generadas? (si/no): ").lower()  # Pregunta si se desea guardar las máscaras generadas.
        
        if save_masks == 'si':  # Si el usuario desea guardar las máscaras:
            print("Guardando Máscaras Generadas...")  # Imprime un mensaje indicando que se están guardando las máscaras.
            if not os.path.exists(output_folder):  # Verifica si la carpeta de salida no existe.
                os.makedirs(output_folder)  # Crea la carpeta de salida.

            mask_filenames = []  # Inicializa una lista para almacenar los nombres de archivo de las máscaras generadas.
            for idx, mask in enumerate(filtered_masks):  # Itera sobre las máscaras filtradas.
                mask_img = np.zeros_like(image)  # Crea una imagen de ceros (negra) del mismo tamaño que la imagen original.
                mask_img[mask['segmentation']] = [255, 255, 255]  # Establece los píxeles de la máscara a blanco en la imagen negra.
                mask_output_path = os.path.join(output_folder, f'mask_{idx}.png')  # Define la ruta de salida para la máscara.
                cv2.imwrite(mask_output_path, cv2.cvtColor(mask_img, cv2.COLOR_RGB2BGR))  # Guarda la máscara en la carpeta de salida en formato BGR.
                mask_filenames.append(f'mask_{idx}.png')  # Añade el nombre del archivo a la lista de nombres de archivo de máscaras.


            discard_masks = input("¿Deseas Descartar Máscaras? (si/no): ").lower()  # Solicita al usuario si desea descartar máscaras y convierte la respuesta a minúsculas.
            if discard_masks == 'si':  # Verifica si la respuesta del usuario es 'si'.
                app = QApplication([])  # Crea una nueva instancia de la aplicación Qt sin argumentos de línea de comandos.
                window = MaskDiscardWindow(image, filtered_masks, mask_filenames, output_folder)  # Crea una instancia de la ventana MaskDiscardWindow, pasando la imagen, las máscaras filtradas, los nombres de archivo de las máscaras y la carpeta de salida.
                window.show()  # Muestra la ventana en pantalla.
                app.exec_()  # Inicia el bucle de eventos de la aplicación, permitiendo que la interfaz gráfica responda a las interacciones del usuario.


# Ejecutar la función principal de segmentación
run_segmentation()
