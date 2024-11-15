![](img/GFA-logo1.png)

# Segmentacion y Analisis Estadistico De NanoParticulas Con SAM 

Este trabajo fue llevado por Michel Leandro dentro de las Practicas profesionalizantes de la escuela de educacion tecnica N°1 "Luciano Reyes", guiado Cerrotta Santiago de parte del laboratorio de fotonica aplicada de la UTN-FRD durante 200 horas entre agosto y noviembre de 2024.

### 1. Objetivo

Crear un programa a partir de SAM que permita mediante un filtrado de area o diagonal mayor segmentar nanoparticulas para luego obtener estadistica en histograma de las mismas.

### 2. Introducción

Segment Anything Model o SAM es un modelo de segmentacion de imagenes open source hecho por la empresa Meta que nos permite a partir de diversos promts obtener mascaras/segmentaciones muy identicas a la forma original con lo cual se busco analizar nanoparticulas para generar mascaras filtrando por area, diagonal mayor o tambien descartando interactivamente mascaras generadas no deseadas para luego obtener histogramas de las mismas.

<div align="center">
	<img src="/img/0.gif">
</div>
<div align="center">
	<em> SAM </em>
</div>
	
### 3. Instalacion y Manual De Uso

En el siguiente enlace se indica paso a paso la instalcion y forma de uso del programa.

_[Manual_De_Uso](https://github.com/Leandrituw/Segmentacion-Y-Analisis-Estadistico-De-NanoParticulas-Con-SAM/blob/main/Otros/Manual_De_Uso_SAM.pdf)_

_[Presentacion De Proyectos](https://www.canva.com/design/DAGVnWDVBz0/gKAowva_oulsz6L4RoCJJQ/edit?utm_content=DAGVnWDVBz0&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)_

### 4. Resultados

Luego de varias consultas con un representante del grupo de nanofotonica e iteraciones para mejorar el codigo se obtuvieron los siguientes resultados.

1. NANOPARTICULAS REDONDAS

<div align="center">
	<img src="/img/1.png">
</div>
<div align="center">
	<em> Foto Original </em>
</div>

<div align="center">
	<img src="/img/2.png">
</div>
<div align="center">
	<em> Foto Segmentada </em>
</div>

<div align="center">
	<img src="/img/Descarte.gif">
</div>
<div align="center">
	<em> Descarte Interactivo </em>
</div>

<div align="center">
	<img src="/img/3.png">
</div>
<div align="center">
	<em> Estadistica </em>
</div>


1. NANOPARTICULAS CILINDRICAS

<div align="center">
	<img src="/img/4.png">
</div>
<div align="center">
	<em> Foto Original </em>
</div>

<div align="center">
	<img src="/img/5.png">
</div>
<div align="center">
	<em> Foto Segmentada </em>
</div>

<div align="center">
	<img src="/img/6.png">
</div>
<div align="center">
	<em> Estadistica </em>
</div>

### 5. Conclusion 

Despues de un analisis sobre SAM, su codigo y las necesidades del grupo de nanofotonica se formo un programa capaz de generar mascaras muy cercanas a la forma de las nanoparticulas filtrando segun su area o diagonal mayor ademas de tener un grafico interactivo para el descarte de mascaras que no se deseen para posteriormente generar un analisis en histogramas sobre todas las mascaras generadas.

### 6. Recursos 

Dentro Del Repositorio Se Encuentra: 
* 📂Codigos📂 
* 🡪SAM.py🡨
* 🡪Estadistica_Mascaras.py🡨
* 📂Fotos📂
* 🡪NanoParticulas.zip🡨
* 🡪NanoParticulasCalibradas.zip🡨
* 📂Otros📂
* 🡪Manual_De_Uso-Bitacoras-Informacion🡨
  
* ⚠️SE RECOMIENDA LEER LOS COMENTARIOS DE LOS CODIGOS⚠️

### 7. Fuentes

_[Segment Anything Model](https://segment-anything.com/)_

_[SAM GitHub](https://github.com/facebookresearch/segment-anything)_
