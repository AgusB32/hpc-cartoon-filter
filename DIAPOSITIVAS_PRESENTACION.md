# Presentación TPFI - Computación Avanzada

**Tema:** Generación de Imágenes Cartoon (Secuencial, OpenMP, MPI e Híbrido)

---

### 1. Objetivo del Trabajo
- Implementar un generador de imágenes estilo *cartoon* a partir de imágenes RGB.
- Desarrollar y comparar cuatro versiones del algoritmo:
  1. Secuencial
  2. Memoria Compartida (OpenMP)
  3. Memoria Distribuida (MPI)
  4. Paralela Híbrida (MPI + OpenMP)
- Evaluar el desempeño midiendo tiempo de ejecución, *Speedup* y *Eficiencia*.

---

### 2. Etapas del Algoritmo
El procesamiento combina dos ramas independientes sobre la imagen RGB:

1. **Detección de bordes:**
   - **Filtrado (Blur):** filtro de media (3x3 o 5x5) para reducir ruido.
   - **Resaltado (Sobel):** cálculo del gradiente horizontal ($G_x$) y vertical ($G_y$), y su magnitud $M = \sqrt{G_x^2 + G_y^2}$.
   - **Umbralado:** si $M \ge 64$ el píxel toma valor 255, si no 0.
2. **Posterizado:**
   - Reduce los 256 niveles de cada canal RGB a un rango menor (3 o 9 valores).
3. **Suma final:**
   - Se suman los bordes detectados y la imagen posterizada saturando en $[0, 255]$ (con borde auxiliar neutro negro).

---

### 3. Implementación de cada Versión

- **Secuencial (`cartoon_seq.cpp`):**
  - Procesa fila por fila en un único hilo. Sirve como referencia base ($T_1$) para calcular Speedup y Eficiencia.
- **OpenMP (`cartoon_omp.cpp`):**
  - Paraleliza el bucle de filas mediante `#pragma omp parallel for schedule(static)`.
  - Todos los hilos leen y escriben sobre la misma memoria compartida sin colisiones.
- **MPI (`cartoon_mpi.cpp`):**
  - El proceso 0 carga la imagen y reparte franjas horizontales de filas a cada proceso usando `MPI_Send` / `MPI_Recv`.
  - Se agregan filas vecinas (*halo*) arriba y abajo de cada franja para poder calcular las máscaras 3x3 y 5x5 en los bordes.
  - Al finalizar, `MPI_Gatherv` reúne las franjas procesadas en el proceso 0.
- **Híbrido (`cartoon_hybrid.cpp`):**
  - Combina el reparto de franjas por procesos con MPI (`MPI_Init_thread`) y el procesamiento interno de cada franja con hilos OpenMP.

---

### 4. Pruebas Realizadas (`benchmark.py`)
- **Imágenes:** `800x800`, `2000x2000` y `5000x5000`.
- **Filtros:** `3x3` y `5x5`.
- **Posterizado:** `3` y `9` niveles.
- **Configuraciones:**
  - OpenMP: 2, 4, 8 y 12 hilos.
  - MPI: 2, 4, 8 y 12 procesos.
  - Híbrido: 2p×2t, 2p×4t, 4p×2t y 3p×4t.
- **Métricas:**
  - $\text{Speedup} = T_{\text{sec}} / T_{\text{par}}$
  - $\text{Eficiencia} = \text{Speedup} / N_{\text{workers}}$

---

### 5. Conclusiones
- **Correctitud:** las cuatro versiones producen exactamente la misma imagen de salida.
- **Tamaño del problema:** en imágenes chicas (`800x800`) el costo de crear hilos o enviar mensajes MPI reduce la ganancia; en imágenes grandes (`5000x5000`) el cómputo domina y se logran los mejores speedups.
- **Comparativa:** en una única máquina local, OpenMP obtiene el menor tiempo por no copiar memoria entre procesos, mientras que MPI y el enfoque híbrido permiten escalar el problema a múltiples nodos.
