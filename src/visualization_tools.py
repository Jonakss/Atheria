# /home/jonathan.correa/Projects/Atheria/src/visualization_tools.py
#
# Este módulo contiene funciones para generar visualizaciones avanzadas
# para el proyecto Aetheria, como el Disco de Poincaré y Slices 3D.
#
# Dependencias:
# pip install torch sklearn matplotlib numpy

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE

def visualize_poincare_disk(latent_space_tensor, energy_values=None, title="Visualización en Disco de Poincaré"):
    """
    Proyecta un espacio latente a 2D usando t-SNE y lo visualiza en un Disco de Poincaré.

    Args:
        latent_space_tensor (torch.Tensor): Tensor del espacio latente.
                                            Shape: (batch, channels, height, width)
        energy_values (np.array, optional): Valores para colorear los puntos. Defaults to None.
        title (str, optional): Título del gráfico.
    """
    print("--- Generando Visualización en Disco de Poincaré ---")
    if latent_space_tensor.dim() != 4:
        raise ValueError("El tensor del espacio latente debe tener 4 dimensiones (B, C, H, W)")

    # Aplanar el tensor a (batch, features)
    flat_latent = latent_space_tensor.reshape(latent_space_tensor.shape, -1).cpu().detach().numpy()
    print(f"Shape del espacio latente aplanado: {flat_latent.shape}")

    if flat_latent.shape < 2:
        print("ERROR: Se necesita un batch de al menos 2 para usar t-SNE.")
        return

    # Reducir dimensionalidad a 2D con t-SNE
    perplexity = min(30.0, flat_latent.shape - 1.0)
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate='auto', init='random')
    embedding_2d = tsne.fit_transform(flat_latent)

    # Normalizar y escalar para que quepa dentro del disco
    norms = np.linalg.norm(embedding_2d, axis=1, keepdims=True)
    embedding_normalized = (embedding_2d / (norms + 1e-8)) * 0.95

    # Visualización con Matplotlib
    fig, ax = plt.subplots(figsize=(10, 10))
    circle = plt.Circle((0, 0), 1, color='lightblue', alpha=0.2, zorder=0)
    ax.add_artist(circle)
    
    scatter = ax.scatter(
        embedding_normalized[:, 0], 
        embedding_normalized[:, 1], 
        c=energy_values if energy_values is not None else np.random.rand(embedding_normalized.shape), 
        cmap='viridis',
        zorder=1
    )
    
    fig.colorbar(scatter, ax=ax, label="Valor de Energía/Propiedad")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title)
    plt.xlabel("Componente 1")
    plt.ylabel("Componente 2")
    
    output_path = "/home/jonathan.correa/Projects/Atheria/poincare_visualization.png"
    plt.savefig(output_path)
    print(f"Visualización guardada en: {output_path}")
    plt.close()

def visualize_3d_slices(tensor, title="Visualización de Canales como Slices 3D"):
    """
    Visualiza los canales de un tensor como slices apilados en un cubo 3D.

    Args:
        tensor (torch.Tensor): Tensor a visualizar.
                               Shape: (channels, height, width) o (batch, channels, height, width)
        title (str, optional): Título del gráfico.
    """
    print("--- Generando Visualización de Slices 3D ---")
    if tensor.dim() == 4:
        tensor = tensor
    
    if tensor.dim() != 3:
        raise ValueError("El tensor debe tener 3 dimensiones (C, H, W)")

    data = tensor.cpu().detach().numpy()
    num_channels, height, width = data.shape

    x = np.arange(width)
    y = np.arange(height)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    for z in range(num_channels):
        ax.contourf(X, Y, data[z, :, :], zdir='z', offset=z, cmap='viridis', alpha=0.7)

    ax.set_zlim(0, num_channels)
    ax.set_xlabel('Ancho (X)')
    ax.set_ylabel('Altura (Y)')
    ax.set_zlabel('Canal (Z)')
    ax.set_title(title)
    ax.view_init(elev=30, azim=-45)
    
    output_path = "/home/jonathan.correa/Projects/Atheria/3d_slices_visualization.png"
    plt.savefig(output_path)
    print(f"Visualización guardada en: {output_path}")
    plt.close()