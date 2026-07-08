# %% [code]
# %% [code]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from IPython.display import display
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.transforms import v2
from umap import UMAP

from natural_images import AugmentationWrapper
from screen_images import plot_image_list
from dataset_utils import get_labels


def extract_labels(dataset):
    labels = get_labels(dataset)
    return labels.cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)


def class_count_frame(dataset, class_names):
    labels = extract_labels(dataset)
    counts = pd.Series(labels).value_counts().sort_index()
    name_lookup = {index: name for index, name in enumerate(class_names)}
    return pd.DataFrame({
        'class_index': counts.index,
        'class_name': [name_lookup.get(index, str(index)) for index in counts.index],
        'count': counts.values,
    })


def numbered_table(frame, start=1, index_name='rank'):
    numbered = frame.reset_index(drop=True).copy()
    numbered.index = np.arange(start, start + len(numbered))
    numbered.index.name = index_name
    return numbered


def split_summary_frame(train_ds, val_ds, test_ds):
    splits = [('train', train_ds), ('val', val_ds), ('test', test_ds)]
    total = sum(len(ds) for _, ds in splits)
    return pd.DataFrame({
        'split': [name for name, _ in splits],
        'samples': [len(ds) for _, ds in splits],
        'share_%': [round(len(ds) / total * 100, 2) for _, ds in splits],
    })


def plot_class_count_histogram(counts, title):
    plt.figure(figsize=(8, 4))
    plt.hist(counts.values, bins=min(20, max(5, len(counts) // 4)), color='#2a9d8f')
    plt.title(title)
    plt.xlabel('Images per class')
    plt.ylabel('Number of classes')
    plt.tight_layout()
    plt.show()


def show_sample_grid(dataset, class_names=None, title='Samples', n_samples=9, seed=317):
    sample_count = min(n_samples, len(dataset))
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=sample_count, replace=False)
    images = []
    titles = []
    for index in indices:
        image, label = dataset[index]
        image_array = image.permute(1, 2, 0).cpu().numpy() if torch.is_tensor(image) else np.asarray(image)
        images.append(image_array)
        if class_names is not None and label < len(class_names):
            titles.append(class_names[label])
        else:
            titles.append(str(label))
    print(title)
    plot_image_list(images, titles=titles, cols=3)


def make_analysis_transform(resize=(48, 48)):
    return v2.Compose([
        v2.Resize(resize),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])


def dataset_to_matrix(dataset, resize=(48, 48), batch_size=128):
    analysis_transform = make_analysis_transform(resize=resize)
    loader = DataLoader(
        AugmentationWrapper(dataset, analysis_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    feature_blocks = []
    label_blocks = []
    for images, labels in loader:
        feature_blocks.append(images.flatten(1).cpu().numpy())
        label_blocks.append(labels.cpu().numpy())
    return np.concatenate(feature_blocks), np.concatenate(label_blocks)


def project_with_umap(features, seed=317):
    scaled = StandardScaler().fit_transform(features)
    n_components = min(50, scaled.shape[0] - 1, scaled.shape[1])
    reduced = PCA(n_components=n_components, random_state=seed).fit_transform(scaled) if n_components >= 2 else scaled
    projector = UMAP(n_neighbors=15, min_dist=0.15, metric='euclidean', random_state=seed)
    return projector.fit_transform(reduced)


def plot_umap(coords, labels, class_names, title):
    frame = pd.DataFrame(coords, columns=['umap_1', 'umap_2'])
    frame['label'] = labels
    frame['class_name'] = frame['label'].map(lambda index: class_names[index])
    palette = dict(zip(class_names, plt.cm.tab20.colors[:len(class_names)]))

    plt.figure(figsize=(11, 8))
    for class_name in class_names:
        subset = frame[frame['class_name'] == class_name]
        plt.scatter(
            subset['umap_1'],
            subset['umap_2'],
            s=18,
            alpha=0.8,
            label=class_name,
            color=palette[class_name],
            linewidths=0,
        )
    plt.title(title)
    plt.xlabel('UMAP-1')
    plt.ylabel('UMAP-2')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title='Class', fontsize=7)
    plt.tight_layout()
    plt.show()


def summarize_umap(coords, labels, class_names, k=15):
    labels = np.asarray(labels)
    n_classes = len(class_names)
    centroids = np.vstack([coords[labels == class_index].mean(axis=0) for class_index in range(n_classes)])
    centroid_distance = pd.DataFrame(pairwise_distances(centroids), index=class_names, columns=class_names)

    n_neighbors = min(k + 1, len(coords))
    neighbors = NearestNeighbors(n_neighbors=n_neighbors).fit(coords).kneighbors(return_distance=False)[:, 1:]
    overlap = np.zeros((n_classes, n_classes), dtype=float)
    counts = np.bincount(labels, minlength=n_classes)

    for row_index, source_class in enumerate(labels):
        neighbor_labels = labels[neighbors[row_index]]
        overlap[source_class] += np.bincount(neighbor_labels, minlength=n_classes) / max(1, len(neighbor_labels))

    overlap = overlap / counts[:, None]
    overlap_df = pd.DataFrame(overlap, index=class_names, columns=class_names)
    purity = pd.Series(np.diag(overlap), index=class_names, name=f'purity@{k}').sort_values(ascending=False)

    pair_rows = []
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            pair_rows.append({
                'class_a': class_names[i],
                'class_b': class_names[j],
                'neighbor_overlap': 0.5 * (overlap[i, j] + overlap[j, i]),
                'centroid_distance': centroid_distance.iat[i, j],
            })

    pairwise = pd.DataFrame(pair_rows)
    return {
        'centroid_distance': centroid_distance,
        'overlap': overlap_df,
        'purity': purity,
        'pairwise': pairwise,
    }


def print_umap_summary(summary, title):
    print(f'\n{title}')
    print('Per-class neighborhood purity')
    display(summary['purity'].to_frame())
    print('Class-by-class overlap matrix')
    display(summary['overlap'].round(3))
    print('Class centroid distances in UMAP space')
    display(summary['centroid_distance'].round(3))
    print('Most overlapping class pairs')
    display(summary['pairwise'].sort_values('neighbor_overlap', ascending=False).head(10).round(3))
    print('Closest class centroids')
    display(summary['pairwise'].sort_values('centroid_distance').head(10).round(3))