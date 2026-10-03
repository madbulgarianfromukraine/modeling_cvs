"""
Utility module for randomly sampling images from category folders
where images follow the pattern: image_{4-digit number}.<ext> (e.g., image_0001.jpg).
"""

import argparse
from pathlib import Path
import random
import re
from typing import Callable, Dict, List, Optional, Tuple, Union
from PIL import Image


def find_category_images(
    category_dir: Union[str, Path],
    pattern: str = r"^image_\d{4}\.(jpg|jpeg|png|bmp|webp)$"
) -> List[Path]:
    """
    Finds all image files in category_dir matching the 4-digit index image pattern.
    """
    category_path = Path(category_dir)
    if not category_path.exists() or not category_path.is_dir():
        return []

    regex = re.compile(pattern, re.IGNORECASE)
    matching_files = [
        f for f in category_path.iterdir()
        if f.is_file() and regex.match(f.name)
    ]
    return sorted(matching_files, key=lambda p: p.name)


def load_random_category_images(
    root_dir: Union[str, Path],
    categories: List[str],
    num_samples: int = 3,
    seed: Optional[int] = None,
    convert_rgb: bool = True,
    resize: Optional[Tuple[int, int]] = (300, 200),
    transform: Optional[Callable] = None,
    pattern: str = r"^image_\d{4}\.(jpg|jpeg|png|bmp|webp)$"
) -> Dict[str, List[Tuple[Path, Union[Image.Image, object]]]]:
    """
    Loads `num_samples` randomly selected images for each category in `categories`.

    Parameters
    ----------
    root_dir : str or Path
        Root directory containing category subfolders (e.g. ./data/caltech101/101_ObjectCategories).
    categories : list of str
        List of category folder names to sample from.
    num_samples : int, default=3
        Number of images to randomly select per category.
    seed : int, optional
        Random seed for reproducible sampling.
    convert_rgb : bool, default=True
        Whether to convert PIL images to RGB mode.
    resize : tuple of (int, int), optional, default=(300, 200)
        Dimensions (width, height) to resize images to. Set to None to keep original size.
    transform : callable, optional
        Optional transformation function to apply to each loaded PIL Image.
    pattern : str
        Regex pattern matching the image filenames (default: 'image_XXXX.<ext>').

    Returns
    -------
    dict[str, list[tuple[Path, Image.Image | object]]]
        Dictionary mapping category names to lists of (file_path, image) tuples.
    """
    root_path = Path(root_dir)
    if not root_path.exists():
        raise FileNotFoundError(f"Root directory '{root_path}' does not exist.")

    if seed is not None:
        random.seed(seed)

    sampled_results: Dict[str, List[Tuple[Path, Union[Image.Image, object]]]] = {}

    for cat in categories:
        cat_dir = root_path / cat
        if not cat_dir.exists() or not cat_dir.is_dir():
            print(f"Warning: Category directory '{cat_dir}' not found. Skipping.")
            sampled_results[cat] = []
            continue

        images = find_category_images(cat_dir, pattern=pattern)

        if not images:
            print(f"Warning: No images matching pattern '{pattern}' found in '{cat_dir}'. Skipping.")
            sampled_results[cat] = []
            continue

        k = min(num_samples, len(images))
        if k < num_samples:
            print(f"Notice: Category '{cat}' has only {len(images)} matching images (requested {num_samples}).")

        selected_files = random.sample(images, k)
        loaded_list = []

        resample_filter = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC)

        for img_path in selected_files:
            img = Image.open(img_path)
            if convert_rgb:
                img = img.convert("RGB")
            if resize is not None:
                img = img.resize(resize, resample_filter)
            if transform is not None:
                transformed_img = transform(img)
                loaded_list.append((img_path, transformed_img))
            else:
                loaded_list.append((img_path, img))

        sampled_results[cat] = loaded_list

    return sampled_results


def download_enrico_wireframes_if_missing(root_dir: Union[str, Path] = "./data") -> Path:
    """
    Downloads wireframes.zip from Aalto University Enrico resources if not present locally.
    """
    import urllib.request

    root_p = Path(root_dir)
    wf_zip = root_p / "wireframes.zip"
    wf_dir = root_p / "wireframes"
    if not wf_zip.exists() and not wf_dir.exists():
        url = "http://userinterfaces.aalto.fi/enrico/resources/wireframes.zip"
        print(f"Wireframes dataset not found. Downloading wireframes.zip from {url}...")
        root_p.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, wf_zip)
        print(f"Successfully downloaded wireframes.zip ({wf_zip.stat().st_size / 1e6:.1f} MB)!")
    return wf_zip


def load_random_enrico_category_images(
    root_dir: Union[str, Path] = "./data",
    categories: Optional[List[str]] = None,
    num_samples: int = 3,
    seed: Optional[int] = None,
    convert_rgb: bool = True,
    resize: Optional[Tuple[int, int]] = (180, 320),
    transform: Optional[Callable] = None,
    pair_with_wireframes: bool = True,
    gap_px: int = 4
) -> Dict[str, List[Tuple[Path, Union[Image.Image, object]]]]:
    """
    Loads `num_samples` randomly selected UI screen images (paired side-by-side with wireframes)
    for each category in the Enrico dataset.

    Parameters
    ----------
    root_dir : str or Path
        Root directory containing design_topics.csv and screenshots / wireframes zip files or folders.
    categories : list of str, optional
        List of Enrico topic category names (e.g. ['login', 'menu', 'gallery', 'settings', 'tutorial']).
        If None, defaults to ['login', 'menu', 'gallery', 'settings', 'tutorial'].
    num_samples : int, default=3
        Number of image pairs to randomly select per category.
    seed : int, optional
        Random seed for reproducible sampling.
    convert_rgb : bool, default=True
        Whether to convert images to RGB mode.
    resize : tuple of (int, int), optional, default=(180, 320)
        Dimensions (width, height) to resize EACH image (screenshot and wireframe) to.
    transform : callable, optional
        Optional transformation to apply to loaded PIL Images.
    pair_with_wireframes : bool, default=True
        Whether to pair each screenshot side-by-side with its corresponding wireframe.
    gap_px : int, default=4
        Pixel spacing between screenshot and wireframe in side-by-side composite.

    Returns
    -------
    dict[str, list[tuple[Path, Image.Image | object]]]
        Dictionary mapping category names to lists of (screen_id_path, composite_image) tuples.
    """
    import pandas as pd
    import zipfile

    root_path = Path(root_dir)
    csv_path = root_path / "design_topics.csv"
    if not csv_path.exists():
        if (root_path / "data" / "design_topics.csv").exists():
            root_path = root_path / "data"
            csv_path = root_path / "design_topics.csv"
        else:
            raise FileNotFoundError(f"Could not find design_topics.csv in '{root_dir}'.")

    df = pd.read_csv(csv_path)
    df["screen_id"] = df["screen_id"].astype(str)

    if categories is None:
        categories = ["login", "menu", "gallery", "settings", "tutorial"]

    if seed is not None:
        random.seed(seed)

    if pair_with_wireframes:
        download_enrico_wireframes_if_missing(root_path)

    # Locate screenshots directory or zip
    sc_dir = root_path / "screenshots/screenshots"
    sc_zip_path = root_path / "screenshots.zip"
    sc_zip_obj = zipfile.ZipFile(sc_zip_path, "r") if (not sc_dir.exists() and sc_zip_path.exists()) else None

    # Locate wireframes directory or zip
    wf_dir = root_path / "wireframes/wireframes"
    wf_zip_path = root_path / "wireframes.zip"
    wf_zip_obj = zipfile.ZipFile(wf_zip_path, "r") if (not wf_dir.exists() and wf_zip_path.exists()) else None

    sampled_results: Dict[str, List[Tuple[Path, Union[Image.Image, object]]]] = {}
    resample_filter = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC)

    def _open_image_from_sources(screen_id: str, is_wireframe: bool) -> Optional[Image.Image]:
        ext = "png" if is_wireframe else "jpg"
        target_dir = wf_dir if is_wireframe else sc_dir
        target_zip = wf_zip_obj if is_wireframe else sc_zip_obj
        filename = f"{screen_id}.{ext}"

        file_path = target_dir / filename
        if target_dir.exists() and file_path.exists():
            return Image.open(file_path).copy()

        if target_zip is not None:
            matches = [n for n in target_zip.namelist() if n.endswith(f"/{filename}") or n == filename]
            if matches:
                with target_zip.open(matches[0]) as zf:
                    return Image.open(zf).copy()

        return None

    for cat in categories:
        cat_df = df[df["topic"] == cat]
        if cat_df.empty:
            print(f"Warning: Category '{cat}' not found in {csv_path.name}. Skipping.")
            sampled_results[cat] = []
            continue

        screen_ids = cat_df["screen_id"].tolist()
        k = min(num_samples, len(screen_ids))
        selected_ids = random.sample(screen_ids, k)

        loaded_list = []
        for sid in selected_ids:
            sc_img = _open_image_from_sources(sid, is_wireframe=False)
            if sc_img is None:
                print(f"Warning: Could not locate screenshot for screen_id {sid} in '{cat}'. Skipping.")
                continue

            if convert_rgb:
                sc_img = sc_img.convert("RGB")
            if resize is not None:
                sc_img = sc_img.resize(resize, resample_filter)

            final_img = sc_img

            if pair_with_wireframes:
                wf_img = _open_image_from_sources(sid, is_wireframe=True)
                if wf_img is not None:
                    if convert_rgb:
                        wf_img = wf_img.convert("RGB")
                    if resize is not None:
                        wf_img = wf_img.resize(resize, resample_filter)

                    # Create side-by-side composite: Screenshot | Wireframe
                    w_sc, h_sc = sc_img.size
                    w_wf, h_wf = wf_img.size
                    composite_w = w_sc + gap_px + w_wf
                    composite_h = max(h_sc, h_wf)

                    composite = Image.new("RGB", (composite_w, composite_h), color=(255, 255, 255))
                    composite.paste(sc_img, (0, 0))
                    composite.paste(wf_img, (w_sc + gap_px, 0))
                    final_img = composite
                else:
                    print(f"Notice: Wireframe not found for screen_id {sid}. Using screenshot only.")

            if transform is not None:
                final_img = transform(final_img)

            loaded_list.append((Path(f"screen_{sid}"), final_img))

        sampled_results[cat] = loaded_list

    if sc_zip_obj is not None:
        sc_zip_obj.close()
    if wf_zip_obj is not None:
        wf_zip_obj.close()

    return sampled_results




def plot_sampled_images(
    sampled_data: Dict[str, List[Tuple[Path, Image.Image]]],
    save_path: Optional[Union[str, Path]] = None,
    figsize: Optional[Tuple[int, int]] = None
) -> None:
    """
    Visualizes the sampled images in a grid format (categories x samples).
    """
    import matplotlib.pyplot as plt

    categories = [cat for cat, imgs in sampled_data.items() if len(imgs) > 0]
    if not categories:
        print("No sampled images to display.")
        return

    max_samples = max(len(sampled_data[cat]) for cat in categories)
    n_cats = len(categories)

    if figsize is None:
        figsize = (3 * max_samples, 2.5 * n_cats)

    fig, axes = plt.subplots(n_cats, max_samples, figsize=figsize, squeeze=False)

    for r, cat in enumerate(categories):
        img_tuples = sampled_data[cat]
        for c in range(max_samples):
            ax = axes[r, c]
            if c < len(img_tuples):
                img_path, img = img_tuples[c]
                ax.imshow(img)
                ax.set_title(f"{cat}\n{img_path.name}", fontsize=9)
            ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Grid plot saved to: {save_path}")
    plt.show()


def plot_diagnostic_figure(
    sampled_data: Dict[str, List[Tuple[Path, Image.Image]]],
    save_path: Optional[Union[str, Path]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    wspace: float = 0.08,
    hspace: float = 0.12,
    fontsize: int = 11,
    dpi: int = 300,
    show_plot: bool = False
) -> None:
    """
    Creates a clean diagnostic figure showing category rows of images:
      class 1: image -- image -- image
      class 2: image -- image -- image
      ...
    No other titles except the category name on the left of each row.

    Parameters
    ----------
    sampled_data : dict
        Dictionary mapping category names to lists of (file_path, PIL_Image) tuples.
    save_path : str or Path, optional
        Filepath to save output figure.
    figsize : tuple of (width, height), optional
        Custom figure size.
    wspace : float, default=0.08
        Horizontal spacing between image columns.
    hspace : float, default=0.12
        Vertical spacing between class rows.
    fontsize : int, default=11
        Font size of category labels on the left.
    dpi : int, default=300
        DPI resolution for rendering and saving.
    show_plot : bool, default=False
        Whether to display the plot interactively via plt.show().
    """
    import matplotlib.pyplot as plt

    categories = [cat for cat in sampled_data.keys() if len(sampled_data[cat]) > 0]
    if not categories:
        print("No sampled images available for diagnostic figure.")
        return

    num_rows = len(categories)
    num_cols = max(len(sampled_data[cat]) for cat in categories)

    if figsize is None:
        figsize = (2.5 * num_cols + 1.2, 2.0 * num_rows)

    fig, axes = plt.subplots(
        nrows=num_rows,
        ncols=num_cols,
        figsize=figsize,
        dpi=dpi,
        squeeze=False
    )

    for r, cat in enumerate(categories):
        img_tuples = sampled_data[cat]
        for c in range(num_cols):
            ax = axes[r, c]
            if c < len(img_tuples):
                _, img = img_tuples[c]
                ax.imshow(img)
            else:
                ax.set_visible(False)

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            if c == 0:
                ax.set_ylabel(
                    cat,
                    rotation=0,
                    labelpad=15,
                    ha="right",
                    va="center",
                    fontsize=fontsize,
                    fontweight="bold"
                )

    plt.subplots_adjust(wspace=wspace, hspace=hspace)

    if save_path:
        save_p = Path(save_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=dpi)
        print(f"Diagnostic figure saved to: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Randomly pick images from category directories (image_XXXX format)."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="./data/caltech101/101_ObjectCategories",
        help="Path to dataset root folder containing category directories."
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["accordion", "airplanes", "brain", "Faces", "Leopards"],
        help="List of category names to load."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Number of images to randomly pick per category (default: 3)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling reproducibility."
    )
    parser.add_argument(
        "--save-grid",
        type=str,
        default=None,
        help="Optional filepath to save visualization grid (e.g. output_grid.png)."
    )

    args = parser.parse_args()

    print(f"Root dataset directory: {args.root}")
    print(f"Categories to sample:   {args.categories}")
    print(f"Samples per category:   {args.samples}")

    results = load_random_category_images(
        root_dir=args.root,
        categories=args.categories,
        num_samples=args.samples,
        seed=args.seed
    )

    print("\n--- Summary of Loaded Images ---")
    for category, items in results.items():
        print(f"Category '{category}': {len(items)} image(s) loaded")
        for path, img in items:
            print(f"  - Path: {path} | Size: {img.size} | Mode: {img.mode}")

    if args.save_grid:
        plot_sampled_images(results, save_path=args.save_grid)
