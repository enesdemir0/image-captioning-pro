import json
import os
import shutil
import argparse
import urllib.request


COCO_IMAGE_URL = "http://images.cocodataset.org/train2014/{filename}"


def _download_image(img_name, dest_path):
    url = COCO_IMAGE_URL.format(filename=img_name)
    try:
        urllib.request.urlretrieve(url, dest_path)
        return True
    except Exception as e:
        print(f"  WARN: could not download {img_name}: {e}")
        return False


def create_sample_dataset(
    src_annotations="data/annotations/captions_train2014.json",
    src_image_dir=None,
    output_dir="data/sample",
    n_images=20,
    image_prefix="COCO_train2014_",
):
    out_img_dir = os.path.join(output_dir, "images")
    out_ann_dir = os.path.join(output_dir, "annotations")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_ann_dir, exist_ok=True)

    with open(src_annotations, "r") as f:
        coco = json.load(f)

    # Collect n_images unique image IDs from the annotation file
    selected_ids = []
    seen = set()
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id not in seen:
            seen.add(img_id)
            selected_ids.append(img_id)
        if len(selected_ids) == n_images:
            break

    print(f"Selected {len(selected_ids)} image IDs from annotations.")

    # Copy from local dir OR download from COCO servers
    successful_ids = []
    for img_id in selected_ids:
        img_name = f"{image_prefix}{str(img_id).zfill(12)}.jpg"
        dst = os.path.join(out_img_dir, img_name)

        if os.path.exists(dst):
            successful_ids.append(img_id)
            continue

        if src_image_dir:
            src = os.path.join(src_image_dir, img_name)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                successful_ids.append(img_id)
                print(f"  copied  {img_name}")
                continue

        # Fall back to downloading from COCO
        print(f"  downloading {img_name} ...")
        if _download_image(img_name, dst):
            successful_ids.append(img_id)

    print(f"\n{len(successful_ids)}/{len(selected_ids)} images ready in '{out_img_dir}'")

    # Build filtered annotations for only the images we actually got
    successful_set = set(successful_ids)
    sample_annotations = [a for a in coco["annotations"] if a["image_id"] in successful_set]

    out_ann_path = os.path.join(out_ann_dir, "captions_sample.json")
    with open(out_ann_path, "w") as f:
        json.dump({"annotations": sample_annotations}, f)

    print(f"Captions : {len(sample_annotations)}  → {out_ann_path}")
    print()
    print("Next steps:")
    print("  dvc add data/sample")
    print("  git add data/sample.dvc data/.gitignore")
    print("  git commit -m 'feat: add DVC-tracked sample dataset'")
    print("  dvc push")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Carve a tiny sample from MS-COCO for pipeline dev."
    )
    parser.add_argument("--n", type=int, default=20, help="Number of unique images (default: 20)")
    parser.add_argument(
        "--src-images",
        default=None,
        help="Local COCO image directory. If omitted, images are downloaded from COCO servers.",
    )
    parser.add_argument(
        "--src-annotations",
        default="data/annotations/captions_train2014.json",
    )
    parser.add_argument("--output", default="data/sample")
    args = parser.parse_args()

    create_sample_dataset(
        src_annotations=args.src_annotations,
        src_image_dir=args.src_images,
        output_dir=args.output,
        n_images=args.n,
    )
