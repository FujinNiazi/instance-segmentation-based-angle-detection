import json

def filter_to_line_only(json_path, output_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Only keep "line" class and set its new category_id to 0
    target_class_name = "line"
    new_category_id = 0

    # Build old ID -> name map from original JSON
    old_id_to_name = {cat["id"]: cat["name"] for cat in data["categories"]}

    print(f"\nOriginal categories in {json_path}:")
    for k, v in old_id_to_name.items():
        print(f"  id: {k}, name: {v}")

    # Find the original ID of "line" class
    line_class_id = None
    for cat in data["categories"]:
        if cat["name"] == target_class_name:
            line_class_id = cat["id"]
            break

    if line_class_id is None:
        raise ValueError(f"❌ Couldn't find '{target_class_name}' class in the dataset.")

    print(f"✅ Found '{target_class_name}' with original id={line_class_id}, remapping to id=0")

    # Filter annotations to only include those with the "line" category
    filtered_annotations = []
    for ann in data["annotations"]:
        if ann["category_id"] == line_class_id:
            ann["category_id"] = new_category_id  # Remap to 0
            filtered_annotations.append(ann)

    print(f"📉 Reduced annotations from {len(data['annotations'])} to {len(filtered_annotations)}")

    # Filter images to only keep those that still have valid annotations
    valid_image_ids = set(ann["image_id"] for ann in filtered_annotations)
    filtered_images = [img for img in data["images"] if img["id"] in valid_image_ids]

    print(f"🖼️ Reduced images from {len(data['images'])} to {len(filtered_images)}")

    # Replace categories, annotations, and images
    data["categories"] = [{
        "id": new_category_id,
        "name": target_class_name,
        "supercategory": "objects"
    }]
    data["annotations"] = filtered_annotations
    data["images"] = filtered_images

    # Save the cleaned dataset
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"✅ Filtered annotation file saved to: {output_path}")


# Run on both train and val
filter_to_line_only(
    "../datasets/300_2/annotations/instances_train.json",
    "../datasets/300_2/annotations/instances_train_line_only.json"
)

filter_to_line_only(
    "../datasets/300_2/annotations/instances_val.json",
    "../datasets/300_2/annotations/instances_val_line_only.json"
)