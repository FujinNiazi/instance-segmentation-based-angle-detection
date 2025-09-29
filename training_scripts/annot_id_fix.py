import json

def fix_category_ids(json_path, output_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Define desired mapping
    category_name_to_new_id = {
        "robot": 1,
        "camera": 2,
        "line": 3
    }

    # Build old ID -> name map from JSON
    old_id_to_name = {cat["id"]: cat["name"] for cat in data["categories"]}

    print(f"\nCategories found in {json_path}:")
    for k, v in old_id_to_name.items():
        print(f"  id: {k}, name: {v}")

    # Create new categories list
    data["categories"] = [
        {"id": new_id, "name": name, "supercategory": "objects"}
        for name, new_id in category_name_to_new_id.items()
    ]

    # Map old category IDs to new ones
    old_id_to_new_id = {}
    for old_id, name in old_id_to_name.items():
        if name in category_name_to_new_id:
            old_id_to_new_id[old_id] = category_name_to_new_id[name]
        else:
            print(f"⚠️ Skipping unused category '{name}' (id={old_id})")

    # Fix all annotation category_ids
    for ann in data["annotations"]:
        old_cat_id = ann["category_id"]
        if old_cat_id not in old_id_to_new_id:
            raise ValueError(f"Annotation uses unknown category_id={old_cat_id}")
        ann["category_id"] = old_id_to_new_id[old_cat_id]

    # Save fixed file
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"✅ Fixed annotation file saved to: {output_path}")


# Run on both train and val
fix_category_ids(
    "../datasets/600_3/annotations/instances_train.json",
    "../datasets/600_3/annotations/instances_train_fixed.json"
)

fix_category_ids(
    "../datasets/600_3/annotations/instances_val.json",
    "../datasets/600_3/annotations/instances_val_fixed.json"
)
