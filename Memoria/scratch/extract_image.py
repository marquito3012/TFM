import json
import base64

notebook_path = "/mnt/AI/TFM/compartida/notebooks/02_limpieza_ingenieria.ipynb"
output_image_path = "/mnt/AI/TFM/Memoria/images/class_distribution.png"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find the cell that has class distribution plot
found = False
for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        outputs = cell.get("outputs", [])
        for out in outputs:
            data = out.get("data", {})
            if "image/png" in data:
                png_base64 = data["image/png"]
                # Clean up newlines in base64 string if any
                png_base64 = png_base64.replace("\n", "").replace("\r", "")
                image_data = base64.b64decode(png_base64)
                with open(output_image_path, "wb") as img_f:
                    img_f.write(image_data)
                print(f"Successfully extracted image to {output_image_path}")
                found = True
                break
        if found:
            break

if not found:
    print("Could not find image/png in any cell output.")
