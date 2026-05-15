import os
# las siguientes clases son audios ya fragmentados,es decir segmentados
CLASSES = ['Asma', 'Epoc', 'Neumonia', 'Normal']

# se pasará base_path = "..\\data\\raw"
def ingest_data(base_path):
    data = []

    for label, clase in enumerate(CLASSES):
        class_path = os.path.join(base_path, clase)

        for file in os.listdir(class_path):
            if file.endswith(".wav"):
                full_path = os.path.join(class_path, file)

                data.append({
                    "path": full_path,
                    "label": label,
                    "class_name": clase
                })

    print(f"✅ Total audios cargados: {len(data)}")
    return data

