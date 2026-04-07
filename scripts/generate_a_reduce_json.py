import json

def reduce_json():

    json_ = json.load(open("data/json/extend/2022_BCN_FinalM_1_ball.json"))
    json_copy = json_.copy()

    json_copy["images"] = json_copy["images"][:1500] 

    ids_validos = set(img["id"] for img in json_copy["images"])

    json_copy["annotations"] = [ann for ann in json_copy["annotations"] if ann["image_id"] in ids_validos]

    with open("data/json/reduced/2022_BCN_FinalM_1_ball_reduced.json", "w") as f:
        json.dump(json_copy, f, indent=4)

def main():
    reduce_json()

if __name__ == "__main__":
    main()