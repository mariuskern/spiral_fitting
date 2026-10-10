import os

from dotenv import load_dotenv

from utils import run_inference

load_dotenv()


# config = {
#     "model1_folder": "../checkpoints/surface_detection/v3",
#     "model1_checkpoint": "checkpoint_final.pth",
#     "model1_device": "cuda",

#     "model2_folder": "../checkpoints/surface_detection/v2",
#     "model2_checkpoint": "checkpoint_final.pth",
#     "model2_device": "cuda",

#     "input_dir": "../data/332/volume",
#     "output_zip": "submission.zip"
# }


config = {
    "model1_folder": os.getenv("MODEL1_FOLDER", "../checkpoints/surface_detection/v3"),
    "model1_checkpoint": os.getenv("MODEL1_CHECKPOINT", "checkpoint_final.pth"),
    "model1_device": os.getenv("MODEL1_DEVICE", "cuda"),

    "model2_folder": os.getenv("MODEL2_FOLDER", "../checkpoints/surface_detection/v2"),
    "model2_checkpoint": os.getenv("MODEL2_CHECKPOINT", "checkpoint_final.pth"),
    "model2_device": os.getenv("MODEL2_DEVICE", "cuda"),

    "input_dir": os.getenv("INPUT_DIR"),
    "output_zip": os.getenv("OUTPUT_ZIP"),
}


run_inference(config)
