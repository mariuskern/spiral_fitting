from utils import run_inference


config = {
    "model1_folder": "../checkpoints/surface_detection/v3",
    "model1_checkpoint": "checkpoint_final.pth",
    "model1_device": "cuda",

    "model2_folder": "../checkpoints/surface_detection/v2",
    "model2_checkpoint": "checkpoint_final.pth",
    "model2_device": "cuda",

    "input_dir": "../data/surface_detection/test_images",
    "output_zip": "submission.zip"
}


run_inference(config)
