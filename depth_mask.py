import cv2
import os
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoImageProcessor, DPTForDepthEstimation, SegformerFeatureExtractor, SegformerForSemanticSegmentation
import argparse
from itertools import tee

from tqdm import tqdm

def check_outputdir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir



def get_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    for _ in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))), desc=''):
        # Read frame by frame
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
        ret, frame = cap.read() # Read every other frame
        if not ret:
            break
    cap.release()


# Open the video

def resize_to_fit(image, target_width, target_height):
    target_ratio = target_width / target_height
    image_ratio = image.shape[1] / image.shape[0]

    # Determine which dimension (width or height) is the limiting factor
    if image_ratio > target_ratio:
        # Width is the limiting factor, so scale by width
        new_width = target_width
        scale_factor = new_width / image.shape[1]
        new_height = int(image.shape[0] * scale_factor)
    else:
        # Height is the limiting factor, so scale by height
        new_height = target_height
        scale_factor = new_height / image.shape[0]
        new_width = int(image.shape[1] * scale_factor)

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return resized_image

def get_real_size(frames, target_width, target_height):
    img0 = frames[0]
    img1 = resize_to_fit(img0, target_width, target_height)
    width = img1.shape[1]
    height = img1.shape[0]
    return width, height


def resize_frames(frames_iterator, target_width, target_height):
    for frame in frames_iterator:
        yield resize_to_fit(frame, target_width, target_height)


def get_models(model_name, device):
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    model = DPTForDepthEstimation.from_pretrained(model_name)
    model = model.to(device)
    return image_processor, model


def process_batch(images,image_processor, model, device):
    width = images[0].shape[1]
    height = images[0].shape[0]

    inputs = image_processor(images=images, return_tensors="pt", padding=True)
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
    # Perform depth estimation
    with torch.no_grad():
        outputs = model(**inputs)
        if(model.config.model_type == "dpt"):
            predicted_depth = outputs.predicted_depth.to('cpu')
        else:
            predicted_depth = outputs.to('cpu')
        
        # Resize predicted depth maps if necessary
        depth_resized = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(0),  # Add batch and channel dimension
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        )
    return depth_resized[0]


def process_batches(frames_iterator,image_processor, model, device, batch_size):
    global b
    imgs = []
    while True:
        try:
            elem = next(frames_iterator)
            imgs.append(elem)
            if(len(imgs) == batch_size):
                for depth in process_batch(imgs,image_processor, model, device):
                    yield depth
                imgs = []
        except StopIteration:
            for depth in process_batch(imgs,image_processor, model, device):
                yield depth
            break


def normalize_depth(depth_iterator):
    for depthtensor in depth_iterator:
        depth =  depthtensor.numpy()
        min_val = np.min(depth)
        max_val = np.max(depth)
        normalized_depth = (depth - min_val) / (max_val - min_val)
        yield (normalized_depth * 255).astype(np.uint8)



def to_grayscale(frames_iterator):
    for frame in frames_iterator:
        yield cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def zipdict(dict_of_iterables):
    for list_of_values in zip(*dict_of_iterables.values()):
        # Since info_tuple now contains the values directly, we need to align these with keys manually
        yield dict(zip(dict_of_iterables.keys(), list_of_values))

def save_all(info_iterable, output_dir):
    for i, info in enumerate(zipdict(info_iterable)):
        # Save the frames to the output directory
        for key, frame in info.items():
            # Resize the frame to 1120x832, adding black bars if necessary
            resized_frame = np.zeros((832, 1120), dtype=np.uint8)
            start_x = (1120 - frame.shape[1]) // 2
            start_y = (832 - frame.shape[0]) // 2
            resized_frame[start_y:start_y+frame.shape[0], start_x:start_x+frame.shape[1]] = frame
            cv2.imwrite(f"{output_dir}/{i:04d}_{key}.png", resized_frame)


def parse_target_size(value):
    try:
        # Strip parentheses and split by comma
        x, y = value.strip("()").split(',')
        return int(x), int(y)
    except ValueError:
        raise argparse.ArgumentTypeError("Target size must be (x,y)")


def parse_model_name(value):
    if value == "fast":
        return "Intel/dpt-swinv2-tiny-256"
    elif value == "accurate":
        return "Intel/dpt-hybrid-midas"
    else:
        return value

class CustomSegformerModel(nn.Module):
    def __init__(self, pretrained_model_name):
        super(CustomSegformerModel, self).__init__()
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(pretrained_model_name)
        self.config = self.segformer.config
    def forward(self, pixel_values):
        outputs = self.segformer(pixel_values=pixel_values)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        # Get the mask for the 12th feature, assuming the 12th index exists
        mask = probabilities[:, 12, :, :]  # This slices the tensor to dim [batch_size, height, width]
        return mask

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description='Example script to demonstrate argparse usage.')
    
    # Add arguments
    parser.add_argument('input', help='Input file')
    parser.add_argument('output', help='Output file', default='output')
    parser.add_argument('--targetsize', help='Specify target screen size as --targetsize=(x,y)', type=parse_target_size, default=(1120, 832))
    
    parser.add_argument('--model', help="either 'fast', 'accurate', or a DPT model name (ex: 'Intel/dpt-large')", type=parse_model_name, default="fast")    


    # Parse arguments
    args = parser.parse_args()
    
    # Access arguments
    # print(f"Input file: {args.input}")
    # print(f"Output file: {args.output}")
    

    video_path = args.input
    output_dir = args.output
    target_width, target_height = args.targetsize
    model_name = args.model

    check_outputdir(output_dir)

    device = "cuda"
    image_processor, model = get_models(model_name, device)
    # segment_processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")
    # segment_model = CustomSegformerModel("nvidia/segformer-b2-finetuned-ade-512-512")
    # segment_model = segment_model.to("cpu")

    batch_size = 20

    # get size after resizing
    frames_iterator = get_frames(video_path)
    frames_resized = resize_frames(frames_iterator, target_width, target_height)

    to_depth, to_segmentation, to_gray = tee(frames_resized, 3)

    predicted_depth = process_batches(to_depth, image_processor, model, device, batch_size)
    norm_depth = normalize_depth(predicted_depth)
    # predicted_segmentation = process_batches(to_segmentation, segment_processor, segment_model, "cpu", batch_size)
    # norm_segmentation = normalize_depth(predicted_segmentation)
    frames_gray = to_grayscale(to_gray)
    info = {'depth': norm_depth, 'frame': frames_gray}
    save_all(info, output_dir)



    # Your program logic here

if __name__ == '__main__':
    main()
