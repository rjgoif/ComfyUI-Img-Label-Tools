"""
Label processing nodes for ComfyUI
"""

import torch
import math

MAX_RESOLUTION = 16384


class ImageEqualizer:
    """
    Equalizes image sizes in a batch through padding and/or scaling.
    Inspired by KJNodes for ComfyUI by github user kijai.
    """
    
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "size_mode": (["grow", "shrink"], {"default": "grow"}),
                "upscale_method": (cls.upscale_methods, {"default": "bicubic"}),
                "keep_proportion": (["pad", "stretch", "resize", "crop", "total_pixels"], {"default": "pad"}),
                "pad_color": (["black", "white", "gray", "average", "average_edge"], {"default": "black"}),
                "crop_position": (["center", "top", "bottom", "left", "right"], {"default": "center"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "equalize"
    CATEGORY = "Image Label Tools"
    DESCRIPTION = """
Resizes images to match the largest or smallest image among them.

size_mode determines target dimensions:
- grow: all images match the largest dimensions
- shrink: all images match the smallest dimensions

keep_proportion maintains aspect ratio by highest dimension:
- pad: adds padding to fit target size (default)
- stretch: directly resizes to target
- resize: scales to fit within target
- crop: crops to fill target
- total_pixels: maintains total pixel count

pad_color options:
- black: RGB(0,0,0)
- white: RGB(255,255,255)  
- gray: RGB(128,128,128)
- average: gamma-corrected weighted mean color of entire image
- average_edge: weighted mean of peripheral 5% of pixels
"""

    def equalize(self, images, size_mode, upscale_method, keep_proportion, pad_color, crop_position):
        from comfy.utils import common_upscale
        
        # When INPUT_IS_LIST=True, all parameters come as lists - extract the first value
        size_mode = size_mode[0] if isinstance(size_mode, list) else size_mode
        upscale_method = upscale_method[0] if isinstance(upscale_method, list) else upscale_method
        keep_proportion = keep_proportion[0] if isinstance(keep_proportion, list) else keep_proportion
        pad_color = pad_color[0] if isinstance(pad_color, list) else pad_color
        crop_position = crop_position[0] if isinstance(crop_position, list) else crop_position
        
        device = torch.device("cpu")
        
        # Collect all individual images
        all_images = []
        if isinstance(images, list):
            for batch in images:
                for i in range(batch.shape[0]):
                    all_images.append(batch[i:i+1])
        else:
            for i in range(images.shape[0]):
                all_images.append(images[i:i+1])
        
        num_images = len(all_images)
        
        # Find target dimensions
        target_height = all_images[0].shape[1]
        target_width = all_images[0].shape[2]
        
        for img in all_images:
            h, w = img.shape[1], img.shape[2]
            if size_mode == "grow":
                target_height = max(target_height, h)
                target_width = max(target_width, w)
            else:  # shrink
                target_height = min(target_height, h)
                target_width = min(target_width, w)
        
        print(f"Image Equalizer: {num_images} images | {size_mode} to {target_width}x{target_height} | method: {keep_proportion}")
        
        # Process each image
        processed = []
        
        for idx, img in enumerate(all_images):
            img_h, img_w = img.shape[1], img.shape[2]
            
            # Skip if already correct size
            if img_w == target_width and img_h == target_height:
                processed.append(img.cpu())
                continue
            
            out_image = img.to(device)
            
            if keep_proportion == "stretch":
                # Direct resize to target
                out_image = common_upscale(out_image.movedim(-1, 1), target_width, target_height, upscale_method, crop="disabled").movedim(1, -1)
            
            elif keep_proportion == "crop":
                # Crop to aspect ratio then resize
                target_aspect = target_width / target_height
                img_aspect = img_w / img_h
                
                if img_aspect > target_aspect:
                    crop_w = int(img_h * target_aspect)
                    crop_h = img_h
                else:
                    crop_w = img_w
                    crop_h = int(img_w / target_aspect)
                
                x = (img_w - crop_w) // 2
                y = (img_h - crop_h) // 2
                
                if crop_position == "top":
                    y = 0
                elif crop_position == "bottom":
                    y = img_h - crop_h
                elif crop_position == "left":
                    x = 0
                elif crop_position == "right":
                    x = img_w - crop_w
                
                out_image = out_image[:, y:y+crop_h, x:x+crop_w, :]
                out_image = common_upscale(out_image.movedim(-1, 1), target_width, target_height, upscale_method, crop="disabled").movedim(1, -1)
            
            else:  # pad, resize, pillarbox_blur, total_pixels
                # Calculate scaled size
                if keep_proportion == "total_pixels":
                    total_pixels = target_width * target_height
                    aspect = img_w / img_h
                    scaled_h = int(math.sqrt(total_pixels / aspect))
                    scaled_w = int(math.sqrt(total_pixels * aspect))
                else:
                    ratio = min(target_width / img_w, target_height / img_h)
                    scaled_w = int(img_w * ratio)
                    scaled_h = int(img_h * ratio)
                
                # Resize to scaled size
                out_image = common_upscale(out_image.movedim(-1, 1), scaled_w, scaled_h, upscale_method, crop="disabled").movedim(1, -1)
                
                # Pad if needed
                if keep_proportion == "pad" and (scaled_w != target_width or scaled_h != target_height):
                    pad_w = target_width - scaled_w
                    pad_h = target_height - scaled_h
                    
                    pad_left = pad_w // 2
                    pad_right = pad_w - pad_left
                    pad_top = pad_h // 2
                    pad_bottom = pad_h - pad_top
                    
                    if crop_position == "top":
                        pad_bottom += pad_top
                        pad_top = 0
                    elif crop_position == "bottom":
                        pad_top += pad_bottom
                        pad_bottom = 0
                    elif crop_position == "left":
                        pad_right += pad_left
                        pad_left = 0
                    elif crop_position == "right":
                        pad_left += pad_right
                        pad_right = 0
                    
                    # Get pad color
                    if pad_color == "black":
                        color_val = "0, 0, 0"
                    elif pad_color == "white":
                        color_val = "255, 255, 255"
                    elif pad_color == "gray":
                        color_val = "128, 128, 128"
                    elif pad_color == "average":
                        avg = out_image.pow(2.2).mean(dim=[0, 1, 2]).pow(1/2.2)
                        color_val = f"{int(avg[0]*255)}, {int(avg[1]*255)}, {int(avg[2]*255)}"
                    elif pad_color == "average_edge":
                        edge_h = max(1, int(scaled_h * 0.05))
                        edge_w = max(1, int(scaled_w * 0.05))
                        # Get edge pixels and reshape them to combine
                        top = out_image[:, :edge_h, :, :].reshape(-1, 3)
                        bottom = out_image[:, -edge_h:, :, :].reshape(-1, 3)
                        left = out_image[:, :, :edge_w, :].reshape(-1, 3)
                        right = out_image[:, :, -edge_w:, :].reshape(-1, 3)
                        all_edges = torch.cat([top, bottom, left, right], dim=0)
                        avg = all_edges.pow(2.2).mean(dim=0).pow(1/2.2)
                        color_val = f"{int(avg[0]*255)}, {int(avg[1]*255)}, {int(avg[2]*255)}"
                    
                    out_image = self._apply_padding(out_image, pad_left, pad_right, pad_top, pad_bottom, color_val, "color")
            
            processed.append(out_image.cpu())
        
        # When INPUT_IS_LIST=True, always return a list
        return (processed,)
    
    def _apply_padding(self, image, pad_left, pad_right, pad_top, pad_bottom, color_value, pad_mode):
        """Apply padding to image"""
        B, H, W, C = image.shape
        
        # Parse color value
        rgb = [int(x.strip()) / 255.0 for x in color_value.split(',')]
        
        # Create padded image
        new_h = H + pad_top + pad_bottom
        new_w = W + pad_left + pad_right
        padded = torch.zeros((B, new_h, new_w, C), device=image.device)
        
        # Fill with color
        for c in range(C):
            padded[:, :, :, c] = rgb[c]
        
        # Place original image
        padded[:, pad_top:pad_top+H, pad_left:pad_left+W, :] = image
        
        return padded


NODE_CLASS_MAPPINGS = {
    "ImageEqualizer": ImageEqualizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageEqualizer": "Image Equalizer",
}
