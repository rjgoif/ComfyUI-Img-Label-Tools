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
                "keep_proportion": (["pad", "stretch", "resize", "crop", "pillarbox_blur", "total_pixels"], {"default": "pad"}),
                "pad_color": (["black", "white", "gray", "average", "average_edge"], {"default": "black"}),
                "crop_position": (["center", "top", "bottom", "left", "right"], {"default": "center"}),
                "output": (["batch", "list"], {"default": "batch"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
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
- pillarbox_blur: pads with blurred edges
- total_pixels: maintains total pixel count

pad_color options:
- black: RGB(0,0,0)
- white: RGB(255,255,255)  
- gray: RGB(128,128,128)
- average: gamma-corrected weighted mean color of entire image
- average_edge: weighted mean of peripheral 5% of pixels
"""

    def equalize(self, images, size_mode, upscale_method, keep_proportion, pad_color, crop_position, output):
        from comfy.utils import common_upscale
        
        # Convert list to batch if needed
        if isinstance(images, list):
            images = torch.cat(images, dim=0)
        
        B, H, W, C = images.shape
        device = torch.device("cpu")
        
        # Determine target dimensions based on size_mode
        if size_mode == "grow":
            target_width = W
            target_height = H
            for i in range(B):
                target_width = max(target_width, images[i].shape[1])
                target_height = max(target_height, images[i].shape[0])
        else:  # shrink
            target_width = W
            target_height = H
            for i in range(B):
                target_width = min(target_width, images[i].shape[1])
                target_height = min(target_height, images[i].shape[0])
        
        processed_images = []
        
        for i in range(B):
            img = images[i:i+1]
            img_h, img_w = img.shape[1], img.shape[2]
            
            # Skip if already correct size
            if img_w == target_width and img_h == target_height:
                processed_images.append(img)
                continue
            
            width = target_width
            height = target_height
            pillarbox_blur = keep_proportion == "pillarbox_blur"
            
            # Initialize padding variables
            pad_left = pad_right = pad_top = pad_bottom = 0
            
            if keep_proportion in ["resize", "total_pixels", "pad", "pillarbox_blur"]:
                if keep_proportion == "total_pixels":
                    total_pixels = width * height
                    aspect_ratio = img_w / img_h
                    new_height = int(math.sqrt(total_pixels / aspect_ratio))
                    new_width = int(math.sqrt(total_pixels * aspect_ratio))
                else:
                    ratio = min(width / img_w, height / img_h)
                    new_width = round(img_w * ratio)
                    new_height = round(img_h * ratio)
                
                if keep_proportion in ["pad", "pillarbox_blur"]:
                    # Calculate padding based on position
                    if crop_position == "center":
                        pad_left = (width - new_width) // 2
                        pad_right = width - new_width - pad_left
                        pad_top = (height - new_height) // 2
                        pad_bottom = height - new_height - pad_top
                    elif crop_position == "top":
                        pad_left = (width - new_width) // 2
                        pad_right = width - new_width - pad_left
                        pad_top = 0
                        pad_bottom = height - new_height
                    elif crop_position == "bottom":
                        pad_left = (width - new_width) // 2
                        pad_right = width - new_width - pad_left
                        pad_top = height - new_height
                        pad_bottom = 0
                    elif crop_position == "left":
                        pad_left = 0
                        pad_right = width - new_width
                        pad_top = (height - new_height) // 2
                        pad_bottom = height - new_height - pad_top
                    elif crop_position == "right":
                        pad_left = width - new_width
                        pad_right = 0
                        pad_top = (height - new_height) // 2
                        pad_bottom = height - new_height - pad_top
                
                width = new_width
                height = new_height
            
            # Move to device
            out_image = img.to(device)
            
            # Crop logic
            if keep_proportion == "crop":
                old_height = out_image.shape[1]
                old_width = out_image.shape[2]
                old_aspect = old_width / old_height
                new_aspect = target_width / target_height
                
                if old_aspect > new_aspect:
                    crop_w = round(old_height * new_aspect)
                    crop_h = old_height
                else:
                    crop_w = old_width
                    crop_h = round(old_width / new_aspect)
                
                if crop_position == "center":
                    x = (old_width - crop_w) // 2
                    y = (old_height - crop_h) // 2
                elif crop_position == "top":
                    x = (old_width - crop_w) // 2
                    y = 0
                elif crop_position == "bottom":
                    x = (old_width - crop_w) // 2
                    y = old_height - crop_h
                elif crop_position == "left":
                    x = 0
                    y = (old_height - crop_h) // 2
                elif crop_position == "right":
                    x = old_width - crop_w
                    y = (old_height - crop_h) // 2
                
                out_image = out_image[:, y:y+crop_h, x:x+crop_w, :]
            
            # Resize
            out_image = common_upscale(out_image.movedim(-1, 1), width, height, upscale_method, crop="disabled").movedim(1, -1)
            
            # Padding logic
            if (keep_proportion in ["pad", "pillarbox_blur"]) and (pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0):
                # Calculate pad color
                if pad_color == "black":
                    color_value = "0, 0, 0"
                elif pad_color == "white":
                    color_value = "255, 255, 255"
                elif pad_color == "gray":
                    color_value = "128, 128, 128"
                elif pad_color == "average":
                    # Gamma-corrected average of entire image
                    avg = out_image.pow(2.2).mean(dim=[0, 1, 2]).pow(1/2.2)
                    color_value = f"{int(avg[0]*255)}, {int(avg[1]*255)}, {int(avg[2]*255)}"
                elif pad_color == "average_edge":
                    # Average of peripheral 5% of pixels
                    edge_h = max(1, int(height * 0.05))
                    edge_w = max(1, int(width * 0.05))
                    
                    top_edge = out_image[:, :edge_h, :, :]
                    bottom_edge = out_image[:, -edge_h:, :, :]
                    left_edge = out_image[:, :, :edge_w, :]
                    right_edge = out_image[:, :, -edge_w:, :]
                    
                    edge_pixels = torch.cat([top_edge, bottom_edge, left_edge, right_edge], dim=1)
                    avg = edge_pixels.pow(2.2).mean(dim=[0, 1, 2]).pow(1/2.2)
                    color_value = f"{int(avg[0]*255)}, {int(avg[1]*255)}, {int(avg[2]*255)}"
                
                pad_mode = "pillarbox_blur" if pillarbox_blur else "color"
                
                # Apply padding
                out_image = self._apply_padding(out_image, pad_left, pad_right, pad_top, pad_bottom, color_value, pad_mode)
            
            processed_images.append(out_image.cpu())
        
        # Combine results
        result = torch.cat(processed_images, dim=0)
        
        # Convert to list if requested
        if output == "list":
            result = [result[i:i+1] for i in range(result.shape[0])]
        
        return (result,)
    
    def _apply_padding(self, image, pad_left, pad_right, pad_top, pad_bottom, color_value, pad_mode):
        """Apply padding to image"""
        B, H, W, C = image.shape
        
        if pad_mode == "color":
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
        
        elif pad_mode == "pillarbox_blur":
            from comfy.utils import common_upscale
            
            # Create blurred background
            new_h = H + pad_top + pad_bottom
            new_w = W + pad_left + pad_right
            
            # Downscale then upscale for blur effect
            blur_scale = 0.1
            blur_h = max(1, int(H * blur_scale))
            blur_w = max(1, int(W * blur_scale))
            
            blurred = common_upscale(image.movedim(-1, 1), blur_w, blur_h, "bilinear", crop="disabled")
            blurred = common_upscale(blurred, new_w, new_h, "bilinear", crop="disabled").movedim(1, -1)
            
            # Place sharp image on top
            blurred[:, pad_top:pad_top+H, pad_left:pad_left+W, :] = image
            
            return blurred
        
        return image


NODE_CLASS_MAPPINGS = {
    "ImageEqualizer": ImageEqualizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageEqualizer": "Image Equalizer",
}
