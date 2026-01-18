"""
ComfyUI Img Label Tools
Custom nodes for image processing and labeling in ComfyUI

Credits:
- Image Equalizer inspired by KJNodes for ComfyUI by github user kijai
- Image Array label application logic inspired by Mikey Nodes by github user bash-j
"""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import folder_paths
import math
from comfy.utils import common_upscale

MAX_RESOLUTION = 16384

"""
Label processing nodes for ComfyUI
"""

import torch
import math
import random

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
                "upscale_method": (cls.upscale_methods, {"default": "lanczos"}),
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
                        # Reshape to [pixels, 3] and take mean across pixels
                        avg = out_image.pow(2.2).reshape(-1, 3).mean(dim=0).pow(1/2.2)
                        # Handle NaN and clip to valid range
                        avg = torch.nan_to_num(avg, nan=0.5)
                        avg = torch.clamp(avg, 0.0, 1.0)
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




class ImageArray:
    """Creates labeled image arrays in various layouts"""
    INPUT_IS_LIST = True
    
    @classmethod
    def INPUT_TYPES(cls):
        # Check for fonts directory
        if os.path.exists(os.path.join(folder_paths.base_path, 'fonts')):
            cls.font_dir = os.path.join(folder_paths.base_path, 'fonts')
            cls.font_files = [f for f in os.listdir(cls.font_dir) if os.path.isfile(os.path.join(cls.font_dir, f))]
            font_default = cls.font_files[0] if cls.font_files else 'arial.ttf'
        else:
            cls.font_dir = None
            cls.font_files = ['arial.ttf']
            font_default = 'arial.ttf'
        
        return {
            'required': {
                'images': ('IMAGE',),
                'background': (['white', 'black'], {'default': 'white'}),
                'resize': (['grow', 'shrink'], {'default': 'grow'}),
                'size_method': (['pad', 'stretch', 'crop_center', 'fill'], {'default': 'pad'}),
                'pad': ('BOOLEAN', {'default': True}),
                'shape': (['horizontal', 'vertical', 'square', 'smart_square', 'smart_landscape', 'smart_portrait'], {'default': 'square'}),
                'labels': ('STRING', {'multiline': True, 'default': ''}),
                'label_end': (['loop', 'end'], {'default': 'loop'}),
                'label_location': (['top', 'bottom', 'left_vert', 'left_hor', 'right_vert', 'right_hor'], {'default': 'bottom'}),
                'label_size': ('INT', {'default': 32, 'min': 0, 'max': 200, 'step': 1}),
                'font': (cls.font_files, {'default': font_default}),
                'spacing': ('INT', {'default': 5, 'min': 0, 'max': 100, 'step': 1}),
            },
            'optional': {
                'label_input': ('STRING', {'forceInput': True}),
            }
        }
    
    RETURN_TYPES = ('IMAGE', 'IMAGE')
    RETURN_NAMES = ('array', 'images')
    OUTPUT_IS_LIST = (False, True)
    FUNCTION = 'create_array'
    CATEGORY = 'Image Label Tools'
    DESCRIPTION = "Creates an array of images with optional labels in various layouts"
    
    def parse_labels(self, labels_text, label_input=None):
        """Parse labels from either input or text widget"""
        if label_input:
            # Handle list or single input
            if isinstance(label_input, list):
                labels = []
                for item in label_input:
                    if isinstance(item, (int, float)):
                        labels.append(self._format_number(item))
                    else:
                        labels.append(str(item))
                return labels
            else:
                if isinstance(label_input, (int, float)):
                    return [self._format_number(label_input)]
                return [str(label_input)]
        
        # Parse from text widget
        if not labels_text.strip():
            return []
        
        # Split by actual newlines (not \n strings)
        # Replace literal \n with a placeholder first
        labels_text = labels_text.replace('\\n', '\x00')  # Use null char as placeholder
        
        labels = []
        lines = labels_text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Check if line contains semicolons
            if ';' in line:
                for label in line.replace('; ', ';').split(';'):
                    label = label.strip()
                    if label:
                        # Restore \n as actual newlines within the label
                        label = label.replace('\x00', '\n')
                        labels.append(label)
            else:
                # Restore \n as actual newlines within the label
                label = line.replace('\x00', '\n')
                labels.append(label)
        return labels
    
    def _format_number(self, num):
        """Format number, truncating decimals intelligently"""
        if isinstance(num, int):
            return str(num)
        
        # Convert to float
        num = float(num)
        
        # If integer value, return as int
        if num == int(num):
            return str(int(num))
        
        # Find significant decimal places (up to 5)
        str_num = f"{num:.5f}".rstrip('0')
        return str_num
    
    def get_text_size(self, font, text):
        """Get width and height of text"""
        left, top, right, bottom = font.getbbox(text)
        width = right - left
        height = bottom - top
        return width, height
    
    def calculate_label_dimensions(self, label_text, location, label_size, font_path, img_width, img_height):
        """Calculate label dimensions without creating the actual label image"""
        # Load font
        try:
            if self.font_dir:
                font_file = os.path.join(self.font_dir, font_path)
            else:
                font_file = 'C:/Windows/Fonts/Arial.ttf'
            font = ImageFont.truetype(font_file, label_size)
        except:
            font = ImageFont.load_default()
        
        is_vertical = location in ['left_vert', 'right_vert']
        is_left_right_hor = location in ['left_hor', 'right_hor']
        
        # Calculate dimensions based on location
        if is_vertical:
            max_width = img_height
            wrapped_lines = self.wrap_text(label_text, font, max_width) if label_text else ['']
            _, line_height = self.get_text_size(font, "Hg")
            label_width = max(1, len(wrapped_lines)) * line_height + 30
            label_height = img_height
            
        elif is_left_right_hor:
            max_width = img_width // 2
            wrapped_lines = self.wrap_text(label_text, font, max_width) if label_text else ['']
            _, line_height = self.get_text_size(font, "Hg")
            
            if label_text and wrapped_lines:
                max_line_width = max(int(font.getlength(line)) for line in wrapped_lines if line)
            else:
                max_line_width = 0
            label_width = max_line_width + 30
            label_height = max(1, len(wrapped_lines)) * (line_height + 5) + 30
            
        else:  # top or bottom
            max_width = img_width
            wrapped_lines = self.wrap_text(label_text, font, max_width) if label_text else ['']
            _, line_height = self.get_text_size(font, "Hg")
            label_width = img_width
            label_height = max(1, len(wrapped_lines)) * (line_height + 5) + 30
        
        return label_width, label_height
    
    def wrap_text(self, text, font, max_width):
        """Wrap text to fit width"""
        wrapped_lines = []
        for line in text.split('\n'):
            words = line.split(' ')
            if not words:
                wrapped_lines.append('')
                continue
            
            new_line = words[0]
            for word in words[1:]:
                if int(font.getlength(new_line + ' ' + word)) <= max_width:
                    new_line += ' ' + word
                else:
                    wrapped_lines.append(new_line)
                    new_line = word
            wrapped_lines.append(new_line)
        return wrapped_lines
    
    def add_label_to_image(self, image_pil, label_text, location, label_size, font_path, bg_color, text_color, fixed_label_width=None, fixed_label_height=None):
        """Add label to a PIL image"""
        # Always add label padding, even if text is empty
        width, height = image_pil.size
        
        # Load font
        try:
            if self.font_dir:
                font_file = os.path.join(self.font_dir, font_path)
            else:
                font_file = 'C:/Windows/Fonts/Arial.ttf'
            font = ImageFont.truetype(font_file, label_size)
        except:
            font = ImageFont.load_default()
        
        # Determine if vertical or horizontal
        is_vertical = location in ['left_vert', 'right_vert']
        is_left_right_hor = location in ['left_hor', 'right_hor']
        
        # Calculate label dimensions
        # Get line height for text drawing (needed in all cases)
        _, line_height = self.get_text_size(font, "Hg")
        
        if fixed_label_width and fixed_label_height:
            # Use provided fixed dimensions
            label_width = fixed_label_width
            label_height = fixed_label_height
            # Still need to wrap text for drawing
            if is_vertical:
                max_width = height
            elif is_left_right_hor:
                max_width = width // 2
            else:
                max_width = width
            wrapped_lines = self.wrap_text(label_text, font, max_width) if label_text else ['']
        elif is_vertical:
            # For vertical text, we'll rotate it
            max_width = height
            wrapped_lines = self.wrap_text(label_text, font, max_width) if label_text else ['']
            
            # Create vertical label
            label_width = max(1, len(wrapped_lines)) * line_height + 30
            label_height = height
            
        elif is_left_right_hor:
            # Horizontal text on left/right side
            max_width = width // 2  # Max half the image width
            wrapped_lines = self.wrap_text(label_text, font, max_width) if label_text else ['']
            
            # Calculate actual width needed
            if label_text and wrapped_lines:
                max_line_width = max(int(font.getlength(line)) for line in wrapped_lines if line)
            else:
                max_line_width = 0
            label_width = max_line_width + 30
            label_height = max(1, len(wrapped_lines)) * (line_height + 5) + 30
            
        else:
            # Top or bottom
            max_width = width
            wrapped_lines = self.wrap_text(label_text, font, max_width) if label_text else ['']
            
            label_width = width
            label_height = max(1, len(wrapped_lines)) * (line_height + 5) + 30
        
        # Create label image
        label_img = Image.new('RGB', (label_width, label_height), bg_color)
        draw = ImageDraw.Draw(label_img)
        
        # Draw text
        if is_vertical:
            # Draw text horizontally first, then rotate
            temp_width = label_height
            temp_height = label_width
            temp_img = Image.new('RGB', (temp_width, temp_height), bg_color)
            temp_draw = ImageDraw.Draw(temp_img)
            
            # For vertical text, align to bottom (closest to image)
            # Calculate total text height
            total_text_height = sum(line_height + 5 for _ in wrapped_lines) - 5
            y_pos = temp_height - total_text_height - 15  # Start from bottom minus padding
            
            for line in wrapped_lines:
                text_width = int(font.getlength(line))
                x_pos = (temp_width - text_width) // 2  # Horizontal center
                temp_draw.text((x_pos, y_pos), line, text_color, font=font)
                y_pos += line_height + 5
            
            # Rotate based on side
            if location == 'left_vert':
                # 90 degrees counterclockwise - bottom of text faces right (toward image)
                label_img = temp_img.rotate(90, expand=True)
            else:  # right_vert
                # 270 degrees counterclockwise (or 90 clockwise) - bottom faces left (toward image)
                label_img = temp_img.rotate(270, expand=True)
            
        else:
            # Horizontal text (top, bottom, left_hor, right_hor)
            if location == 'top':
                # Align to bottom (closest to image)
                total_text_height = sum(line_height + 5 for _ in wrapped_lines) - 5
                y_pos = label_height - total_text_height - 15
            elif location == 'bottom':
                # Align to top (closest to image)
                y_pos = 15
            else:  # left_hor, right_hor
                # Vertically center
                total_text_height = sum(line_height + 5 for _ in wrapped_lines) - 5
                y_pos = (label_height - total_text_height) // 2
            
            for line in wrapped_lines:
                text_width = int(font.getlength(line))
                x_pos = (label_width - text_width) // 2  # Horizontal center
                draw.text((x_pos, y_pos), line, text_color, font=font)
                y_pos += line_height + 5
        
        # Combine image and label based on location
        if location == 'top':
            combined = Image.new('RGB', (width, height + label_height), bg_color)
            combined.paste(label_img, (0, 0))
            combined.paste(image_pil, (0, label_height))
        elif location == 'bottom':
            combined = Image.new('RGB', (width, height + label_height), bg_color)
            combined.paste(image_pil, (0, 0))
            combined.paste(label_img, (0, height))
        elif location in ['left_vert', 'left_hor']:
            combined = Image.new('RGB', (width + label_width, height), bg_color)
            # Center label vertically if needed
            if label_height < height:
                y_offset = (height - label_height) // 2
                combined.paste(label_img, (0, y_offset))
            else:
                combined.paste(label_img, (0, 0))
            combined.paste(image_pil, (label_width, 0))
        else:  # right_vert, right_hor
            combined = Image.new('RGB', (width + label_width, height), bg_color)
            combined.paste(image_pil, (0, 0))
            # Center label vertically if needed
            if label_height < height:
                y_offset = (height - label_height) // 2
                combined.paste(label_img, (width, y_offset))
            else:
                combined.paste(label_img, (width, 0))
        
        return combined
    
    def resize_image(self, image_pil, target_width, target_height, method, bg_color):
        """Resize image using specified method"""
        if method == 'stretch':
            return image_pil.resize((target_width, target_height), Image.LANCZOS)
        
        elif method == 'crop_center':
            # Scale to fill, then crop center
            img_ratio = image_pil.width / image_pil.height
            target_ratio = target_width / target_height
            
            if img_ratio > target_ratio:
                # Image is wider, scale by height
                new_height = target_height
                new_width = int(image_pil.width * (target_height / image_pil.height))
            else:
                # Image is taller, scale by width
                new_width = target_width
                new_height = int(image_pil.height * (target_width / image_pil.width))
            
            resized = image_pil.resize((new_width, new_height), Image.LANCZOS)
            
            # Crop center
            left = (new_width - target_width) // 2
            top = (new_height - target_height) // 2
            return resized.crop((left, top, left + target_width, top + target_height))
        
        elif method == 'fill':
            # Scale to fill completely (may crop)
            img_ratio = image_pil.width / image_pil.height
            target_ratio = target_width / target_height
            
            if img_ratio > target_ratio:
                new_width = target_width
                new_height = int(image_pil.height * (target_width / image_pil.width))
            else:
                new_height = target_height
                new_width = int(image_pil.width * (target_height / image_pil.height))
            
            return image_pil.resize((new_width, new_height), Image.LANCZOS)
        
        else:  # pad
            # Scale to fit, then pad
            img_ratio = image_pil.width / image_pil.height
            target_ratio = target_width / target_height
            
            if img_ratio > target_ratio:
                # Image is wider, scale by width
                new_width = target_width
                new_height = int(image_pil.height * (target_width / image_pil.width))
            else:
                # Image is taller, scale by height
                new_height = target_height
                new_width = int(image_pil.width * (target_height / image_pil.height))
            
            resized = image_pil.resize((new_width, new_height), Image.LANCZOS)
            
            # Create padded image
            padded = Image.new('RGB', (target_width, target_height), bg_color)
            
            # Paste centered
            x_offset = (target_width - new_width) // 2
            y_offset = (target_height - new_height) // 2
            padded.paste(resized, (x_offset, y_offset))
            
            return padded
    
    def calculate_grid_dimensions(self, num_images, shape, cell_width=None, cell_height=None):
        """Calculate grid rows and columns based on shape"""
        if shape == 'horizontal':
            return 1, num_images
        elif shape == 'vertical':
            return num_images, 1
        elif shape == 'square':
            # Find closest to square without blank rows
            side = math.ceil(math.sqrt(num_images))
            rows = side
            cols = math.ceil(num_images / rows)
            return rows, cols
        elif shape == 'smart_square':
            # Consider actual image dimensions for aspect ratio
            if cell_width and cell_height:
                # Calculate what grid dimensions best approximate a square canvas
                target_ratio = 1.0  # Square
                best_diff = float('inf')
                best_rows, best_cols = 1, num_images
                
                for rows in range(1, num_images + 1):
                    cols = math.ceil(num_images / rows)
                    # Only consider if last row has at least one image
                    if (rows - 1) * cols < num_images:
                        # Calculate canvas aspect ratio with these dimensions
                        canvas_width = cols * cell_width
                        canvas_height = rows * cell_height
                        canvas_ratio = canvas_width / canvas_height
                        diff = abs(canvas_ratio - target_ratio)
                        if diff < best_diff:
                            best_diff = diff
                            best_rows = rows
                            best_cols = cols
                
                return best_rows, best_cols
            else:
                # Fallback if dimensions not provided
                best_rows = math.ceil(math.sqrt(num_images))
                best_cols = math.ceil(num_images / best_rows)
                return best_rows, best_cols
        elif shape == 'smart_landscape':
            # Target 3:2 ratio (landscape) considering actual image dimensions
            target_ratio = 3 / 2
            best_diff = float('inf')
            best_rows, best_cols = 1, num_images
            
            if cell_width and cell_height:
                for rows in range(1, num_images + 1):
                    cols = math.ceil(num_images / rows)
                    # Only consider if last row has at least one image
                    if (rows - 1) * cols < num_images:
                        # Calculate canvas aspect ratio
                        canvas_width = cols * cell_width
                        canvas_height = rows * cell_height
                        canvas_ratio = canvas_width / canvas_height
                        diff = abs(canvas_ratio - target_ratio)
                        if diff < best_diff:
                            best_diff = diff
                            best_rows = rows
                            best_cols = cols
            else:
                # Fallback: use number of images
                for rows in range(1, num_images + 1):
                    cols = math.ceil(num_images / rows)
                    # Only consider if last row has at least one image
                    if (rows - 1) * cols < num_images:
                        ratio = cols / rows
                        diff = abs(ratio - target_ratio)
                        if diff < best_diff:
                            best_diff = diff
                            best_rows = rows
                            best_cols = cols
            
            return best_rows, best_cols
        elif shape == 'smart_portrait':
            # Target 2:3 ratio (portrait) considering actual image dimensions
            target_ratio = 2 / 3
            best_diff = float('inf')
            best_rows, best_cols = num_images, 1
            
            if cell_width and cell_height:
                for rows in range(1, num_images + 1):
                    cols = math.ceil(num_images / rows)
                    # Only consider if last row has at least one image
                    if (rows - 1) * cols < num_images:
                        # Calculate canvas aspect ratio
                        canvas_width = cols * cell_width
                        canvas_height = rows * cell_height
                        canvas_ratio = canvas_width / canvas_height
                        diff = abs(canvas_ratio - target_ratio)
                        if diff < best_diff:
                            best_diff = diff
                            best_rows = rows
                            best_cols = cols
            else:
                # Fallback: use number of images
                for rows in range(1, num_images + 1):
                    cols = math.ceil(num_images / rows)
                    # Only consider if last row has at least one image
                    if (rows - 1) * cols < num_images:
                        ratio = cols / rows
                        diff = abs(ratio - target_ratio)
                        if diff < best_diff:
                            best_diff = diff
                            best_rows = rows
                            best_cols = cols
            
            return best_rows, best_cols
        
        return 1, num_images
    
    def create_array(self, images, background, resize, size_method, pad, shape, 
                    labels, label_end, label_location, label_size, font, spacing, label_input=None):
        """Create array of labeled images"""
        # Extract parameters from lists
        background = background[0] if isinstance(background, list) else background
        resize = resize[0] if isinstance(resize, list) else resize
        size_method = size_method[0] if isinstance(size_method, list) else size_method
        pad = pad[0] if isinstance(pad, list) else pad
        shape = shape[0] if isinstance(shape, list) else shape
        labels = labels[0] if isinstance(labels, list) else labels
        label_end = label_end[0] if isinstance(label_end, list) else label_end
        label_location = label_location[0] if isinstance(label_location, list) else label_location
        label_size = label_size[0] if isinstance(label_size, list) else label_size
        font = font[0] if isinstance(font, list) else font
        spacing = spacing[0] if isinstance(spacing, list) else spacing
    
        # Convert background to RGB
        bg_color = (255, 255, 255) if background == 'white' else (0, 0, 0)
        # FIXED: Label background is OPPOSITE of main background
        label_bg = (0, 0, 0) if background == 'white' else (255, 255, 255)
        # FIXED: Text color is OPPOSITE of label background
        text_color = (255, 255, 255) if background == 'white' else (0, 0, 0)
        # Spacing color is OPPOSITE of label background
        spacing_color = (255, 255, 255) if background == 'white' else (0, 0, 0)
    
        # Parse labels
        label_list = self.parse_labels(labels, label_input)
    
        # Convert tensors to PIL images
        pil_images = []
        for img_tensor in images:
            # Handle batch dimension - extract ALL images from batch
            if len(img_tensor.shape) == 4:
                # Tensor is [B, H, W, C] - iterate through batch
                for b in range(img_tensor.shape[0]):
                    img_np = (img_tensor[b].cpu().numpy() * 255).astype(np.uint8)
                    pil_img = Image.fromarray(img_np)
                    pil_images.append(pil_img)
            else:
                # Tensor is [H, W, C] - single image
                img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
                pil_img = Image.fromarray(img_np)
                pil_images.append(pil_img)
    
        num_images = len(pil_images)
    
        # STEP 1: Find target dimensions for padding (BEFORE adding labels)
        widths = [img.width for img in pil_images]
        heights = [img.height for img in pil_images]
    
        if resize == 'grow':
            target_width = max(widths)
            target_height = max(heights)
        else:  # shrink
            target_width = min(widths)
            target_height = min(heights)
    
        # STEP 2: Resize/pad all images to target dimensions if pad is enabled
        if pad:
            processed_images = []
            for pil_img in pil_images:
                processed = self.resize_image(pil_img, target_width, target_height, size_method, bg_color)
                processed_images.append(processed)
        else:
            processed_images = pil_images
            # Recalculate dimensions without forcing uniform size
            widths = [img.width for img in processed_images]
            heights = [img.height for img in processed_images]
            target_width = max(widths)
            target_height = max(heights)
    
        # STEP 3: Calculate maximum label dimensions across all images
        # This ensures all images get the same label padding size
        max_label_width = 0
        max_label_height = 0
        
        # Skip label calculations if label_size is 0
        if label_size > 0:
            # Get all label texts that will be used
            label_texts_to_use = []
            for i in range(num_images):
                label_text = ''
                if label_list:
                    if label_end == 'loop':
                        label_idx = i % len(label_list)
                        label_text = label_list[label_idx]
                    else:  # end
                        if i < len(label_list):
                            label_text = label_list[i]
                label_texts_to_use.append(label_text)
            
            # Calculate max dimensions needed
            for i, pil_img in enumerate(processed_images):
                lw, lh = self.calculate_label_dimensions(
                    label_texts_to_use[i], label_location, label_size, font,
                    pil_img.width, pil_img.height
                )
                max_label_width = max(max_label_width, lw)
                max_label_height = max(max_label_height, lh)
        else:
            # No labels, so create empty label list
            label_texts_to_use = [''] * num_images
        
        # STEP 4: Add labels to padded images using consistent dimensions
        labeled_images = []
        for i, pil_img in enumerate(processed_images):
            # Only add label padding if label_size > 0
            if label_size > 0:
                pil_img = self.add_label_to_image(
                    pil_img, label_texts_to_use[i], label_location, 
                    label_size, font, label_bg, text_color,
                    fixed_label_width=max_label_width,
                    fixed_label_height=max_label_height
                )
            labeled_images.append(pil_img)
    
        # Convert labeled images (without spacing) to tensors for individual output
        labeled_tensors = []
        for pil_img in labeled_images:
            img_np = np.array(pil_img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np).unsqueeze(0)
            labeled_tensors.append(img_tensor)
    
        # STEP 4.5: Add spacing border around each image (if spacing > 0)
        if spacing > 0:
            spaced_images = []
            for pil_img in labeled_images:
                # Create new image with spacing border
                new_width = pil_img.width + (spacing * 2)
                new_height = pil_img.height + (spacing * 2)
                spaced_img = Image.new('RGB', (new_width, new_height), spacing_color)
                # Paste original image in center
                spaced_img.paste(pil_img, (spacing, spacing))
                spaced_images.append(spaced_img)
            labeled_images = spaced_images
    
        # STEP 5: Calculate grid dimensions using LABELED image sizes
        # (All images should be same size after padding + labels are uniform)
        labeled_widths = [img.width for img in labeled_images]
        labeled_heights = [img.height for img in labeled_images]
        cell_width = max(labeled_widths)
        cell_height = max(labeled_heights)
    
        # Calculate grid dimensions using cell dimensions for smart layouts
        rows, cols = self.calculate_grid_dimensions(num_images, shape, cell_width, cell_height)
    
        # STEP 6: Create array canvas
        if shape in ['horizontal', 'vertical']:
            if shape == 'horizontal':
                canvas_width = cell_width * cols
                canvas_height = cell_height
            else:
                canvas_width = cell_width
                canvas_height = cell_height * rows
        else:
            canvas_width = cell_width * cols
            canvas_height = cell_height * rows
    
        canvas = Image.new('RGB', (canvas_width, canvas_height), bg_color)
    
        # STEP 7: Place images in grid
        for i, img in enumerate(labeled_images):
            row = i // cols
            col = i % cols
        
            x_offset = col * cell_width
            y_offset = row * cell_height
        
            # Center image in cell if not perfectly sized
            if img.width < cell_width:
                x_offset += (cell_width - img.width) // 2
            if img.height < cell_height:
                y_offset += (cell_height - img.height) // 2
        
            canvas.paste(img, (x_offset, y_offset))
    
        # Convert back to tensor
        canvas_np = np.array(canvas).astype(np.float32) / 255.0
        canvas_tensor = torch.from_numpy(canvas_np).unsqueeze(0)
    
        print(f"Image Array: {num_images} images | {shape} layout | {canvas_width}x{canvas_height}")
    
        return (canvas_tensor, labeled_tensors)


class RandomSubset:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_text": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "num_to_pick": ("INT", {
                    "default": 1,
                    "min": 1,
                    "step": 1
                }),
                "with_replacement": ("BOOLEAN", {
                    "default": False
                }),
                "random_order": ("BOOLEAN", {
                    "default": True
                }),
                "string_delimiter": ("STRING", {
                    "default": "\\n"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff
                })
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("string_list", "merged_string", "pick_indices")
    OUTPUT_IS_LIST = (True, False, True)
    FUNCTION = "select_subset"
    CATEGORY = "Image Label Tools"
    DESCRIPTION = "Selects a random subset of newline-delimited strings."
    
    def select_subset(self, input_text, num_to_pick, with_replacement, random_order, 
                     string_delimiter, seed):
        # Parse input into list of strings
        items = [line for line in input_text.split('\n') if line.strip()]
        
        if not items:
            return ([], "", [])
        
        # Set random seed for reproducibility
        rng = random.Random(seed)
        
        # Determine actual number to pick
        actual_picks = min(num_to_pick, len(items)) if not with_replacement else num_to_pick
        
        # Pick subset
        if with_replacement:
            picked_indices = [rng.randint(0, len(items) - 1) for _ in range(actual_picks)]
        else:
            if actual_picks >= len(items):
                picked_indices = list(range(len(items)))
            else:
                picked_indices = rng.sample(range(len(items)), actual_picks)
        
        # Get picked items
        picked_items = [items[i] for i in picked_indices]
        
        # Randomize order if requested
        if random_order:
            combined = list(zip(picked_items, picked_indices))
            rng.shuffle(combined)
            picked_items, picked_indices = zip(*combined) if combined else ([], [])
            picked_items = list(picked_items)
            picked_indices = list(picked_indices)
        
        # Process delimiter (handle escaped newline)
        actual_delimiter = string_delimiter.replace('\\n', '\n')
        
        # Create merged string
        merged_string = actual_delimiter.join(picked_items)
        
        return (picked_items, merged_string, picked_indices)



NODE_CLASS_MAPPINGS = {
    'ImageEqualizer': ImageEqualizer,
    'ImageArray': ImageArray,
    'RandomSubset': RandomSubset,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'ImageEqualizer': 'Image Equalizer',
    'ImageArray': 'Image Array',
    'RandomSubset': 'Random Subset',
}
