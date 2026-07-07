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
                'shape': (['horizontal', 'vertical', 'square', 'smart_square', 'smart_landscape', 'smart_portrait'], {'default': 'horizontal'}),
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


import time


class LocalTimerStart:
    """
    Passthrough node that stamps the current time.
    Connect passthrough to your workflow as normal, then wire
    'timer' to a TimerEnd node to measure how long the nodes
    between them took to execute.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "passthrough": ("*", {"tooltip": "Any value or list — passed through unchanged."}),
            }
        }

    RETURN_TYPES = ("*", "TIMER")
    RETURN_NAMES = ("passthrough", "timer")
    OUTPUT_TOOLTIPS = (
        "The input value(s), unchanged.",
        "Epoch timestamp — wire this to a TimerEnd node.",
    )
    FUNCTION = "stamp"
    CATEGORY = "Image Label Tools"
    DESCRIPTION = "Records the current time and passes it to a TimerEnd node. Place this just before the node(s) you want to time."

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")  # NaN != NaN, so ComfyUI always re-executes this node

    def stamp(self, passthrough):
        return (passthrough, time.time())


class LocalTimerEnd:
    """
    Receives the timestamp from a TimerStart node, computes elapsed
    time when this node executes, and outputs the duration.
    Multiple TimerStart/TimerEnd pairs work independently — no
    shared global state.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "passthrough": ("*",     {"tooltip": "Any value or list — passed through unchanged."}),
                "timer":       ("TIMER", {"tooltip": "Connect from a TimerStart node's 'timer' output."}),
                "format":      ([
                                    "h m s",
                                    "hh:mm:ss",
                                    "h",
                                    "m",
                                    "s",
                                ], {
                                    "default": "h m s",
                                    "tooltip": (
                                        "Output format.\n"
                                        "h m s    → '3h 5m 2s' / '34s' (zero parts omitted)\n"
                                        "hh:mm:ss → '03:05:02'\n"
                                        "h        → float hours (2 dp)\n"
                                        "m        → float minutes (2 dp)\n"
                                        "s        → float seconds (2 dp)"
                                    ),
                                }),
            }
        }

    RETURN_TYPES = ("*", "STRING", "FLOAT")
    RETURN_NAMES = ("passthrough", "time_string", "time_float")
    OUTPUT_TOOLTIPS = (
        "The passthrough input, unchanged.",
        "Elapsed time as a formatted string (empty for float-only formats).",
        "Elapsed time as a float in the unit chosen by 'format' (0.0 for string-only formats).",
    )
    FUNCTION = "measure"
    CATEGORY = "Image Label Tools"
    DESCRIPTION = "Computes elapsed time since the paired TimerStart executed. Wire one TimerStart→TimerEnd per section you want to time."

    def measure(self, passthrough, timer, format):
        elapsed = time.time() - timer

        time_string = ""
        time_float  = 0.0

        if format == "h m s":
            h = int(elapsed // 3600)
            m = int((elapsed % 3600) // 60)
            s = elapsed % 60
            s_str = f"{s:.2f}".rstrip('0').rstrip('.')
            parts = []
            if h:
                parts.append(f"{h}h")
            if m or h:
                parts.append(f"{m}m")
            parts.append(f"{s_str}s")
            time_string = " ".join(parts)

        elif format == "hh:mm:ss":
            h = int(elapsed // 3600)
            m = int((elapsed % 3600) // 60)
            s = int(elapsed % 60)
            time_string = f"{h:02d}:{m:02d}:{s:02d}"

        elif format == "h":
            time_float = round(elapsed / 3600, 2)

        elif format == "m":
            time_float = round(elapsed / 60, 2)

        elif format == "s":
            time_float = round(elapsed, 2)

        print(f"Timer: {elapsed:.3f}s elapsed | format={format} | string='{time_string}' float={time_float}")

        return (passthrough, time_string, time_float)


class DuckDuckGoImageSearch:
    """
    Fetches images from DuckDuckGo image search.
    Requires: pip install duckduckgo-search requests Pillow
    """

    REGIONS = [
        "no region", "us-en", "uk-en", "ca-en", "au-en", "de-de", "fr-fr",
        "es-es", "it-it", "nl-nl", "pl-pl", "pt-pt", "ru-ru", "jp-jp",
        "cn-zh", "in-en", "br-pt", "mx-es", "ar-es", "za-en",
    ]
    SAFE_SEARCH = ["moderate", "on", "off"]
    SIZES = ["all", "Small", "Medium", "Large", "Wallpaper"]
    COLORS = [
        "all", "Monochrome", "Red", "Orange", "Yellow", "Green", "Blue",
        "Purple", "Pink", "Brown", "Black", "Gray", "Teal", "White",
        "color", "2tone",
    ]
    TYPES = ["all", "photo", "clipart", "gif", "transparent", "line"]
    LAYOUTS = ["all", "Square", "Tall", "Wide"]
    LICENSES = [
        "all", "any", "Public", "Share", "ShareCommercially",
        "Modify", "ModifyCommercially",
    ]
    AI_IMAGES = ["hide", "show"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "query":          ("STRING", {"multiline": True, "default": ""}),
                "start":          ("INT",    {"default": 0,  "min": 0,   "step": 1}),
                "num_results":    ("INT",    {"default": 1,  "min": 1,   "step": 1}),
                "output_type":    (["list", "batch"],),
                "region":         (cls.REGIONS,     {"default": "no region"}),
                "safe_search":    (cls.SAFE_SEARCH,  {"default": "moderate"}),
                "size":           (cls.SIZES,        {"default": "all"}),
                "color":          (cls.COLORS,       {"default": "all"}),
                "type_image":     (cls.TYPES,        {"default": "all"}),
                "layout":         (cls.LAYOUTS,      {"default": "all"}),
                "license_image":  (cls.LICENSES,     {"default": "all"}),
                "ai_images":      (cls.AI_IMAGES,    {"default": "hide"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image(s)", "url(s)")
    OUTPUT_IS_LIST = (False, True)
    FUNCTION = "search"
    CATEGORY = "Image Label Tools"
    DESCRIPTION = "Fetches images from DuckDuckGo image search by query."

    def search(self, query, start, num_results, output_type,
               region, safe_search, size, color, type_image,
               layout, license_image, ai_images):
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            raise ImportError("duckduckgo-search is not installed. Run: pip install duckduckgo-search")

        import requests
        from io import BytesIO

        # DDG library wants empty string, not None, to mean "no filter"
        def _opt(val):
            return "" if val in ("all", "no region") else val

        # duckduckgo_search uses lowercase safe search values
        safesearch_map = {"on": "on", "moderate": "moderate", "off": "off"}

        # AI image filter: DuckDuckGo uses "1" to filter AI images
        filter_ai = "1" if ai_images == "hide" else None

        kwargs = {"keywords": query, "max_results": start + num_results}

        if region != "no region":
            kwargs["region"] = region
        if safe_search != "moderate":
            kwargs["safesearch"] = safesearch_map.get(safe_search, "moderate")
        if size != "all":
            kwargs["size"] = size
        if color != "all":
            kwargs["color"] = color
        if type_image != "all":
            kwargs["type_image"] = type_image
        if layout != "all":
            kwargs["layout"] = layout
        if license_image != "all":
            kwargs["license_image"] = license_image

        print(f"DDG Image Search: '{query}' | start={start} count={num_results}")

        import time as _time

        results = []
        with DDGS(timeout=20) as ddgs:
            for r in ddgs.images(keywords=query):
                results.append(r)
                if len(results) >= start + num_results:
                    break

        results = results[start:start + num_results]

        if not results:
            raise ValueError(f"DuckDuckGo image search returned no results for: '{query}'")

        headers = {"User-Agent": "Mozilla/5.0"}
        tensors = []
        urls = []

        for r in results:
            url = r.get("image") or r.get("url", "")
            urls.append(url)
            try:
                resp = requests.get(url, headers=headers, timeout=10)
                resp.raise_for_status()
                img = Image.open(BytesIO(resp.content)).convert("RGB")
                arr = np.array(img).astype(np.float32) / 255.0
                tensors.append(torch.from_numpy(arr))
            except Exception as e:
                print(f"DDG Image Search: failed to fetch {url}: {e}")
                # Insert a small black placeholder so index alignment is preserved
                tensors.append(torch.zeros(64, 64, 3))

        if output_type == "batch":
            # Pad all images to the same size before stacking
            max_h = max(t.shape[0] for t in tensors)
            max_w = max(t.shape[1] for t in tensors)
            padded = []
            for t in tensors:
                h, w, c = t.shape
                p = torch.zeros(max_h, max_w, c)
                p[:h, :w, :] = t
                padded.append(p)
            output_tensor = torch.stack(padded, dim=0)
        else:
            # list mode: wrap each in a batch dim of 1
            output_tensor = [t.unsqueeze(0) for t in tensors]

        print(f"DDG Image Search: fetched {len(tensors)} image(s)")
        return (output_tensor, urls)

class LabelImage:
    """Adds a single text label to a single image."""

    @classmethod
    def INPUT_TYPES(cls):
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
                'image':          ('IMAGE',),
                'label':          ('STRING', {'multiline': True, 'default': ''}),
                'label_location': (['top', 'bottom', 'left_vert', 'left_hor', 'right_vert', 'right_hor'], {'default': 'bottom'}),
                'label_size':     ('INT', {'default': 32, 'min': 1, 'max': 200, 'step': 1}),
                'font':           (cls.font_files, {'default': font_default}),
                'label_style':    (['black on white', 'white on black', 'white on dark gray', 'black on light gray'], {'default': 'white on black'}),
            },
            'optional': {
                'label_input': ('STRING', {'forceInput': True}),
            }
        }

    RETURN_TYPES = ('IMAGE',)
    RETURN_NAMES = ('image',)
    FUNCTION = 'label_image'
    CATEGORY = 'Image Label Tools'
    DESCRIPTION = "Adds a text label to a single image. Label text can be typed or connected from another node."

    # ------------------------------------------------------------------ helpers
    # Re-use the same helpers as ImageArray (they are standalone methods so we
    # duplicate them here to keep the class self-contained).

    def get_text_size(self, font, text):
        left, top, right, bottom = font.getbbox(text)
        return right - left, bottom - top

    def wrap_text(self, text, font, max_width):
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

    def _load_font(self, font_path, label_size):
        try:
            if self.font_dir:
                font_file = os.path.join(self.font_dir, font_path)
            else:
                font_file = 'C:/Windows/Fonts/Arial.ttf'
            return ImageFont.truetype(font_file, label_size)
        except Exception:
            return ImageFont.load_default()

    def _add_label(self, image_pil, label_text, location, label_size, font_path, bg_color, text_color):
        """Add label to a PIL image and return the combined PIL image."""
        width, height = image_pil.size
        font = self._load_font(font_path, label_size)
        _, line_height = self.get_text_size(font, "Hg")

        is_vertical      = location in ('left_vert', 'right_vert')
        is_side_hor      = location in ('left_hor',  'right_hor')

        # ---- compute label canvas size ----
        if is_vertical:
            max_width    = height
            wrapped      = self.wrap_text(label_text, font, max_width) if label_text else ['']
            label_width  = max(1, len(wrapped)) * line_height + 30
            label_height = height

        elif is_side_hor:
            max_width    = width // 2
            wrapped      = self.wrap_text(label_text, font, max_width) if label_text else ['']
            max_lw       = max((int(font.getlength(l)) for l in wrapped if l), default=0)
            label_width  = max_lw + 30
            label_height = max(1, len(wrapped)) * (line_height + 5) + 30

        else:  # top / bottom
            max_width    = width
            wrapped      = self.wrap_text(label_text, font, max_width) if label_text else ['']
            label_width  = width
            label_height = max(1, len(wrapped)) * (line_height + 5) + 30

        # ---- draw label ----
        label_img = Image.new('RGB', (label_width, label_height), bg_color)
        draw      = ImageDraw.Draw(label_img)

        if is_vertical:
            # Draw horizontally, then rotate
            temp_img  = Image.new('RGB', (label_height, label_width), bg_color)
            temp_draw = ImageDraw.Draw(temp_img)
            total_th  = sum(line_height + 5 for _ in wrapped) - 5
            y_pos     = label_width - total_th - 15
            for line in wrapped:
                tw    = int(font.getlength(line))
                x_pos = (label_height - tw) // 2
                temp_draw.text((x_pos, y_pos), line, text_color, font=font)
                y_pos += line_height + 5
            if location == 'left_vert':
                label_img = temp_img.rotate(90, expand=True)
            else:
                label_img = temp_img.rotate(270, expand=True)

        else:
            if location == 'top':
                total_th = sum(line_height + 5 for _ in wrapped) - 5
                y_pos    = label_height - total_th - 15
            elif location == 'bottom':
                y_pos = 15
            else:  # side hor
                total_th = sum(line_height + 5 for _ in wrapped) - 5
                y_pos    = (label_height - total_th) // 2

            for line in wrapped:
                tw    = int(font.getlength(line))
                x_pos = (label_width - tw) // 2
                draw.text((x_pos, y_pos), line, text_color, font=font)
                y_pos += line_height + 5

        # ---- composite ----
        if location == 'top':
            combined = Image.new('RGB', (width, height + label_height), bg_color)
            combined.paste(label_img, (0, 0))
            combined.paste(image_pil, (0, label_height))
        elif location == 'bottom':
            combined = Image.new('RGB', (width, height + label_height), bg_color)
            combined.paste(image_pil, (0, 0))
            combined.paste(label_img, (0, height))
        elif location in ('left_vert', 'left_hor'):
            combined = Image.new('RGB', (width + label_width, height), bg_color)
            y_off    = (height - label_height) // 2 if label_height < height else 0
            combined.paste(label_img, (0, y_off))
            combined.paste(image_pil, (label_width, 0))
        else:  # right_vert, right_hor
            combined = Image.new('RGB', (width + label_width, height), bg_color)
            combined.paste(image_pil, (0, 0))
            y_off    = (height - label_height) // 2 if label_height < height else 0
            combined.paste(label_img, (width, y_off))

        return combined

    # ------------------------------------------------------------------ main
    def label_image(self, image, label, label_location, label_size, font,
                    label_style, label_input=None):

        style_map = {
            'black on white':      ((255, 255, 255), (0, 0, 0)),
            'white on black':      ((0, 0, 0),       (255, 255, 255)),
            'white on dark gray':  ((64, 64, 64),    (255, 255, 255)),
            'black on light gray': ((192, 192, 192), (0, 0, 0)),
        }
        bg_color, txt_color = style_map.get(label_style, ((255, 255, 255), (0, 0, 0)))

        # label_input overrides the text widget
        label_text = str(label_input) if label_input is not None else label

        # image is a batch tensor (B, H, W, C); process each frame
        results = []
        for i in range(image.shape[0]):
            frame_np  = (image[i].cpu().numpy() * 255).astype(np.uint8)
            frame_pil = Image.fromarray(frame_np, 'RGB')

            labeled   = self._add_label(frame_pil, label_text, label_location,
                                        label_size, font, bg_color, txt_color)

            out_np    = np.array(labeled).astype(np.float32) / 255.0
            results.append(torch.from_numpy(out_np))

        out_tensor = torch.stack(results, dim=0)
        print(f"LabelImage: {image.shape[0]} frame(s) | location={label_location} | text='{label_text[:40]}'")
        return (out_tensor,)


class AdvancedStitcher:
    """
    Splits an image list into groups, stitches each group into an array,
    and returns a batch of stitched images.

    Group modes
    -----------
    every_N  : interleaved — group k contains images at indices k, k+N, k+2N, …
               Produces N groups. Leftovers (when total not divisible by N) are
               distributed front-to-back: group 0 gets the first extra image,
               group 1 gets the second, etc.

    length_N : chunked — group 0 = [0..N-1], group 1 = [N..2N-1], …
               Last group is padded with blank filler if short.

    Label cycling
    -------------
    The label list (K entries) restarts independently for each stitched group.
    If K < group_size the remaining slots get blank (or repeat-last) labels.
    label_end='blank'       → trailing slots are empty
    label_end='repeat_last' → trailing slots repeat the final label in the list
    """

    INPUT_IS_LIST = True

    # ------------------------------------------------------------------ setup
    @classmethod
    def INPUT_TYPES(cls):
        if os.path.exists(os.path.join(folder_paths.base_path, 'fonts')):
            cls.font_dir  = os.path.join(folder_paths.base_path, 'fonts')
            cls.font_files = [f for f in os.listdir(cls.font_dir)
                              if os.path.isfile(os.path.join(cls.font_dir, f))]
            font_default = cls.font_files[0] if cls.font_files else 'arial.ttf'
        else:
            cls.font_dir   = None
            cls.font_files = ['arial.ttf']
            font_default   = 'arial.ttf'

        return {
            'required': {
                'images':         ('IMAGE',),
                'N':              ('INT', {'default': 4, 'min': 1, 'max': 256, 'step': 1}),
                'group_mode':     (['every_N', 'length_N'], {'default': 'every_N'}),
                # ---- label options ----
                'labels':         ('STRING', {'multiline': True, 'default': ''}),
                'label_end':      (['blank', 'repeat_last'], {'default': 'blank'}),
                'label_location': (['top', 'bottom', 'left_vert', 'left_hor',
                                    'right_vert', 'right_hor'], {'default': 'bottom'}),
                'label_size':     ('INT', {'default': 32, 'min': 0, 'max': 200, 'step': 1}),
                'font':           (cls.font_files, {'default': font_default}),
                'label_style':    (['white on black', 'black on white',
                                    'white on dark gray', 'black on light gray'],
                                   {'default': 'white on black'}),
                # ---- layout options ----
                'shape':          (['horizontal', 'vertical', 'square',
                                    'smart_square', 'smart_landscape', 'smart_portrait'],
                                   {'default': 'horizontal'}),
                'background':     (['black', 'white'], {'default': 'black'}),
                'resize':         (['grow', 'shrink'], {'default': 'grow'}),
                'size_method':    (['pad', 'stretch', 'crop_center', 'fill'], {'default': 'pad'}),
                'pad':            ('BOOLEAN', {'default': True}),
                'spacing':        ('INT', {'default': 5, 'min': 0, 'max': 100, 'step': 1}),
            },
            'optional': {
                'label_input': ('STRING', {'forceInput': True}),
            }
        }

    RETURN_TYPES  = ('IMAGE',)
    RETURN_NAMES  = ('stitched_groups',)
    OUTPUT_IS_LIST = (True,)
    FUNCTION      = 'stitch'
    CATEGORY      = 'Image Label Tools'
    DESCRIPTION   = ("Splits an image list into N groups (interleaved or chunked), "
                     "labels and stitches each group, and returns them as a list of images.")

    # ------------------------------------------------------------------ helpers (mirrors ImageArray)

    def _load_font(self, font_path, size):
        try:
            font_file = (os.path.join(self.font_dir, font_path)
                         if self.font_dir else 'C:/Windows/Fonts/Arial.ttf')
            return ImageFont.truetype(font_file, size)
        except Exception:
            return ImageFont.load_default()

    def _get_text_size(self, font, text):
        l, t, r, b = font.getbbox(text)
        return r - l, b - t

    def _wrap_text(self, text, font, max_width):
        wrapped = []
        for line in text.split('\n'):
            words = line.split(' ')
            if not words:
                wrapped.append('')
                continue
            cur = words[0]
            for w in words[1:]:
                if int(font.getlength(cur + ' ' + w)) <= max_width:
                    cur += ' ' + w
                else:
                    wrapped.append(cur)
                    cur = w
            wrapped.append(cur)
        return wrapped

    def _parse_labels(self, labels_text, label_input):
        """Same logic as ImageArray.parse_labels."""
        if label_input:
            items = label_input if isinstance(label_input, list) else [label_input]
            return [str(x) for x in items]
        if not labels_text.strip():
            return []
        labels_text = labels_text.replace('\\n', '\x00')
        out = []
        for line in labels_text.split('\n'):
            line = line.strip()
            if not line:
                continue
            if ';' in line:
                for part in line.replace('; ', ';').split(';'):
                    part = part.strip()
                    if part:
                        out.append(part.replace('\x00', '\n'))
            else:
                out.append(line.replace('\x00', '\n'))
        return out

    def _calc_label_dims(self, label_text, location, label_size, font_path,
                         img_w, img_h):
        font = self._load_font(font_path, label_size)
        _, lh = self._get_text_size(font, "Hg")
        is_vert     = location in ('left_vert', 'right_vert')
        is_side_hor = location in ('left_hor',  'right_hor')
        if is_vert:
            wrapped = self._wrap_text(label_text, font, img_h) if label_text else ['']
            return max(1, len(wrapped)) * lh + 30, img_h
        elif is_side_hor:
            wrapped = self._wrap_text(label_text, font, img_w // 2) if label_text else ['']
            mw = max((int(font.getlength(l)) for l in wrapped if l), default=0)
            return mw + 30, max(1, len(wrapped)) * (lh + 5) + 30
        else:
            wrapped = self._wrap_text(label_text, font, img_w) if label_text else ['']
            return img_w, max(1, len(wrapped)) * (lh + 5) + 30

    def _add_label(self, image_pil, label_text, location, label_size, font_path,
                   bg_color, text_color,
                   fixed_label_width=None, fixed_label_height=None):
        """Mirrors ImageArray.add_label_to_image exactly."""
        width, height = image_pil.size
        font = self._load_font(font_path, label_size)
        _, line_height = self._get_text_size(font, "Hg")

        is_vertical     = location in ('left_vert', 'right_vert')
        is_side_hor     = location in ('left_hor',  'right_hor')

        if fixed_label_width and fixed_label_height:
            label_width  = fixed_label_width
            label_height = fixed_label_height
            if is_vertical:       mw = height
            elif is_side_hor:     mw = width // 2
            else:                 mw = width
            wrapped = self._wrap_text(label_text, font, mw) if label_text else ['']
        elif is_vertical:
            wrapped      = self._wrap_text(label_text, font, height) if label_text else ['']
            label_width  = max(1, len(wrapped)) * line_height + 30
            label_height = height
        elif is_side_hor:
            wrapped      = self._wrap_text(label_text, font, width // 2) if label_text else ['']
            mw           = max((int(font.getlength(l)) for l in wrapped if l), default=0)
            label_width  = mw + 30
            label_height = max(1, len(wrapped)) * (line_height + 5) + 30
        else:
            wrapped      = self._wrap_text(label_text, font, width) if label_text else ['']
            label_width  = width
            label_height = max(1, len(wrapped)) * (line_height + 5) + 30

        label_img = Image.new('RGB', (label_width, label_height), bg_color)
        draw      = ImageDraw.Draw(label_img)

        if is_vertical:
            temp = Image.new('RGB', (label_height, label_width), bg_color)
            td   = ImageDraw.Draw(temp)
            total_th = sum(line_height + 5 for _ in wrapped) - 5
            y = label_width - total_th - 15
            for line in wrapped:
                x = (label_height - int(font.getlength(line))) // 2
                td.text((x, y), line, text_color, font=font)
                y += line_height + 5
            label_img = temp.rotate(90 if location == 'left_vert' else 270, expand=True)
        else:
            if location == 'top':
                total_th = sum(line_height + 5 for _ in wrapped) - 5
                y = label_height - total_th - 15
            elif location == 'bottom':
                y = 15
            else:
                total_th = sum(line_height + 5 for _ in wrapped) - 5
                y = (label_height - total_th) // 2
            for line in wrapped:
                x = (label_width - int(font.getlength(line))) // 2
                draw.text((x, y), line, text_color, font=font)
                y += line_height + 5

        if location == 'top':
            out = Image.new('RGB', (width, height + label_height), bg_color)
            out.paste(label_img, (0, 0)); out.paste(image_pil, (0, label_height))
        elif location == 'bottom':
            out = Image.new('RGB', (width, height + label_height), bg_color)
            out.paste(image_pil, (0, 0)); out.paste(label_img, (0, height))
        elif location in ('left_vert', 'left_hor'):
            out = Image.new('RGB', (width + label_width, height), bg_color)
            yo = (height - label_height) // 2 if label_height < height else 0
            out.paste(label_img, (0, yo)); out.paste(image_pil, (label_width, 0))
        else:
            out = Image.new('RGB', (width + label_width, height), bg_color)
            yo = (height - label_height) // 2 if label_height < height else 0
            out.paste(image_pil, (0, 0)); out.paste(label_img, (width, yo))
        return out

    def _resize_image(self, image_pil, tw, th, method, bg_color):
        """Mirrors ImageArray.resize_image."""
        if method == 'stretch':
            return image_pil.resize((tw, th), Image.LANCZOS)
        ir = image_pil.width / image_pil.height
        tr = tw / th
        if method == 'crop_center':
            if ir > tr:
                nh = th; nw = int(image_pil.width * th / image_pil.height)
            else:
                nw = tw; nh = int(image_pil.height * tw / image_pil.width)
            r = image_pil.resize((nw, nh), Image.LANCZOS)
            l = (nw - tw) // 2; t = (nh - th) // 2
            return r.crop((l, t, l + tw, t + th))
        elif method == 'fill':
            if ir > tr:
                nw = tw; nh = int(image_pil.height * tw / image_pil.width)
            else:
                nh = th; nw = int(image_pil.width * th / image_pil.height)
            return image_pil.resize((nw, nh), Image.LANCZOS)
        else:  # pad
            if ir > tr:
                nw = tw; nh = int(image_pil.height * tw / image_pil.width)
            else:
                nh = th; nw = int(image_pil.width * th / image_pil.height)
            r = image_pil.resize((nw, nh), Image.LANCZOS)
            padded = Image.new('RGB', (tw, th), bg_color)
            padded.paste(r, ((tw - nw) // 2, (th - nh) // 2))
            return padded

    def _calc_grid(self, n, shape, cw=None, ch=None):
        """Mirrors ImageArray.calculate_grid_dimensions."""
        if shape == 'horizontal':   return 1, n
        if shape == 'vertical':     return n, 1
        if shape == 'square':
            side = math.ceil(math.sqrt(n))
            return side, math.ceil(n / side)
        target = {'smart_square': 1.0, 'smart_landscape': 1.5, 'smart_portrait': 2/3}[shape]
        best_diff = float('inf'); best = (1, n)
        for rows in range(1, n + 1):
            cols = math.ceil(n / rows)
            if (rows - 1) * cols < n:
                ratio = ((cols * cw) / (rows * ch)) if (cw and ch) else (cols / rows)
                d = abs(ratio - target)
                if d < best_diff:
                    best_diff = d; best = (rows, cols)
        return best

    # ------------------------------------------------------------------ grouping

    def _make_groups(self, pil_images, N, mode):
        """
        Returns a list of groups, where each group is a list of PIL images
        (or None for blank filler slots).
        """
        total = len(pil_images)
        if mode == 'every_N':
            # N groups, interleaved
            # Distribute leftovers: first (total % N) groups get one extra image
            leftovers = total % N
            groups = []
            for k in range(N):
                indices = list(range(k, total, N))
                groups.append(indices)
            # leftovers are already handled correctly by range(k, total, N)
            # The groups with k < leftovers will naturally have one more element
            # Now build PIL lists, padding shorter groups to match the longest
            max_len = max(len(g) for g in groups)
            result = []
            for g in groups:
                imgs = [pil_images[i] for i in g]
                imgs += [None] * (max_len - len(imgs))
                result.append(imgs)
            return result
        else:  # length_N
            groups = []
            for start in range(0, total, N):
                chunk = list(pil_images[start:start + N])
                chunk += [None] * (N - len(chunk))
                groups.append(chunk)
            return groups

    # ------------------------------------------------------------------ stitch one group

    def _stitch_group(self, group_images, label_list, label_end,
                      label_location, label_size, font,
                      bg_color, label_bg, text_color, spacing_color,
                      resize, size_method, do_pad, shape):
        """
        Takes a list of PIL images (None = filler), labels them, and stitches
        into a single PIL image using the same pipeline as ImageArray.
        """
        group_size = len(group_images)

        # Build per-slot label texts
        # Label list restarts from 0 for each group; K < group_size → blank or repeat_last
        slot_labels = []
        K = len(label_list)
        for i in range(group_size):
            if K == 0:
                slot_labels.append('')
            elif i < K:
                slot_labels.append(label_list[i])
            else:
                # past end of label list
                if label_end == 'repeat_last':
                    slot_labels.append(label_list[-1])
                else:  # blank
                    slot_labels.append('')

        # Build filler image (same size as first non-None image)
        ref_img = next((img for img in group_images if img is not None), None)
        if ref_img is None:
            return None  # entire group is blank — shouldn't happen but guard anyway
        filler = Image.new('RGB', ref_img.size, bg_color)

        pil_imgs = [img if img is not None else filler for img in group_images]

        # --- resize/pad to uniform size ---
        widths  = [img.width  for img in pil_imgs]
        heights = [img.height for img in pil_imgs]
        if resize == 'grow':
            tw, th = max(widths), max(heights)
        else:
            tw, th = min(widths), min(heights)

        if do_pad:
            pil_imgs = [self._resize_image(img, tw, th, size_method, bg_color)
                        for img in pil_imgs]
        else:
            tw, th = max(widths), max(heights)

        # --- calculate uniform label dimensions ---
        max_lw = max_lh = 0
        if label_size > 0:
            for i, img in enumerate(pil_imgs):
                lw, lh = self._calc_label_dims(
                    slot_labels[i], label_location, label_size, font,
                    img.width, img.height)
                max_lw = max(max_lw, lw)
                max_lh = max(max_lh, lh)

        # --- label each image ---
        labeled = []
        for i, img in enumerate(pil_imgs):
            if label_size > 0:
                img = self._add_label(img, slot_labels[i], label_location,
                                      label_size, font, label_bg, text_color,
                                      fixed_label_width=max_lw,
                                      fixed_label_height=max_lh)
            labeled.append(img)

        # --- add spacing ---
        if spacing_color is not None and self._spacing > 0:
            spaced = []
            for img in labeled:
                si = Image.new('RGB',
                               (img.width + self._spacing * 2,
                                img.height + self._spacing * 2),
                               spacing_color)
                si.paste(img, (self._spacing, self._spacing))
                spaced.append(si)
            labeled = spaced

        # --- grid layout ---
        cw = max(img.width  for img in labeled)
        ch = max(img.height for img in labeled)
        rows, cols = self._calc_grid(len(labeled), shape, cw, ch)

        canvas_w = cw * cols
        canvas_h = ch * rows
        canvas = Image.new('RGB', (canvas_w, canvas_h), bg_color)

        for i, img in enumerate(labeled):
            r, c   = divmod(i, cols)
            xo     = c * cw + (cw - img.width)  // 2
            yo     = r * ch + (ch - img.height) // 2
            canvas.paste(img, (xo, yo))

        return canvas

    # ------------------------------------------------------------------ main

    def stitch(self, images, N, group_mode, labels, label_end,
               label_location, label_size, font, label_style,
               shape, background, resize, size_method, pad, spacing,
               label_input=None):

        # Unwrap list-wrapped scalars (INPUT_IS_LIST=True)
        def uw(v): return v[0] if isinstance(v, list) else v
        N              = uw(N)
        group_mode     = uw(group_mode)
        labels         = uw(labels)
        label_end      = uw(label_end)
        label_location = uw(label_location)
        label_size     = uw(label_size)
        font           = uw(font)
        label_style    = uw(label_style)
        shape          = uw(shape)
        background     = uw(background)
        resize         = uw(resize)
        size_method    = uw(size_method)
        do_pad         = uw(pad)
        spacing_val    = uw(spacing)
        self._spacing  = spacing_val  # stored for _stitch_group access

        # Colors
        bg_color = (0, 0, 0) if background == 'black' else (255, 255, 255)
        style_map = {
            'white on black':      ((0, 0, 0),       (255, 255, 255)),
            'black on white':      ((255, 255, 255), (0, 0, 0)),
            'white on dark gray':  ((64, 64, 64),    (255, 255, 255)),
            'black on light gray': ((192, 192, 192), (0, 0, 0)),
        }
        label_bg, text_color = style_map.get(label_style, ((0, 0, 0), (255, 255, 255)))
        spacing_color = bg_color  # spacing strip matches background

        # Collect PIL images from tensor list
        pil_images = []
        for img_tensor in images:
            if len(img_tensor.shape) == 4:
                for b in range(img_tensor.shape[0]):
                    arr = (img_tensor[b].cpu().numpy() * 255).astype(np.uint8)
                    pil_images.append(Image.fromarray(arr))
            else:
                arr = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
                pil_images.append(Image.fromarray(arr))

        total = len(pil_images)
        print(f"AdvancedStitcher: {total} images | mode={group_mode} | N={N} | shape={shape}")

        # Parse labels
        label_list = self._parse_labels(labels, label_input)

        # Build groups
        groups = self._make_groups(pil_images, N, group_mode)

        # Stitch each group
        output_tensors = []
        for gi, group in enumerate(groups):
            stitched = self._stitch_group(
                group, label_list, label_end,
                label_location, label_size, font,
                bg_color, label_bg, text_color, spacing_color,
                resize, size_method, do_pad, shape)
            if stitched is None:
                continue
            arr = np.array(stitched).astype(np.float32) / 255.0
            output_tensors.append(torch.from_numpy(arr).unsqueeze(0))
            print(f"  Group {gi}: {stitched.width}x{stitched.height}")

        if not output_tensors:
            # Fallback: return a blank tensor
            blank = torch.zeros((1, 64, 64, 3))
            output_tensors = [blank]

        return (output_tensors,)


NODE_CLASS_MAPPINGS = {
    'ImageEqualizer': ImageEqualizer,
    'ImageArray': ImageArray,
    'RandomSubset': RandomSubset,
    'LocalTimerStart': LocalTimerStart,
    'LocalTimerEnd': LocalTimerEnd,
    'DuckDuckGoImageSearch': DuckDuckGoImageSearch,
    'LabelImage': LabelImage,
    'AdvancedStitcher': AdvancedStitcher,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'ImageEqualizer': 'Image Equalizer',
    'ImageArray': 'Image Array',
    'RandomSubset': 'Random Subset',
    'LocalTimerStart': 'Local Timer Start',
    'LocalTimerEnd': 'Local Timer End',
    'DuckDuckGoImageSearch': 'DuckDuckGo img search',
    'LabelImage': 'Label Image',
    'AdvancedStitcher': 'Advanced Stitcher',
}
