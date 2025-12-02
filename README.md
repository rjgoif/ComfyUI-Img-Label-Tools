# ComfyUI Img Label Tools

Custom nodes for ComfyUI that provide advanced image processing and labeling capabilities for creating organized image arrays and equalizing image dimensions.

## Installation

### Via ComfyUI Manager (Recommended)

1. Open ComfyUI Manager
2. Search for "Img Label Tools"
3. Click Install

### Manual Installation

1. Navigate to your ComfyUI custom_nodes directory:
   ```bash
   cd ComfyUI/custom_nodes/
   ```

2. Clone this repository:
   ```bash
   git clone https://github.com/rjgoif/ComfyUI-Img-Label-Tools.git
   ```

3. Restart ComfyUI

## Nodes

### Image Equalizer

Equalizes multiple images to matching dimensions by either growing all images to the largest size or shrinking all to the smallest size.

**Category:** Image Label Tools

**Inputs:**
- `images` - List or batch of images to equalize

**Parameters:**
- `size_mode` - How to determine target dimensions:
  - `grow` - Scale all images up to match the largest dimensions
  - `shrink` - Scale all images down to match the smallest dimensions
- `upscale_method` - Scaling algorithm (`nearest-exact`, `bilinear`, `area`, `bicubic`, `lanczos`)
- `keep_proportion` - How to handle aspect ratios:
  - `pad` - Add padding to maintain aspect ratio (default)
  - `stretch` - Stretch to fill target size
  - `resize` - Scale to fit within target
  - `crop` - Crop to fill target
  - `total_pixels` - Maintain total pixel count
- `pad_color` - Color for padding:
  - `black` - RGB(0,0,0)
  - `white` - RGB(255,255,255)
  - `gray` - RGB(128,128,128)
  - `average` - Gamma-corrected weighted mean color of entire image
  - `average_edge` - Weighted mean of peripheral 5% of pixels
- `crop_position` - Position for cropping (`center`, `top`, `bottom`, `left`, `right`)

**Outputs:**
- `images` - Batch or list of equalized images (all same dimensions)

**Use Cases:**
- Prepare images for batch processing
- Create uniform image sets for training data
- Normalize mixed-size images from different sources

---

### Image Array

Creates organized grids/arrays of images with optional text labels. Images are first equalized in size, then labeled, and finally arranged in customizable layouts.

**Category:** Image Label Tools

**Inputs:**
- `images` - List or batch of images to arrange
- `label_input` (optional) - Dynamic label input (string, number, or list)

**Parameters:**

**Layout & Sizing:**
- `background` - Canvas background color (`white`, `black`)
- `resize` - How to normalize image sizes:
  - `grow` - Scale all up to largest dimensions
  - `shrink` - Scale all down to smallest dimensions
- `size_method` - Resizing approach:
  - `pad` - Scale to fit, add padding (maintains aspect ratio)
  - `stretch` - Stretch to fill
  - `crop_center` - Scale to fill, crop excess from center
  - `fill` - Scale to cover entire area
- `pad` - Enable/disable uniform sizing (boolean)
- `shape` - Array layout:
  - `horizontal` - Single row
  - `vertical` - Single column
  - `square` - N×M grid closest to square (minimizes blank cells)
  - `smart_square` - Optimizes for 1:1 aspect ratio considering image dimensions
  - `smart_landscape` - Optimizes for 3:2 aspect ratio considering image dimensions
  - `smart_portrait` - Optimizes for 2:3 aspect ratio considering image dimensions

**Labels:**
- `labels` - Multiline text box for labels (one per line, or semicolon-separated)
- `label_end` - Behavior when labels run out:
  - `loop` - Wrap around to first label
  - `end` - No label for remaining images (empty padding still added)
- `label_location` - Where to place labels:
  - `top` - Above image (text bottom-aligned)
  - `bottom` - Below image (text top-aligned)
  - `left_vert` - Left of image, vertical text rotated 90° (bottom faces right)
  - `left_hor` - Left of image, horizontal text (vertically centered)
  - `right_vert` - Right of image, vertical text rotated 270° (bottom faces left)
  - `right_hor` - Right of image, horizontal text (vertically centered)
- `label_size` - Font size in pixels (8-200, default 32)
- `font` - Font selection from ComfyUI/fonts directory (falls back to Arial)

**Spacing:**
- `spacing` - Pixel border around each image (0-100, default 0)
  - Color is opposite of background for contrast
  - Applied after all processing (labels, padding, etc.)

**Outputs:**
- `image` - Single combined image containing the complete array

**Processing Order:**
1. Convert inputs to PIL images (handles both lists and batches)
2. Equalize image sizes (resize/pad based on settings)
3. Calculate maximum label dimensions across all images
4. Add labels with consistent dimensions to all images
5. Add spacing border (if enabled)
6. Calculate optimal grid layout (for smart modes, considers actual image dimensions)
7. Create canvas and place images in grid

**Label Text Format:**
Labels can be provided in multiple ways:
- **Multiline text widget:** One label per line
  ```
  Image 1
  Image 2
  Image 3
  ```
- **Semicolon-separated:** `Image 1; Image 2; Image 3`
- **Dynamic input:** Connect string, number, or list output from another node
- **Numbers:** Automatically formatted (decimals truncated to 5 places, trailing zeros removed)

**Smart Layout Behavior:**
Smart layouts consider the actual dimensions of images (after padding and labels) to optimize the grid arrangement:
- **Example:** 6 tall images (400×1200 each after labels)
  - Regular `square`: Might arrange as 2×3 (canvas 800×3600)
  - `smart_square`: Arranges as 3×2 (canvas 1200×2400) - closer to 1:1 aspect ratio
  
This ensures the final canvas has the desired proportions regardless of individual image shapes.

**Use Cases:**
- Create contact sheets or image galleries
- Generate labeled comparison grids
- Build training data visualizations
- Create annotated image catalogs
- Make presentation-ready image arrays

---

## Examples

### Image Equalizer
```
Input: 5 images with various sizes (800×600, 1920×1080, 640×480, 1024×768, 1280×720)

Settings:
- size_mode: grow
- keep_proportion: pad
- pad_color: black

Output: 5 images all sized 1920×1080 with black padding to maintain aspect ratios
```

### Image Array
```
Input: 12 product images of different sizes

Settings:
- background: white
- resize: grow
- size_method: pad
- shape: smart_square
- labels: "Product A\nProduct B\n..." (12 labels)
- label_location: bottom
- label_size: 24
- spacing: 10

Output: Single image with 4×3 grid (optimized for square canvas), each product labeled at bottom, 
        with 10px black borders separating images
```

## Requirements

- ComfyUI (latest version recommended)
- Python 3.8+
- Pillow (PIL)
- PyTorch

## Tips

**Image Equalizer:**
- Use `average_edge` padding for seamless borders that match image content
- Use `grow` + `pad` for lossless size normalization
- Use `shrink` + `crop` to create uniform thumbnails

**Image Array:**
- Enable `pad` to ensure uniform cell sizes in the grid
- Use `smart_square` for photo galleries with mixed portrait/landscape images
- Set `spacing` to 5-10 pixels for clean visual separation
- Place labels on `bottom` or `top` for easier reading
- Use `left_vert` or `right_vert` for space-efficient labeling
- For dynamic labels, connect output from other nodes like counters or text processors

## Font Setup

For custom fonts:
1. Create a `fonts` directory in your ComfyUI installation root
2. Place TrueType (.ttf) font files in this directory
3. Restart ComfyUI - fonts will appear in the dropdown

If no fonts directory exists, the node will fall back to Arial (Windows) or the system default.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Credits

- Image Equalizer inspired by KJNodes by GitHub user kijai
- Labels for Image Array inspired by Mikey Nodes by GitHub user bash-j
- Created for the ComfyUI community

## License

MIT License - see LICENSE file for details

## Support

If you encounter any issues or have suggestions, please open an issue on GitHub:
https://github.com/rjgoif/ComfyUI-Img-Label-Tools/issues
