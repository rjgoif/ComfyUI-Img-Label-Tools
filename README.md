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
- `size_mode` - Either shrinking all large imgages or growing smaller images up.
- `upscale_method` - Scaling algorithm (`nearest-exact`, `bilinear`, `area`, `bicubic`, `lanczos`)
- `keep_proportion` - How to handle aspect ratios:
  - `pad` - Add padding to maintain aspect ratio (default)
  - `stretch` - Stretch to fill target size
  - `resize` - Scale to fit within target (results may not all be the same size, but will fit in a bounding box)
  - `crop` - Crop to fill target
  - `total_pixels` - Maintain total pixel count (results may not all be the same size, will depend on aspect ratio)
- `pad_color` - Color for padding: black, white, gray, a weighted mean of all colors in the image, or just of the edge of the image. 
- `crop_position` - Position for cropping (`center`, `top`, `bottom`, `left`, `right`)

**Outputs:**
- `images` - Batch or list of equalized images (all same dimensions)

**Use Cases:**
- Prepare images for array display (primary use)
- Prepare images for batch processing without losing data
- Create uniform image sets for training data (one of many nodes that can)

---

### Image Array

Creates organized grids/arrays of images with optional text labels. Images are first equalized in size, then labeled, and finally arranged in customizable layouts.

**Category:** Image Label Tools

**Inputs:**
- `images` - List or batch of images to arrange
- `label_input` (optional) - Dynamic label input (string, number, or list)

**Parameters:**

**Layout & Sizing:**
- `background` - Decides the pad color and text color, and the opposite color will be used for label backgrounds
- `shape` - Array layout:
  - `horizontal` - Single row (default)
  - `vertical` - Single column
  - `square` - N×M grid closest to square (minimizes blank cells)
  - `smart_square` - Optimizes for 1:1 aspect ratio considering image dimensions
  - `smart_landscape` - Optimizes for 3:2 aspect ratio considering image dimensions
  - `smart_portrait` - Optimizes for 2:3 aspect ratio considering image dimensions
- `smart_ratio` - Custom W:H target for smart layout. When filled, overrides the shape picklist and runs the grid search against this ratio. Accepts `16:9`, `16/9`, `1.78`, `3x1`, `3W x 1H`, etc. Leave blank to use the shape picklist.
- `resize_any` - ON: normalize image sizes before arranging. OFF: place at native sizes (boolean)
- `size_guide` - Target size when `resize_any` is ON: `largest` or `smallest` image in the group
- `size_method` - How to transform each image to the target size. Will only come into play if you have different sized inputs.
  - `pad small dim` - Letterbox: scale to fit, fill leftover space with background color (default)
  - `stretch` - Distort to exact target size
  - `crop_center` - Scale to fill, crop center to exact size
  - `crop large dim` - Scale to fill; large axis overflows and is clipped by the canvas

**Labels:**
- `labels` - Multiline text box for labels (one per line, or semicolon-separated)
- `label_end` - If more images than labels, what to do? Either start over with repeats or just stop
- `label_location` - Where to place labels.
- `label_size` - Font size in pixels (0-200, default 32; set to 0 to disable labels)
- `font` - Font selection from ComfyUI/fonts directory (falls back to Arial)

**Spacing:**
- `spacing` - Pixel border around each image (0-100, default 5), high contrast

**Outputs:**
- `array` - Single combined image containing the complete array
- `images` - List of individual labeled images

**Label Text Format:**
Labels can be provided in multiple ways:
- **Multi-line labels:** Use `\n` within a label for line breaks in the text
  ```
  Image 1
  Image 2\nLine 2
  Image 3
  ```
  This creates 3 labels, where the second label displays on two lines.
    
- **Semicolon-separated:** `Image 1; Image 2; Image 3`
- **Dynamic input:** Connect string, number, or list output from another node
- **Numbers:** Automatically formatted (decimals truncated to 5 places, trailing zeros removed)

**Note:** Actual newlines (pressing Enter) separate different labels. Use `\n` (backslash-n) for line breaks within a single label.


**Use Cases:**
- Image galleries/comparisons (for Reddit, etc)
- Manual XYZ stuff.

---

### Label Image

Adds a text label to a single image.

**Category:** Image Label Tools

**Inputs:**
- `image` - Image to label
- `label_input` (optional) - Label text from another node (overrides the text box)

**Parameters:**
- `label` - Free text box (multiline)
- `label_location` - Where to place the label (`top`, `bottom`, `left_vert`, `left_hor`, `right_vert`, `right_hor`)
- `label_size` - Font size in pixels (1-200, default 32)
- `font` - Font selection from ComfyUI/fonts directory (falls back to Arial)
- `label_style` - Color scheme:
  - `white on black` (default)
  - `black on white`
  - `white on dark gray`
  - `black on light gray`

**Outputs:**
- `image` - Labeled image

---

### Advanced Stitcher

Splits an image list into groups, stitches each group into a labeled grid, and returns the results as a list of images. Primary use case is interleaved grouping — comparing every Nth image across a batch side by side.

**Category:** Image Label Tools

**Inputs:**
- `images` - Image list
- `label_input` (optional) - Label text from another node

**Parameters:**

**Grouping:**
- `N` - Group count or group size (default 1)
- `group_mode`:
  - `every_N` - Interleaved: produces N groups, each containing every Nth image. With 12 images and N=3: groups are [0,3,6,9], [1,4,7,10], [2,5,8,11]. Leftover images fill front-to-back.
  - `length_N` - Chunked: sequential groups of N images. With 12 images and N=3: groups are [0,1,2], [3,4,5], [6,7,8], [9,10,11]. Short final groups are padded with a blank filler image.

**Labels:**
- `labels` - Multiline text; the label list restarts independently for each group. If K labels < group size, trailing slots are blank or repeat the last label depending on `label_end`.
- `label_end`:
  - `blank` - Trailing slots past the end of the label list are empty
  - `repeat_last` - Trailing slots repeat the final label
- `label_location`, `label_size`, `font` - Same as Image Array
- `label_style` - Color scheme (same 4 options as Label Image; default `white on black`)

**Layout & Sizing:**
- `shape` - Grid layout for each stitched group (same options as Image Array; default `horizontal`)
- `smart_ratio` - Custom W:H target, same format as Image Array
- `background` - Canvas fill color (`black` / `white`)
- `resize_any` - ON: normalize sizes before stitching. OFF: place at native sizes
- `size_guide` - `largest` or `smallest` target when `resize_any` is ON
- `size_method` - Same 4 options as Image Array
- `pad_axis` - Which dimension to normalize:
  - `both` - Normalize width and height (default)
  - `height_only` - Normalize height only; widths vary. Ideal for horizontal stitching of variable-width images.
  - `width_only` - Normalize width only; heights vary. Ideal for vertical stitching.
- `spacing` - Pixel border around each image (0-100, default 5)

**Outputs:**
- `stitched_groups` - List of stitched images, one per group

---

### Random Subset
Selects a random subset from newline-delimited text. Supports sampling with/without replacement, optional randomization of output order, and configurable delimiters. Returns the subset as a string list, merged string, and list of indices.

---

## Examples
Eventually...

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
- Use `smart_square` for photo galleries with mixed portrait/landscape images
- Set `spacing` to 5-10 pixels for clean visual separation
- Place labels on `bottom` or `top` for easier reading
- Use `left_vert` or `right_vert` for space-efficient labeling
- For dynamic labels, connect output from other nodes like counters or text processors
- `smart_ratio` accepts most common formats: `16:9`, `16/9`, `1.78`, `3x1`, `3W x 1H`

**Advanced Stitcher:**
- Use `every_N` grouping to compare corresponding frames across N different generation runs side by side
- Use `pad_axis: height_only` with horizontal stitching when input images have different widths but matching heights — avoids width padding and keeps each image at its native width
- With `resize_any` OFF, images are placed at native sizes; the canvas grows to fit without padding

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