# ComfyUI Img Label Tools

Custom nodes for ComfyUI that provide advanced image processing and labeling capabilities for creating organized image arrays and equalizing image dimensions.

Every widget has a hover tooltip that explains what it does. This README covers the same options in more detail.

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

## Nodes at a glance

All nodes are in the **Image Label Tools** category.

| Node | What it does |
| --- | --- |
| [Image Equalizer](#image-equalizer) | Brings a set of images to one common size (largest, smallest, or a chosen image). |
| [Image Array](#image-array) | Arranges images into a labeled grid and returns one combined image. |
| [Label Image](#label-image) | Adds a text label to a single image. |
| [Advanced Stitcher](#advanced-stitcher) | Splits images into groups and stitches each group into its own labeled grid. |
| [Random Subset](#random-subset) | Picks a random subset of the lines in a block of text. |
| [Load Images From Folder(s)](#load-images-from-folders) | Loads images from a folder with filtering, stepping, limiting and sorting. |
| [DuckDuckGo img search](#duckduckgo-img-search) | Downloads images from DuckDuckGo image search. |
| [Local Timer Start / End](#local-timer-start-and-local-timer-end) | Measures how long the nodes between them take to run. |

## Nodes

### Image Equalizer

Equalizes multiple images to matching dimensions. Every image from every connected batch or list is treated as one set, in input order.

**Inputs:**
- `images` - List or batch of images to equalize

**Parameters:**
- `size_mode` - Which size every image is brought to:
  - `grow` - Match the largest width and height among the images (default)
  - `shrink` - Match the smallest width and height among the images
  - `match` - Match the size of image number `match_N`
- `match_N` - Only used when `size_mode` is `match` (default 1, max 9999). Which image sets the target size, counting from 1 in input order. If there are fewer images than this number, the last image is used, so a large value such as `9999` always means the last image.
- `upscale_method` - Resampling method used whenever an image is resized (`nearest-exact`, `bilinear`, `area`, `bicubic`, `lanczos`)
- `keep_proportion` - How each image is fitted to the target size:
  - `pad` - Scale to fit inside the target, then add padding to reach the exact size (default)
  - `stretch` - Resize to the exact target size, ignoring aspect ratio
  - `resize` - Scale to fit inside the target with no padding. Results may not all be the same size, but each fits in the target bounding box.
  - `crop` - Scale to fill the target and cut off the overflow
  - `total_pixels` - Scale each image, keeping its own aspect ratio, to the same total pixel count as the target. Results may not all be the same size, depending on aspect ratio.
- `pad_color` - Padding color. Only used when `keep_proportion` is `pad`.
  - `black`, `white`, `gray` - Solid colors
  - `average` - Gamma-corrected average color of the resized image
  - `average_edge` - Gamma-corrected average color of the outer 5% border of the resized image
- `crop_position` - `center`, `top`, `bottom`, `left` or `right`.
  - With `pad`: where the image sits inside the padded frame. The padding goes on the opposite side.
  - With `crop`: which part of the image is kept.
  - `top` and `bottom` only matter when height is padded or cropped. `left` and `right` only matter when width is.
  - Ignored by `stretch`, `resize` and `total_pixels`.

**Outputs:**
- `images` - A list of the resized images, in input order. All images share one size, except with `keep_proportion` set to `resize` or `total_pixels`.

**Use Cases:**
- Prepare images for array display (primary use)
- Prepare images for batch processing without losing data
- Create uniform image sets for training data (one of many nodes that can)
- Bring every image to the size of one reference image (`size_mode: match`)

---

### Image Array

Creates organized grids/arrays of images with optional text labels. Images are first resized to a common size, then labeled, and finally arranged in a grid. Images always fill the grid left to right, then top to bottom, whatever the shape.

**Inputs:**
- `images` - List or batch of images to arrange. Every image from every connected batch or list is used, in input order.
- `label_input` (optional) - Dynamic label input (string, number, or list). When connected, it replaces the `labels` box.

**Parameters:**

**Layout:**
- `background` - Canvas color (`white` or `black`). It is also used for the gaps from `spacing` and for padding added by `size_method`. Label boxes use the opposite color, with their text in the background color.
- `shape` - Array layout:
  - `horizontal` - Single row (default)
  - `vertical` - Single column
  - `square` - N×M grid closest to square (minimizes blank cells)
  - `smart_square` - Optimizes for 1:1 aspect ratio considering image dimensions
  - `smart_landscape` - Optimizes for 3:2 aspect ratio considering image dimensions
  - `smart_portrait` - Optimizes for 2:3 aspect ratio considering image dimensions
  - `force_rows` - Output has `forced_N` rows. Columns are worked out from the image count (see `forced_N`).
  - `force_cols` - Output has `forced_N` columns. Rows are worked out from the image count (see `forced_N`).
- `smart_ratio` - Custom W:H target for smart layout. When filled, overrides the shape picklist and runs the grid search against this ratio. Accepts `16:9`, `16/9`, `1.78`, `3x1`, `3W x 1H`, etc. Leave blank to use the shape picklist. `force_rows` / `force_cols` take priority over `smart_ratio`.
- `forced_N` - Only used when `shape` is `force_rows` or `force_cols` (default 1). Sets the number of rows or columns in the output. Fill order never changes: images always fill left to right, then top to bottom, so 14 images with `force_rows` = 3 or `force_cols` = 5 give the same 3×5 grid. Capped at the image count. With `force_rows`, if the exact row count can't be reached without an empty row (for example 14 images with 6 rows), the closest row count without one is used (5 rows).

**Sizing:**
- `resize_any` - ON: resize all images to one common size before arranging. OFF: keep native sizes; each grid cell is as large as the largest image, and smaller images are centered in their cell.
- `size_guide` - Target size when `resize_any` is ON: the `largest` or `smallest` image, or `match` to use the size of image number `match_N`
- `match_N` - Only used when `size_guide` is `match` (default 1, max 9999). Which image sets the target size, counting from 1 in input order. If there are fewer images than this number, the last image is used, so a large value such as `9999` always means the last image.
- `size_method` - How each image is fitted to the target size. It only comes into play if the input images are different sizes.
  - `pad small dim` - Scale to fit inside the target and fill the rest with the background color (default)
  - `stretch` - Resize to the exact target size, ignoring aspect ratio
  - `crop_center` - Scale to fill the target, then cut the overflow equally from both sides
  - `crop large dim` - Scale to fit inside the target with no padding and no cropping. One side matches the target and the other is smaller, so image sizes can differ.

**Labels:**
- `labels` - Multiline text box for labels, one per image in order. See [Label text format](#label-text-format).
- `label_end` - What happens when there are more images than labels:
  - `loop` - Start over from the first label (default)
  - `end` - Images after the last label get a blank label. The label space is still reserved.
- `label_location` - Where each label goes relative to its image: `top`, `bottom`, `left_hor`, `right_hor` (horizontal text beside the image) or `left_vert`, `right_vert` (rotated text beside the image). Every image gets the same amount of label space, sized for the longest label, so cells stay aligned.
- `label_size` - Font size in pixels (0-200, default 32). Set to 0 to turn labels off completely; no label space is added.
- `font` - Font selection from the ComfyUI `fonts` directory. See [Font Setup](#font-setup).

**Spacing:**
- `spacing` - Gap in pixels added around every image, after its label (0-100, default 5). It is filled with the background color, so it shows up as a gap. The gap between two neighbors is double this value.

**Outputs:**
- `array` - The finished grid as one image
- `images` - List of the individual images after resizing and labeling, before spacing is added

#### Label text format

Labels can be provided in multiple ways:
- **Multi-line labels:** Use `\n` within a label for line breaks in the text
  ```
  Image 1
  Image 2\nLine 2
  Image 3
  ```
  This creates 3 labels, where the second label displays on two lines.

- **Semicolon-separated:** `Image 1; Image 2; Image 3`
- **Dynamic input:** Connect string, number, or list output from another node to `label_input`. It replaces the `labels` box.
- **Numbers:** Automatically formatted (decimals truncated to 5 places, trailing zeros removed)

**Note:** Actual newlines (pressing Enter) separate different labels. Use `\n` (backslash-n) for line breaks within a single label.

**Use Cases:**
- Image galleries/comparisons (for Reddit, etc)
- Manual XYZ stuff.

---

### Label Image

Adds a text label to a single image. If a batch is connected, every frame gets the same label. The image gets larger to make room for the label.

**Inputs:**
- `image` - Image to label
- `label_input` (optional) - Label text from another node. When connected, it replaces the text box.

**Parameters:**
- `label` - Free text box (multiline). Press Enter for a new line.
- `label_location` - Where to place the label (`top`, `bottom`, `left_vert`, `left_hor`, `right_vert`, `right_hor`). `_hor` puts horizontal text beside the image and `_vert` puts rotated text beside it.
- `label_size` - Font size in pixels (1-200, default 32)
- `font` - Font selection from the ComfyUI `fonts` directory. See [Font Setup](#font-setup).
- `label_style` - Color scheme, written as text color on label box color:
  - `white on black` (default)
  - `black on white`
  - `white on dark gray`
  - `black on light gray`

**Outputs:**
- `image` - Labeled image

---

### Advanced Stitcher

Splits an image list into groups, stitches each group into a labeled grid, and returns the results as a list of images. Primary use case is interleaved grouping: comparing every Nth image across a batch side by side.

Its sizing, layout and label options work like the ones in [Image Array](#image-array). The differences are that label colors come from `label_style` (Image Array derives them from `background`), `background` defaults to `black`, and `label_end` has different options (see below).

**Inputs:**
- `images` - Image list. Every image from every connected batch or list is used, in input order.
- `label_input` (optional) - Label text from another node. When connected, it replaces the `labels` box.

**Parameters:**

**Grouping:**
- `N` - Group count or group size (default 1), depending on `group_mode`
- `group_mode`:
  - `every_N` - Interleaved: produces N groups, each containing every Nth image. With 12 images and N=3: groups are [0,3,6,9], [1,4,7,10], [2,5,8,11]. Leftover images fill front-to-back.
  - `length_N` - Chunked: sequential groups of N images. With 12 images and N=3: groups are [0,1,2], [3,4,5], [6,7,8], [9,10,11]. Short final groups are padded with a blank filler image.

**Labels:**
- `labels` - Multiline text with the same format as Image Array (one per line or semicolon-separated, `\n` for a line break inside a label). The label list restarts independently for each group. If K labels < group size, trailing slots are blank or repeat the last label depending on `label_end`.
- `label_end`:
  - `blank` - Trailing slots past the end of the label list are empty
  - `repeat_last` - Trailing slots repeat the final label
- `label_location`, `label_size`, `font` - Same as Image Array
- `label_style` - Color scheme, written as text color on label box color (same 4 options as Label Image; default `white on black`)

**Layout & Sizing:**
- `shape` - Grid layout for each stitched group (same options as Image Array, including `force_rows` / `force_cols`; default `horizontal`)
- `smart_ratio` - Custom W:H target, same format as Image Array (`force_rows` / `force_cols` take priority)
- `forced_N` - Same as Image Array; sets the rows or columns of each stitched group when `shape` is `force_rows` or `force_cols` (default 1)
- `background` - Canvas color (`black` / `white`). Also used for blank filler slots, for padding added by `size_method`, and for the gaps from `spacing`.
- `resize_any` - ON: resize the images in each group to one common size before stitching. OFF: keep native sizes.
- `size_guide` - `largest` or `smallest` target when `resize_any` is ON, or `match` to use the size of image number `match_N` in each group
- `match_N` - Only used when `size_guide` is `match` (default 1, max 9999). Which image in each group sets the target size, counting from 1 in group order. If a group has fewer images than this number, the last image in that group is used (blank filler slots are skipped), so a large value such as `9999` always means the last image.
- `size_method` - Same 4 options as Image Array
- `pad_axis` - Only used when `resize_any` is ON. Which dimension is made equal across each group:
  - `both` - Same width and height (default)
  - `height_only` - Same height; each image keeps its own width. Ideal for horizontal stitching of variable-width images.
  - `width_only` - Same width; each image keeps its own height. Ideal for vertical stitching.
- `spacing` - Gap in pixels added around every image, after its label (0-100, default 5), in the background color

**Outputs:**
- `stitched_groups` - List of stitched images, one per group

---

### Random Subset

Picks a random subset of the lines in a block of text. Same seed and settings always give the same picks.

**Parameters:**
- `input_text` - The items to choose from, one per line. Blank lines are ignored.
- `num_to_pick` - How many items to pick. Without replacement this is capped at the number of items.
- `with_replacement` - ON: the same item can be picked more than once, and `num_to_pick` is not capped. OFF: each item is picked at most once (default).
- `random_order` - ON: shuffle the picked items one more time before output (default). OFF: skip that extra shuffle; items come out in the order they were drawn. This is not the original list order, except when every item is picked.
- `string_delimiter` - Text placed between items in `merged_string`. Type `\n` for a new line (default).
- `seed` - Random seed

**Outputs:**
- `string_list` - The picked items as a list, one entry per item
- `merged_string` - All picked items joined into one string with `string_delimiter`
- `pick_indices` - Position of each picked item in the input list, counting from 0 (blank lines are not counted)

---

### Load Images From Folder(s)

Loads images from a folder with filtering, stepping, limiting and sorting. Supported file types: png, jpg, jpeg, gif, tiff, tif, webp, bmp. Images are converted to RGB and rotated according to their EXIF orientation.

The order of operations is: find files, filter, sort with `select_sort`, keep every `every_n`th image, apply `limit`, then re-order with `output_sort`.

**Parameters:**
- `folder` - Root folder to search
- `subdirectories` - ON: also search subfolders of `folder`. OFF: only the folder itself (default).
- `filter` - Text or regex pattern to match against file names. Leave blank to keep every supported image.
- `filter_type` - How `filter` is matched: `contains` (default), `starts with`, `ends with` or `regex`. A regex is searched anywhere in the name, and an invalid pattern matches nothing.
- `filter_case_sensitive` - Whether the filter match is case sensitive (default off)
- `filter_include_extension` - ON: match the filter against the full file name including its extension. OFF: match only the name without extension (default).
- `select_sort` - Order the filtered pool is sorted into before `every_n` and `limit` are applied: name, total pixels, aspect ratio or date modified (each ascending or descending), or `random`
- `every_n` - Keep every Nth image from the filtered and sorted pool, starting with the first. 1 keeps every image, 2 keeps every other image, and so on.
- `limit` - Maximum number of images to keep after `every_n` stepping. 0 means no limit.
- `output_sort` - Order applied to the final selection for output. `none` keeps the order from the `every_n` / `limit` step. Other options are the same as `select_sort` plus date created.
- `seed` - Random seed, used whenever `select_sort` or `output_sort` is `random` (the two shuffles are independent)

**Outputs** (all lists except the two counts):
- `images` - The loaded images. Each can be a different size. If nothing is found, a single 1×1 black image.
- `file_paths` - Full path of each loaded image
- `filenames` - File name with extension
- `filenames_no_ext` - File name without extension
- `subdirs` - Each image's folder relative to the root folder, written with backslashes and wrapped in backslashes: `\` for files directly in the root, `\vacation\` for a subfolder
- `directory` - Full folder path of each image (root plus `subdirs`), ending with a backslash, without the file name. `directory` + `filenames` rebuilds the full path.
- `filtered_count` - How many files matched the filter, before `every_n` and `limit`
- `output_count` - How many images were actually loaded

---

### DuckDuckGo img search

Downloads images from DuckDuckGo image search.

**Requires:** `pip install duckduckgo-search requests`

**Parameters:**
- `query` - Search text
- `start` - How many results to skip from the top (0 starts at the first result)
- `num_results` - How many results to return, counted after skipping `start`
- `output_type`:
  - `list` - Each image is returned separately at its own size (default)
  - `batch` - All images are padded with black to the largest width and height and returned as one batch
- `region`, `safe_search`, `size`, `color`, `type_image`, `layout`, `license_image`, `ai_images` - Search filters. **These are not applied yet.** The current code only sends the query text to DuckDuckGo, so changing them has no effect.

**Outputs:**
- `image(s)` - The downloaded images. An image that fails to download becomes a 64×64 black placeholder, so positions still match `url(s)`.
- `url(s)` - Source URL of each image, in the same order

If the search returns nothing, the node stops with an error.

---

### Local Timer Start and Local Timer End

Measure how long the nodes between them take to run. Each Start/End pair works independently, so you can time several sections of a workflow.

**Local Timer Start:** Place it just before the node(s) you want to time. Wire your workflow value into `passthrough`; it comes out unchanged, together with a `timer` output. The node always re-runs, so the timestamp is fresh each time.

**Local Timer End:** Wire `timer` from Local Timer Start into its `timer` input, and the value coming out of the timed section into `passthrough`.

**Parameters (Local Timer End):**
- `format`:
  - `h m s` - Text like `3h 5m 2s` or `34s` (zero parts omitted) (default)
  - `hh:mm:ss` - Text like `03:05:02`
  - `h`, `m`, `s` - A number of hours, minutes or seconds, rounded to 2 decimals

**Outputs (Local Timer End):**
- `passthrough` - The input value, unchanged
- `time_string` - Elapsed time as text (empty for `h`, `m` and `s`)
- `time_float` - Elapsed time as a number in the unit chosen by `format` (0.0 for the text formats)

---

## Examples
Eventually...

## Requirements

- ComfyUI (latest version recommended)
- Python 3.8+
- Pillow (PIL)
- PyTorch
- `duckduckgo-search` and `requests` (only for the DuckDuckGo img search node)

## Tips

**Image Equalizer:**
- Use `average_edge` padding for seamless borders that match image content
- Use `grow` + `pad` for lossless size normalization
- Use `shrink` + `crop` to create uniform thumbnails
- Use `match` with `match_N` to bring every image to the size of one chosen reference image

**Image Array:**
- Use `smart_square` for photo galleries with mixed portrait/landscape images
- Set `spacing` to 5-10 pixels for clean visual separation
- Place labels on `bottom` or `top` for easier reading
- Use `left_vert` or `right_vert` for space-efficient labeling
- For dynamic labels, connect output from other nodes like counters or text processors
- `smart_ratio` accepts most common formats: `16:9`, `16/9`, `1.78`, `3x1`, `3W x 1H`
- Use `force_cols` with `forced_N` for a fixed-width grid of any length (for example 4 columns), or `force_rows` for a fixed-height grid
- Set `size_guide` to `match` and `match_N` to 9999 to size every image to the last one

**Advanced Stitcher:**
- Use `every_N` grouping to compare corresponding frames across N different generation runs side by side
- Use `pad_axis: height_only` with horizontal stitching when input images have different widths but matching heights. It avoids width padding and keeps each image at its native width
- With `resize_any` OFF, images are placed at native sizes; the canvas grows to fit without padding

**Load Images From Folder(s):**
- Use `every_n` with `select_sort` to sample evenly across a folder, or `select_sort: random` with `limit` for a random sample
- Connect `filenames_no_ext` to `label_input` on Image Array to label each image with its own file name

## Font Setup

For custom fonts:
1. Create a `fonts` directory in your ComfyUI installation root
2. Place TrueType (.ttf) font files in this directory
3. Restart ComfyUI - fonts will appear in the dropdown

If no `fonts` directory exists, the dropdown only offers `arial.ttf`, which is loaded from `C:/Windows/Fonts/Arial.ttf`. If a font cannot be loaded (for example Arial on a non-Windows system), a small built-in font is used and `label_size` has no effect.

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
