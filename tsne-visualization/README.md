# Visualizaion Tool for 2D T-SNE with Images
## Requirements
bokeh, matplotlib
## Usage
1. Put your dataset (use link if it is too large) under `bokeh_vis/static`.
2. Put the csv that contains the information of your 2D plot under `bokeh_vis/data`.
3. Run `bokeh.sh`
## Format of the csv file
It must contains at least 4 columns: `tsne_y`, `tsne_x`, `plot_id`, `image_path`

* The `tsne_x` and `tsne_y` are the coordinate of the point
* The `plot_id` is the class of the point (could be string or number)
* The `image_path` is the path to the image. The root of the path is bokeh_vis/static. For example if you have a folder named `dataset` under the `bokeh_vis/static`. Then the image_path could be `./dataset/image_1.png` 
