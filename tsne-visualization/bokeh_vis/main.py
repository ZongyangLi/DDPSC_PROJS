
from bokeh.io import output_notebook
from bokeh.plotting import figure, curdoc, show, ColumnDataSource
from bokeh.models import HoverTool, TapTool, CrosshairTool, Circle
from bokeh.transform import factor_cmap, linear_cmap
from bokeh.palettes import Spectral6, Plasma256, inferno, cividis
from bokeh.layouts import row, widgetbox
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource, HoverTool, PanTool, ResetTool,BoxZoomTool, ColorBar, CustomJS
from bokeh.models.widgets import Button, Slider, TextInput, RadioButtonGroup, Dropdown, Select, PreText
from helpers import load_csv
import os
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np

def big_palette(size, palette_func): 
    if size < 256: 
        return palette_func(size) 
    p = palette_func(256) 
    out = [] 
    for i in range(size):
        idx = int(i * 256.0 / size) 
        out.append(p[idx])
    return out

# load data
def init_data():
    pass 
color_cls = 'cls_id'
data_root_dir = 'bokeh_vis/data'
data_filename_list = os.listdir(data_root_dir)

default_path = os.path.join(data_root_dir,'show.csv')
data_dict = load_csv(default_path)
data_dict['x'] = data_dict.pop('tsne_x') 
data_dict['y'] = data_dict.pop('tsne_y') 
features_list = list(data_dict.keys())
features_list.remove('x')
features_list.remove('y')
if 'image_path' in features_list:
    features_list.remove('image_path')
source = ColumnDataSource(data={
    'x' : list(map(float, data_dict['x'])),
    'y' : list(map(float, data_dict['y'])),
    'cls_id' : data_dict['plot_id'],
    'imgs': data_dict['image_path']})
# source = ColumnDataSource(data=data_dict)
print(source.data.keys())
cls_list = list(set(source.data[color_cls]))
cls_grp = dict()
for i, point in enumerate(source.data[color_cls]):
    if point not in cls_grp.keys():
        cls_grp[point] = [i]
    else:
        cls_grp[point].append(i)

p = figure(plot_width=1000, plot_height=1000, tools=['pan', 'box_zoom', 'reset'], title='')
workaround_palette = big_palette(len(cls_list), cividis)
# cls_color_mapper = factor_cmap(color_cls, palette=inferno(len(cls_list)), factors=cls_list, nan_color=(0, 0, 0, 0))
# lin_color_mapper = linear_cmap(color_cls, palette=inferno(len(cls_list)), low=0, high=len(cls_list), nan_color=(0, 0, 0, 0))
mpl_colormap = cm.get_cmap('jet')
jet_palette = [mpl.colors.rgb2hex(m) for m in mpl_colormap(np.linspace(0,255,len(cls_list)).astype(int))]
cls_color_mapper = factor_cmap(color_cls, palette=jet_palette, factors=cls_list, nan_color=(0, 0, 0, 0))
# lin_color_mapper = linear_cmap(color_cls, palette=inferno(256), low=0, high=len(cls_list), nan_color=(0, 0, 0, 0))
lin_color_mapper = linear_cmap(color_cls, palette=jet_palette, low=0, high=len(cls_list), nan_color=(0, 0, 0, 0))
color_bar = ColorBar(color_mapper=lin_color_mapper['transform'], label_standoff=12, border_line_color=None, location=(0,0))
#color_mapper = 'green'
p.add_layout(color_bar, 'right')
cr = p.circle('x', 'y', color=cls_color_mapper, size=10, alpha=0.5, hover_alpha=1.0, source=source)
hsource = ColumnDataSource({'x': [], 'y': [], color_cls: []})
hcr = p.circle('x', 'y', color=cls_color_mapper, alpha=1, size=10, line_width= 3, line_color='black', source=hsource )
# Add a hover tool, that sets the link data for a hovered circle
base_code = """
var color_cls = '%s'
var cls_id = data.data[color_cls]
var data = {'x': [], 'y': [], 'cls_id': []};
var cdata = circle.data;
var indices = cb_data.index['1d'].indices;
for (var i = 0; i < indices.length; i++) {
    var ind0 = indices[i]
    var cls = cls_id[ind0]
    data['x'].push(cdata.x[ind0]);
    data['y'].push(cdata.y[ind0]);
    data['cls_id'].push(cdata.cls_id[ind0]);
    for (var j = 0; j < cls_grp[cls].length; j++) {
        var ind1 = cls_grp[cls][j];
        data['x'].push(cdata.x[ind1]);
        data['y'].push(cdata.y[ind1]);
        data['cls_id'].push(cdata.cls_id[ind1]);
        
    }
}
highlight.data = data;
""" 
code = base_code%  color_cls
TOOLTIPS = """
    <div>
        <div>
            <img
                src="@imgs" height="100" alt="@imgs" width="100"
                style="float: left; margin: 0px 15px 15px 0px;"
                border="2"
            ></img>
        </div>
        <div>
            <span style="font-size: 17px; font-weight: bold; color: #966;">@cls_id</span>
        </div>
        <div>
            <span style="font-size: 10px; color: #696;">($x, $y)</span>
        </div>
    </div>
""" 
callback = CustomJS(args={'data': source, 'circle': cr.data_source, 'highlight': hcr.data_source, 'cls_grp': cls_grp}, code=code)
hover = HoverTool(tooltips=TOOLTIPS, callback=callback, renderers=[cr])
p.add_tools(hover)
#p.add_tools(TapTool(callback=callback, renderers=[cr]))
filelist_refresh_button = Button(label="Refresh data list", button_type="success")
menu = [("1.csv", "item_1"), ("2.csv", "item_2")]
data_dropdown = Dropdown(label="Data file", button_type="warning", menu=menu)
color_select_text = PreText(text='Color mapped by: ')
color_select_rbButton = RadioButtonGroup(labels=["classes", "linear"], active=0)
col_select = Select(title="Colored by:", value=features_list[0], options=features_list[:])
highlight_select = Select(title="Highlighted by:", value=features_list[0], options=features_list[:])
def update_colormap(i):
    if i==0:
        print('cls color map')
        cls_list = list(set(source.data[color_cls]))
        workaround_palette = big_palette(len(cls_list), cividis)
        jet_palette = [mpl.colors.rgb2hex(m) for m in mpl_colormap(np.linspace(0,255,len(cls_list)).astype(int))]
        c_mapper = factor_cmap(color_cls, palette=jet_palette, factors=cls_list)
        cr.glyph.fill_color = c_mapper
        hcr.glyph.fill_color = c_mapper
    elif i==1:
        print('linear color map')
        cls_list = list(set(source.data[color_cls]))
        low_bound = min(list(map(float, cls_list)))
        high_bound = max(list(map(float, cls_list)))
        print('[low, high]=',[low_bound, high_bound])
        jet_palette = [mpl.colors.rgb2hex(m) for m in mpl_colormap(np.arange(mpl_colormap.N))]
        c_mapper = linear_cmap(color_cls, palette=jet_palette, low=low_bound, high=high_bound, nan_color=(0, 0, 0, 0))
        cr.glyph.fill_color = c_mapper
        hcr.glyph.fill_color = c_mapper
        color_bar.color_mapper = c_mapper['transform']
def update_colored_cls(attr, old, new):
    print(new)
    source.data['cls_id']=data_dict[new]
    cls_list = list(set(source.data[color_cls]))
    cls_grp = dict()
    for i, point in enumerate(source.data[color_cls]):
        if point not in cls_grp.keys():
            cls_grp[point] = [i]
        else:
            cls_grp[point].append(i)
    jet_palette = [mpl.colors.rgb2hex(m) for m in mpl_colormap(np.linspace(0,255,len(cls_list)).astype(int))]       
    workaround_palette = big_palette(len(cls_list), cividis)
    
    c_mapper = factor_cmap(color_cls, palette=jet_palette, factors=cls_list, nan_color=(0, 0, 0, 0))
    cr.glyph.fill_color = c_mapper
    hcr.glyph.fill_color = c_mapper
    hover.callback.args={'data': source, 'circle': cr.data_source, 'highlight': hcr.data_source, 'cls_grp': cls_grp}
    color_select_rbButton.active=0
    p.title.text=new
def update_file_list():
    csv_list = os.listdir('bokeh_vis/data/')
    print(csv_list)
    menu = []
    for i, filename in enumerate(csv_list):
        if filename[-4:] == '.csv':
            menu.append((filename, 'item_'+str(i)))
    data_dropdown.menu=menu
update_file_list()
filelist_refresh_button.on_click(update_file_list)
color_select_rbButton.on_click(update_colormap)
col_select.on_change('value', update_colored_cls)
highlight_select.on_change('value', update_colored_cls)
inputs = widgetbox(filelist_refresh_button, 
                   data_dropdown, 
                   color_select_text, 
                   color_select_rbButton, 
                   col_select) 
                   #highlight_select)
# put the button and plot in a layout and add to the document
curdoc().add_root(row(inputs, p))
