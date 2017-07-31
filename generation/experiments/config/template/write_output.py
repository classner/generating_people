"""Summaries and outputs."""
import os
import os.path as path
from collections import OrderedDict
import logging
from gp_tools.write import append_index


LOGGER = logging.getLogger(__name__)


def save_grid(fetches, image_dir, config, rwgrid, batch):
    index_path = os.path.join(path.dirname(image_dir), "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        index.write("<th>Grid</th>")
        for col_idx in range(rwgrid.gridspec[1]):
            index.write("<th>%d</th>" % col_idx)
        index.write("</tr>\n")
    outputs = fetches['outputs']
    for sample_idx in range(config["batch_size"]):
        y_pos = ((batch * config["batch_size"] + sample_idx) //
                 rwgrid.gridspec[1])
        x_pos = ((batch * config["batch_size"] + sample_idx) %
                 rwgrid.gridspec[1])
        if y_pos >= rwgrid.gridspec[0]:
            break
        filename = "%08d-%08d.png" % (y_pos, x_pos)
        out_path = os.path.join(image_dir, filename)
        with open(out_path, "w") as f:
            f.write(outputs[sample_idx])
        if x_pos == 0:
            index.write("<tr><td>%d</td>" % (y_pos))
        index.write('<td><img src="images/%s"></td>' % (filename))
        if x_pos == rwgrid.gridspec[1] - 1:
            index.write("</tr>\n")
    return index_path


warned_y = False
warned_z = False


def save_images(fetches, image_dir, mode, config, step=None, batch=0):
    global warned_y, warned_z
    image_dir = path.join(image_dir, 'images')
    if not path.exists(image_dir):
        os.makedirs(image_dir)
    row_infos = []
    for im_idx in range(config["batch_size"]):
        if step is not None:
            row_info = OrderedDict([('step', (str(step), 'text')), ])
        else:
            row_info = OrderedDict()
        if mode in ['train', 'trainval', 'val', 'test', 'transform']:
            in_path = fetches["paths"][im_idx]
            name, _ = os.path.splitext(os.path.basename(in_path))
        elif mode == 'sample':
            name = str(config["batch_size"] * batch + im_idx)
            if 'paths' in fetches.keys():
                in_path = fetches["paths"][im_idx]
                fname, _ = os.path.splitext(os.path.basename(in_path))
                name += '-' + fname
        if step is not None:
            name = str(step) + '_' + name
        row_info["name"] = (name, 'text')
        if 'inputs' in fetches.keys():
            row_info["inputs"] = (fetches['inputs'][im_idx], 'image')
        if 'conditioning' in fetches.keys():
            row_info["conditioning"] = (fetches["conditioning"][im_idx],
                                        'image')
        if 'outputs' in fetches.keys():
            row_info["outputs"] = (fetches["outputs"][im_idx], 'image')
        if 'targets' in fetches.keys():
            row_info["targets"] = (fetches["targets"][im_idx], 'image')
        if 'y' in fetches.keys():
            try:
                row_info["y"] = (fetches["y"][im_idx], 'plain')
            except:
                if not warned_y:
                    LOGGER.warn("Not sufficient info for storing y!")
                    warned_y = True
        if 'z' in fetches.keys():
            try:
                row_info["z"] = (fetches["z"][im_idx], 'plain')
            except:
                if not warned_z:
                    LOGGER.warn("Not sufficient info for storing z!")
                    warned_z = True
        row_infos.append(row_info)
        LOGGER.debug("Processed image %d.",
                     batch * config["batch_size"] + im_idx + 1)
    index_fp = append_index(row_infos, image_dir, mode)
    return index_fp
