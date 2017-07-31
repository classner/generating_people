import numpy as np
import os.path as path


def append_index(row_infos, image_dir, mode):
    """Append or create the presentation html file for the images."""
    index_path = path.join(path.dirname(image_dir), "index.html")
    if path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html>\n<body>\n<table>\n<tr>")
        colnames = [key for key, val in row_infos[0].items()
                    if val[1] in ['image', 'text']]
        for coln in colnames:
            index.write("<th>%s</th>\n" % (coln))
        index.write("</tr>\n")
    for row_info in row_infos:
        index.write("<tr>\n")
        for coln, (colc, colt) in row_info.items():
            if colt == 'text':
                index.write("<td>%s</td>" % (colc))
            elif colt == 'image':
                filename = path.join(image_dir,
                                     row_info['name'][0] + '_' + coln + '.png')
                with open(filename, 'w') as outf:
                    outf.write(colc)
                index.write("<td><img src='images/%s'></td>" % (
                    path.basename(filename)))
            elif colt == 'plain':
                filename = path.join(image_dir,
                                     row_info['name'][0] + '_' + coln + '.npy')
                np.save(filename, colc)
            else:
                raise Exception("Unsupported mode: %s." % (mode))
        index.write("</tr>\n")
    return index_path
