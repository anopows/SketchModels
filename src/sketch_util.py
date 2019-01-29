import tensorflow as tf
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# TODO can't use from sketch_io because of circular dependency
def class_names(small=False, escape_spaces=True, base_folder=None):
    # Folders for storage/retrival
    main_directory  = base_folder or '../'
    data_directory  = main_directory + 'kaggle_data/'

    class_name = 'classnames'
    if small: class_name += '_small'
    
    class_names = [] # 345 classes, small: 10 classes
    with open(data_directory + class_name + '.csv', 'r') as cln_file:
        for line in cln_file:
            cl_name = line[:-1]
            if escape_spaces:
                cl_name = cl_name.replace(' ', '_') # escape spaces for kaggle
            class_names.append(cl_name)
    return class_names

def batch_gather(arr, lengths, progress=1.0):
    if progress != 1.0: 
        lengths = tf.cast(lengths, tf.float32)
        lengths = tf.multiply(lengths, progress)
        indices = lengths - 1
        indices = tf.cast(indices, tf.int32)
    else:
        indices = lengths - 1

    indices = tf.maximum(0, indices)
    indices  = tf.stack([tf.range(tf.shape(indices)[0]), indices], axis=1)
    return tf.gather_nd(arr, indices)

def mask_sketch(sketches, lengths):
    def repl(x, max_len):
        ones  = tf.ones(x, dtype=tf.int32)
        zeros = tf.zeros(max_len-x, dtype=tf.int32)
        return tf.concat([ones,zeros], axis=0)

    max_len = tf.shape(sketches)[1]
    mask = tf.map_fn(lambda x: repl(x, max_len), lengths)
    mask_float = tf.cast(mask, tf.float32)
    return tf.expand_dims(mask_float,2) * sketches

def create_snapshot_imgs(length, sketch, num_snapshots=6, 
               imgsize = (255,255), output_size = (224,224), 
               border = (5,5), thickness=6, coloring=True):
    """Create an image from given lines

    Arguments:
    length:  number of non-padded pts
    sketch:  array of pts (x,y,end), where end denotes end of a stroke
    imgsize: creation size of an image (default (255,255))

    Assume point coordinates are between 0 and 255
    """
    assert len(np.array(length).shape)==0 and len(np.array(sketch).shape)==2, "One sample at a time"
    total_len = len(sketch)
    sketch = sketch[:length]
    sketch[-1,2] = 1 # Fix if sketch has no end
    
    def spaced_indices(penups, length, num_to_pick=6):
        all_penups = np.where(penups)[0]
        penup_places = []
        max_index = len(all_penups)-1
        for i in range(num_to_pick-1):
            index = round(i*max_index/(num_to_pick-1))
            penup_places.append(all_penups[index])
        penup_places.append(all_penups[max_index])

        mapping = np.zeros(length, dtype=np.int32)
        np.put(mapping, ind=penup_places, v=1)
        mapping = np.cumsum(mapping, dtype=np.int32)
        return penup_places, mapping # pos --> relevant snapshot image

    penup_places, mapping =  spaced_indices(sketch[:,2], length, num_to_pick=num_snapshots)
    # white images
    imgsizex, imgsizey = (imgsize[0] + 2*border[0], imgsize[1] + 2*border[1])
    imgshape = (imgsizey, imgsizex)    # black and white
    img  = np.zeros(imgshape,                       np.uint8) # for opencv first y-coord, then x-coord
    imgs = np.zeros((num_snapshots,) + output_size, np.uint8) # result
    split_indices = sketch[:,2] == 1 # note where we have 1s
    split_indices = np.where(split_indices)[0][:-1] # convert to indices, drop last one(end)
    split_indices += 1 # split one after end token
    sketch = sketch[:,:2] + np.array(border)
    lines = np.split(sketch[:,:2], split_indices)
    
    counter, cur_len = 0, 0
    for i, line in enumerate(lines):
        cur_len += len(line)
        color = 255 - min(i,12)*15 if coloring else 255
        cv.polylines(img, [line], False, color, thickness, 32)
        
        if cur_len-1 in penup_places: # is this a chosen penup
            # Scale to output image size, also smooths it
            img_transformed = cv.resize(img, output_size, interpolation = cv.INTER_CUBIC)
            imgs[counter]   = img_transformed
            counter   += 1

    return [np.int32(counter), imgs, mapping] # how many images created,  images,  i-th input to j-th image

def create_img(length, sketch, imgmode=None, imgperc=None, 
               imgsize = (255,255), output_size = (224,224), border = (5,5), thickness=6,
               color_after_len=None):
    """Create an image from given lines

    Arguments:
    length:  number of non-padded pts
    sketch:  array of pts (x,y,end), where end denotes end of a stroke
    imgmode: None (whole image), 'atperc' (until imgperc% of image), 'snapshots' (at snapshots)
    imgsize: output size of an image (default (255,255))

    Assume point coordinates are between 0 and 255
    """
    assert len(np.array(length).shape)==0 and len(sketch.shape)==2, "One sample at a time"
    if color_after_len and (imgmode is not None): raise Exception("Not Implemented")

    if imgmode in [None, 'snapshots', 'middle_last']:
        sketch = sketch[:length]
        sketch[-1,2] = 1 # Fix if sketch has no end
    elif imgmode == 'atperc':
        assert imgperc >= 0 and imgperc <= 1
        length = int(imgperc*length - 1)
        sketch = sketch[:length]
        sketch[-1,2] = 1 # last point is end
    else:
        raise Exception("'{}' image mode not implemented".format(imgmode))
        
    # white image
    imgsizex, imgsizey = (imgsize[0] + 2*border[0], imgsize[1] + 2*border[1])
    if color_after_len: imgshape = (imgsizey, imgsizex, 3) # use 3 channels to use colors
    else:               imgshape = (imgsizey, imgsizex)    # black and white
    img = np.zeros(imgshape, np.uint8) # for opencv first y-coord, then x-coord

    split_indices = sketch[:,2] == 1 # note where we have 1s
    split_indices = np.where(split_indices)[0][:-1] # convert to indices, drop last one(end)
    split_indices += 1 # split one after end token
    sketch = sketch[:,:2] + np.array(border)
    lines = np.split(sketch[:,:2], split_indices)
    
    if imgmode == 'snapshots':
        results = []
        for line in lines:
            cv.polylines(img, [line], False, color=255, thickness=thickness, lineType=32)
            # Scale to output image size, also smooths it
            img_transformed = cv.resize(img, output_size, interpolation = cv.INTER_CUBIC)
            results.append(img_transformed)
        imgs = np.stack(results)
        return imgs.reshape((-1, output_size[0]*output_size[1]))
    elif imgmode == 'middle_last':
        imgs = np.empty((2, output_size[0] * output_size[1]), dtype=np.uint8)
        count_els = 0
        over_half = False
        for line in lines: 
            num_els = line.shape[0]
            if (not over_half) and (count_els + num_els >= length // 2): # if half element occurs
                over_half = True
                num_to_take = (length // 2) - count_els
                img_half = img
                cv.polylines(img_half, [line[:num_to_take]], color=255, thickness=thickness, lineType=32)
                # Scale to output image size, also smooths it
                img_half = cv.resize(img_half, output_size, interpolation = cv.INTER_CUBIC)
                imgs[0] = img_half.reshape((224*224))
                
            cv.polylines(img, [line], color=255, thickness=thickness, lineType=32)
            count_els += num_els
        img = cv.resize(img, output_size, interpolation = cv.INTER_CUBIC)
        imgs[1] = img.reshape((224*224))
        return imgs
    elif color_after_len is not None:
        num_pts = 0
        color_mode = False
        for line in lines:
            if color_mode: 
                cv.polylines(img, [line], False, (0,0,255), thickness, 32)
            next_num_pts = num_pts + len(line)

            if next_num_pts > color_after_len:
                pts_old = color_after_len - num_pts
                if pts_old:
                    cv.polylines(img, [line[:pts_old]], False, 255, thickness, 32)
                cv.polylines(img, [line[pts_old:]], False, (0,0,255), thickness, 32)  
                color_mode = True
            else:
                cv.polylines(img, [line], False, 255, thickness, 32) 
        # Scale to output image size, also smooths it
        img = cv.resize(img, output_size, interpolation = cv.INTER_CUBIC)
        return img.reshape(-1)
    else:
        for line in lines:
            cv.polylines(img, [line], False, 255, thickness, 32)

        # Scale to output image size, also smooths it
        img = cv.resize(img, output_size, interpolation = cv.INTER_CUBIC)
        return img.reshape(output_size[0]*output_size[1])

def mov_to_coord(sketch, max_len=None, lengths=None, orig_coord=[122.5, 122.5]):
    def _convert_one_sketch(sketch, orig_coord):
        coords = [np.array(orig_coord)]
        for mov in sketch:
            next = coords[-1] + mov[:2]
            coords.append(next)
        coords = np.array(coords[1:])
        return coords

    if len(sketch.shape) == 2: # just one sketch
        sketch[:,:2] = _convert_one_sketch(sketch, orig_coord)
        return sketch
    elif len(sketch.shape) == 3: # batch of sketches
        assert (max_len is not None) and (lengths is not None)
        list_sketches = []
        for (l,sk) in zip(lengths, sketch):
            padded_sketch        = np.zeros((max_len,3), dtype=np.float32)
            # copy penups
            padded_sketch[:l, 2] = sk[:l,2]
            # convert displacements to coordinates
            padded_sketch[:l,:2] = _convert_one_sketch(sk[:l], orig_coord)
            list_sketches.append(padded_sketch)
        return np.stack(list_sketches, axis=0)
    else: raise Exception

def draw_sketches(lengths, sketches, labels, small=False,
                  plot_title="q sample",
                  orig_sketches=None, orig_length=None,
                  length_half=None): # only 1 sample per batch allowed
    def _normalize(coords):
        mincoords = np.min(coords, axis=0)
        mincoords = np.min([mincoords, [0,0]], axis=0)
        coords -= mincoords
        maxcoords = np.max(coords, axis=0)
        maxcoords = np.max([maxcoords, [255,255]], axis=0) # if below 255, don't do anything else scale down
        coords = (255) * coords / maxcoords
        coords = coords.astype(np.uint8)
        return coords

    assert (length_half is None) or (type(length_half) in [int,np.int64] and len(lengths) == 1), \
        "no batches for half sketches supported. len_half: {} with type {}, len: {} with length {}".format(
            length_half, type(length_half), lengths, len(lengths))
        
    label_names = class_names(small=small, escape_spaces=False)
    names = [label_names[l] if l>=0 else 'unknown' for l in labels]

    fig_rows = 1
    fig_cols = 1 + bool(length_half) + int(orig_sketches is not None)
    
    for i, (length, sketch, name) in enumerate(zip(lengths, sketches, names)):
        fig_i    = 1
        plt.figure(facecolor='white', figsize=(13.5,4.5))
        # Normalize to [0,255] positions
        coords = _normalize(sketch[:,:2])
        # put reconstructed sketch together and create img
        coord_penup = np.concatenate((coords, sketch[:,2:]), axis=1)

        # Optionally add half of the true image
        if length_half: 
            assert orig_sketches is not None
            img_half = create_img(length_half, orig_sketches[i])
            plt.subplot(fig_rows, fig_cols, fig_i)
            fig_i += 1
            plt.title("half sketch: " + name)
            plt.axis('off')
            plt.imshow(img_half.reshape((224,224)), cmap='gray')

        # Print main image
        img_main = create_img(length, coord_penup.astype(np.uint8))
        
        plt.subplot(fig_rows, fig_cols, fig_i)
        fig_i += 1
        plt.title(plot_title + ". Label: " + name)
        plt.axis('off')
        img_main = img_main.reshape((224,224,1))
        img_main = np.tile(img_main, [1,1,3])
        # # Make it blue
        # img_main[:,:,2] = 255
        plt.imshow(img_main)

        # Optionally add true image
        if orig_sketches is not None:
            plt.subplot(fig_rows, fig_cols, fig_i)
            fig_i += 1
            plt.title("drawn sketch: " + name)
            plt.axis('off')
            length = orig_length or length
            img_orig = create_img(length, orig_sketches[i]).reshape((224,224,1))
            img_orig = np.tile(img_orig, [1,1,3])
            # img_orig[:,:,2] = 255
            plt.imshow(img_orig)
