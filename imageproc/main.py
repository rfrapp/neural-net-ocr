
import collections, copy, math, numpy, sys
numpy.set_printoptions(threshold=numpy.nan)
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.mlab as mlab
import matplotlib.patches as patches
import matplotlib.path
import scipy.misc
import skimage.transform, skimage.draw
import skimage.morphology, skimage.filters, skimage.draw, skimage.feature
import cv2

sys.path.append('..')
import NeuralNetwork
import create_mat
import imageproc.blob as blob

def show_hold():
    plt.show(block=False)
    s = input()
    plt.close()


def display_image(imgdata, show=True):
    fig = plt.figure()

    ax0 = fig.add_subplot(111)
    # ax0.axis('off')
    ax0.imshow(imgdata, cmap='gray', origin=None)

    if show:
        plt.tight_layout()
        return show_hold()
    return ax0


def read_image_grayscale(filename):
    return scipy.misc.imread(name=filename, flatten=True)


def invert_graylevel(imgdata):
    return (255 - imgdata) / 255


def resize(imgdata, w, h):
    return scipy.misc.imresize(imgdata, (w, h))


def otsu(imgdata, brute_force=True):
    im_ndarray = imgdata
    imgdata = imgdata.tolist()
    P = collections.defaultdict(float)

    gray_levels = range(0, 256)
    gray_max = int(max(gray_levels))

    N = len(imgdata) * len(imgdata[0])

    for level in gray_levels:
        P[level] = sum([float(row.count(level)) for row in imgdata]) / N

    def bruteforce():
        min_val = 99999
        min_t = 256

        for t in range(1, 255):

            # Compute the probability of each class.
            w0 = sum([P[k] for k in range(1, t + 1)])
            w1 = 1 - w0

            if w0 == 0:
                continue

            # Compute class weighted means.
            mu0 = sum([i * P[i] for i in range(1, t + 1)])
            mu0 /= w0

            mu1 = sum([i * P[i] for i in range(t + 1, gray_max + 1)])
            mu1 /= w1

            # Compute each class' variance.
            sigma0 = sum([(i - mu0) ** 2 * P[i] for i in range(1, t - 1)]) / w0
            sigma1 = sum(
                    [(i - mu1) ** 2 * P[i] for i in range(t + 1, gray_max + 1)]
                ) / w1

            # Compute the weighted within-class variance.
            sigmaw = w0 * sigma0 + w1 * sigma1

            if sigmaw < min_val:
                min_val = sigmaw
                min_t = t

        return min_t

    if brute_force:
        t = bruteforce()

    rows, cols = im_ndarray.shape

    for i in range(rows):
        for j in range(cols):
            if im_ndarray[(i, j)] <= t:
                im_ndarray[(i, j)] = 0
            else:
                im_ndarray[(i, j)] = 255

    return im_ndarray


def detect_edges_sobel(imgdata, show_diagram=False):
    kernel_x = numpy.matrix('-1 0 1; -2 0 2; -1 0 1')
    kernel_y = numpy.matrix('-1 -2 -1; 0 0 0; 1 2 1')

    g_x = numpy.matrix(numpy.zeros(imgdata.shape))
    g_y = numpy.matrix(numpy.zeros(imgdata.shape))
    rows, cols = imgdata.shape

    # Compute g_x
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            dot = imgdata[(i - 1, j - 1)] * kernel_x[(0, 0)]
            dot += imgdata[(i - 1, j + 1)] * kernel_x[(0, 2)]
            dot += imgdata[(i, j - 1)] * kernel_x[(1, 0)]
            dot += imgdata[(i, j + 1)] * kernel_x[(1, 2)]
            dot += imgdata[(i + 1, j - 1)] * kernel_x[(2, 0)]
            dot += imgdata[(i + 1, j + 1)] * kernel_x[(2, 2)]
            g_x[(i, j)] = dot

    # Compute g_y
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            dot = imgdata[(i - 1, j - 1)] * kernel_y[(0, 0)]
            dot += imgdata[(i - 1, j)] * kernel_y[(0, 1)]
            dot += imgdata[(i - 1, j + 1)] * kernel_y[(0, 2)]
            dot += imgdata[(i + 1, j - 1)] * kernel_y[(2, 0)]
            dot += imgdata[(i + 1, j)] * kernel_y[(2, 1)]
            dot += imgdata[(i + 1, j + 1)] * kernel_y[(2, 2)]
            g_y[(i, j)] = dot

    g = numpy.sqrt(numpy.multiply(g_x, g_x) + numpy.multiply(g_y, g_y))
    scipy.misc.imsave("output.jpg", g)
    og = otsu(copy.deepcopy(g))
    scipy.misc.imsave("thresh.jpg", og)

    if show_diagram:
        fig = plt.figure()

        ax0 = fig.add_subplot(221)
        ax0.axis('off')
        ax0.imshow(imgdata, cmap='gray')


        ax1 = fig.add_subplot(222)
        ax1.axis('off')
        ax1.imshow(g_x, cmap='gray')

        ax2 = fig.add_subplot(223)
        ax2.axis('off')
        ax2.imshow(g_y, cmap='gray')

        ax3 = fig.add_subplot(224)
        ax3.axis('off')
        ax3.imshow(g, cmap='gray')

        plt.tight_layout()
        plt.show(block=False)
        s = input("Press Enter to close...")
        plt.close()

        fig = plt.figure()

        ax0 = fig.add_subplot(221)
        ax0.axis('off')
        ax0.imshow(g, cmap='gray')


        ax1 = fig.add_subplot(222)
        ax1.axis('off')
        ax1.imshow(otsu(g), cmap='gray')

        show_hold()

    return g_x, g_y, g


def abs(x):
    return x if x > 0 else -x


def detect_blobs(imgdata, dist_thresh, area_thresh, point_thresh, aspect_thresh):
    blobs = []
    img_h, img_w = imgdata.shape
    ys, xs = imgdata.nonzero()

    for x, y in zip(xs, ys):
        pixel = imgdata[(y, x)]

        if len(blobs) == 0:
            b = blob.Blob(x, y)
            blobs.append(b)
        else:
            min_i = 0
            min_dist = 1000000.
            for i, b in enumerate(blobs):
                d = b.dist_to(x, y)
                if d < min_dist:
                    min_dist = d
                    min_i = i

            if min_dist <= dist_thresh:
                # old_maxx = blobs[min_i].maxx
                blobs[min_i].add_point(x, y)
                # cb = blobs[min_i]
                # if cb.maxx > old_maxx:
                #     # Scan for a line of all 0s inbetween old_maxx and maxx.
                #     split_j = -1

                #     for j in range(old_maxx, cb.maxx + 1):
                #         if sum(imgdata[cb.miny:cb.maxy + 1, j]) == 0:
                #             split_j = j
                #             break
                #     if split_j != -1:
                #         blobs[min_i].maxx = split_j - 1
                #         blobs[min_i].num_points -= 1
                #         b = blob.Blob(x, y)
                #         blobs.append(b)
            else:
                b = blob.Blob(x, y)
                blobs.append(b)

    good_blobs = []
    for b in blobs:
        if b.minx == 0 or b.maxx == img_w or b.miny == 0 or b.maxy == img_h:
            continue
        if b.num_points < point_thresh:
            continue
        if b.aspect < aspect_thresh:
            continue
        if b.area >= area_thresh:
            good_blobs.append(b)

    return good_blobs


def find_bounding_box(imgdata, bg_level, color_thresh, dist_thresh):
    m, n = imgdata.shape
    x0 = y0 = 0
    x1 = y1 = 0
    found_blob = False

    for j in range(n):
        for i in range(m):
            pixel = imgdata[(i, j)]
            cx = (x0 + x1) / 2
            cy = (y0 + y1) / 2

            if abs(bg_level - pixel) >= color_thresh:
                d = math.sqrt((cy - i) ** 2 + (cx - j) ** 2)

                if d <= dist_thresh:
                    if found_blob:
                        x0 = min(x0, j)
                        x1 = max(x1, j)
                        y0 = min(y0, i)
                        y1 = max(y1, i)
                    else:
                        x0 = x1 = j
                        y0 = y1 = i
                        found_blob = True

    return x0, y0, x1, y1


def translate_region(imgdata, x0, y0, x1, y1, dx, dy):
    m, n = imgdata.shape
    new_img = numpy.matrix(numpy.zeros((m, n)))

    for i in range(y0, y1 + 1):
        for j in range(x0, x1 + 1):
            new_img[(i + dy, j + dx)] = imgdata[(i, j)]
    return new_img


def translate_object_to_center(imgdata, bg_level, color_thresh, dist_thresh):
    x0, y0, x1, y1 = find_bounding_box(imgdata, bg_level, color_thresh,
                                       dist_thresh)
    img_cx = imgdata.shape[1] / 2
    img_cy = imgdata.shape[0] / 2
    rect_cx = (x1 + x0) / 2
    rect_cy = (y1 + y0) / 2
    return translate_region(imgdata, x0, y0, x1, y1, int(img_cx - rect_cx),
                            int(img_cy - rect_cy))


def preprocess_nist_img(imgdata, resize_w, resize_h, do_resize=True):
    imgdata = invert_graylevel(imgdata)
    if do_resize:
        imgdata = resize(imgdata, resize_w, resize_h)
    img_h, img_w = imgdata.shape
    diag_dist = math.sqrt(img_h ** 2 + img_w ** 2)
    imgdata = translate_object_to_center(imgdata, 0.0, 0.01, diag_dist)
    imgdata = otsu(imgdata * 255) / 255.
    skimage.morphology.binary_closing(imgdata)
    return imgdata


def detect_lines(img, show=True):
    lines = []

    # Classic straight-line Hough transform
    h, theta, d = skimage.transform.hough_line(img)

    for _, angle, dist in zip(*skimage.transform.hough_line_peaks(h, theta, d, )):
        y0 = dist / numpy.sin(angle)
        y1 = (dist - img.shape[1] * numpy.cos(angle)) / numpy.sin(angle)
        lines.append((0, y0, img.shape[1], y1))

    if show:
        # Generating figure 1
        fig, axes = plt.subplots(1, 3, figsize=(15, 6),
                                 subplot_kw={'adjustable': 'box-forced'})
        ax = axes.ravel()

        ax[0].imshow(img, cmap=cm.gray)
        ax[0].set_title('Input image')
        ax[0].set_axis_off()

        ax[1].imshow(numpy.log(1 + h),
                     extent=[numpy.rad2deg(theta[-1]), numpy.rad2deg(theta[0]), d[-1], d[0]],
                     cmap=cm.gray, aspect=1/1.5)
        ax[1].set_title('Hough transform')
        ax[1].set_xlabel('Angles (degrees)')
        ax[1].set_ylabel('Distance (pixels)')
        ax[1].axis('image')

        ax[2].imshow(img, cmap=cm.gray)
        for x0, y0, x1, y1 in lines:
            ax[2].plot((x0, x1), (y0, y1), '-r')
        ax[2].set_xlim((0, img.shape[1]))
        ax[2].set_ylim((img.shape[0], 0))
        ax[2].set_axis_off()
        ax[2].set_title('Detected lines')

        plt.tight_layout()
        show_hold()

    return lines


def segment_slope_intercept(line):
    x0, y0, x1, y1 = line
    m0 = (y1 - y0) / (x1 - x0)
    b0 = y0 - m0 * x0
    return m0, b0


def intersection_between(line0, line1, img_w, img_h):
    m0, b0 = segment_slope_intercept(line0)
    m1, b1 = segment_slope_intercept(line1)
    # Lines are the same slope.
    if m0 == m1:
        return None
    intersect_x = (b1 - b0) / (m0 - m1)
    intersect_y = intersect_x * m0 + b0

    if intersect_x < 0 or intersect_x > img_w or \
       intersect_y < 0 or intersect_y > img_h:
       return None


    return math.floor(intersect_x), math.floor(intersect_y)


def line_index_intersections(line, lines, img_w, img_h):
    intersections = []
    intersection_indices = []

    for i, line0 in enumerate(lines):
        if line0 != line:
            intersection = intersection_between(line, line0, img_w, img_h)
            if intersection:
                intersections.append(intersection)
                intersection_indices.append(i)

    return intersection_indices, intersections


def adjacent_pairs(arr):
    arr = list(arr)
    pairs = []
    for i in range(len(arr) - 1):
        pairs.append((arr[i], arr[i + 1]))
    return pairs


def quad_area(quad):
    p0, p1, p2, p3 = quad
    p2, p3 = p3, p2
    a0 = p0[0] * p1[1] + p1[0] * p2[1] + p2[0] * p3[1] + p3[0] * p0[1]
    a1 = p0[1] * p1[0] + p1[1] * p2[0] + p2[1] * p3[0] + p3[1] * p0[0]
    return abs(a0 - a1) / 2


def find_quads(lines, img_w, img_h, area_thresh):
    ints = {}
    quads = []
    quad_set = set()

    for line in lines:
        ints[line] = line_index_intersections(line, lines, img_w, img_h)

    op1 = None
    op2 = None

    paired_with = collections.defaultdict(set)

    for i, line0 in enumerate(lines):
        # Get the indices of the lines intersecting with line0.
        indices0, ip0 = ints[line0]

        # A line can only be a part of a quadrilateral if it has at least
        # two lines intersecting with it.
        if len(indices0) >= 2:
            # Search for another line which has at least two of the same
            # intersections. Create quadrilaterals with the intersection
            # points from line0, line1, and each adjacent pair of lines'
            # intersection points to line0 and line1.
            for j, line1 in enumerate(lines):
                if paired_with[j] is not None and i in paired_with[j]:
                    continue
                if i != j and \
                   intersection_between(line0, line1, img_w, img_h) is None:
                    indices1, ip1 = ints[line1]
                    shared = set(indices0) & set(indices1)
                    # print("%d, %d: %s" % (i, j, str(shared)))
                    num_paired = 0

                    if len(shared) >= 2:
                        pairs = adjacent_pairs(shared)

                        for (line2_i, line3_i) in pairs:
                            q0 = intersection_between(line0, lines[line2_i],
                                                      img_w, img_h)
                            q1 = intersection_between(line0, lines[line3_i],
                                                      img_w, img_h)
                            q2 = intersection_between(line1, lines[line2_i],
                                                      img_w, img_h)
                            q3 = intersection_between(line1, lines[line3_i],
                                                      img_w, img_h)

                            quad = (q0, q1, q2, q3)


                            if quad not in quad_set:
                                area = quad_area(quad)

                                # Skip quads which are less than a certain area,
                                # but add it to the fringe set so that it's not
                                # checked again.
                                if area < area_thresh:
                                    quad_set.add(quad)
                                    continue

                                quads.append(quad)
                                quad_set.add(quad)
                                num_paired += 1

                    if num_paired > 0:
                        paired_with[i].add(j)

    return quads


def path_for_quad(quad):
    p0, p1, p2, p3 = quad
    verts = [p0, p1, p3, p2, p0]
    codes = [matplotlib.path.Path.MOVETO,
             matplotlib.path.Path.LINETO,
             matplotlib.path.Path.LINETO,
             matplotlib.path.Path.LINETO,
             matplotlib.path.Path.LINETO
             ]
    path = matplotlib.path.Path(verts, codes)
    return path, verts


def point_in_quad(x, y, quad):
    path, _ = path_for_quad(quad)
    return path.contains_point((x, y))


def pixel_sum_in_quad(image, quad, thresh=200):
    h, w = image.shape
    quad_sum = 0.0
    # print("my quad:", quad)
    for y in range(h):
        for x in range(w):
            p = image[(y, x)]
            factor = 1.
            if p < thresh:
                factor = -1.
            quad_sum += factor * image[(y, x)] * point_in_quad(x, y, quad)
    return quad_sum


def point_on_line(x, y, p0, p1):
    line = p0[0], p0[1], p1[0], p1[1]
    if p0[0] != p1[0]:
        m, b = segment_slope_intercept(line)
        x0, y0, x1, y1 = line
        minx = min(x0, x1)
        maxx = max(x0, x1)
        miny = min(y0, y1)
        maxy = max(y0, y1)
        return (m * x + b) == y and (minx <= x <= maxx) and (miny <= y <= maxy)
    return False


def point_on_quad_perimeter(x, y, quad):
    p0, p1, p2, p3 = quad
    return point_on_line(x, y, p0, p1) or point_on_line(x, y, p1, p3) or \
           point_on_line(x, y, p3, p2) or point_on_line(x, y, p2, p0)


def draw_quad_over_image(image, quad):
    path, verts = path_for_quad(quad)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    patch = patches.PathPatch(path, facecolor='none', lw=2)
    ax.imshow(image, cmap='gray')
    ax.add_patch(patch)

    xs, ys = zip(*verts)
    ax.plot(xs, ys, '--', lw=2, color='green', ms=10)
    show_hold()


def find_document(image, edge_image, quads, show=False):
    max_ratio = 0.0
    max_area = 0.0
    max_quad = None
    h, w = image.shape

    for quad in quads:
        p0, p1, p2, p3 = quad
        points = []
        rr, cc = skimage.draw.line(p0[1], p0[0], p1[1], p1[0])
        points.extend(zip(rr, cc))
        rr, cc = skimage.draw.line(p1[1], p1[0], p3[1], p3[0])
        points.extend(zip(rr, cc))
        rr, cc = skimage.draw.line(p3[1], p3[0], p2[1], p2[0])
        points.extend(zip(rr, cc))
        rr, cc = skimage.draw.line(p2[1], p2[0], p0[1], p0[0])
        points.extend(zip(rr, cc))

        edge_pixels = 0
        non_edge = 0

        for (y, x) in points:
            if edge_image[(y, x)] > 0:
                edge_pixels += 1
            else:
                non_edge += 1
        edge_ratio = edge_pixels / (edge_pixels + non_edge)
        area = quad_area(quad)
        if edge_ratio > max_ratio and area > max_area:
            max_ratio = edge_ratio
            max_quad = quad
            max_area = area
        # =====================================================================

    if show and max_quad:
        draw_quad_over_image(image, max_quad)

    return max_quad


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = numpy.zeros((4, 2), dtype = "float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[numpy.argmin(s)]
    rect[2] = pts[numpy.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = numpy.diff(pts, axis = 1)
    rect[1] = pts[numpy.argmin(diff)]
    rect[3] = pts[numpy.argmax(diff)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unumpyack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = numpy.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = numpy.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = numpy.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = numpy.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = numpy.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped



def group_blobs_ltor_utod(blobs, line_thresh):
    blobs = sorted(blobs, key=lambda a: a.miny)
    max_h = max(blobs, key=lambda a: a.h)
    groups = []
    group = []
    min_cx = 0.0
    min_cy = 0.0

    for blob in blobs:
        if len(group) == 0:
            group.append(blob)
            min_cx = blob.cx
            min_cy = blob.cy
        else:
            if abs(min_cy - blob.cy) <= line_thresh:
                group.append(blob)
                min_cx = min(blob.cx, min_cx)
                min_cy = min(blob.cy, min_cy)
            else:
                groups.append(group)
                group = []
                group.append(blob)
                min_cx = blob.cx
                min_cy = blob.cy

    if len(group) > 0:
        groups.append(group)

    for i, group in enumerate(groups):
        groups[i] = sorted(group, key=lambda a: a.minx)

    return groups


def get_nn():
    num_inputs = create_mat.COL_SIZE
    output_size = len(create_mat.LETTERS)
    hidden_size = 100
    nn = NeuralNetwork.NeuralNetwork(input_size=num_inputs, output_size=output_size)
    weights = scipy.io.loadmat('../learned_weights_shift_sgd_reg8.mat')
    Theta1 = numpy.matrix(weights['w1'])
    Theta2 = numpy.matrix(weights['w2'])
    nn.append_layer(rows=hidden_size, cols=num_inputs + 1,
                    is_output=False, weights=Theta1)
    nn.append_layer(rows=output_size, cols=hidden_size + 1, is_output=True,
                    weights=Theta2)
    return nn


def pad_img(imgdata, pad_size):
    h, w = imgdata.shape
    d = max(h, w) + pad_size
    new_img = numpy.ones((d, d))
    new_img[:h, :w] = imgdata
    return new_img


def get_document_rect(imgdata, doc_area_factor, show_edges, show_quad):
    print("Detecting edges...")
    g = skimage.filters.sobel(imgdata)
    thresh = skimage.filters.threshold_otsu(g)
    g = g > thresh
    print("Finding straight lines...")
    lines = detect_lines(g, show_edges)

    h, w = g.shape
    print("Finding quads...")
    quads = find_quads(lines, w, h, (h * w) / doc_area_factor)

    print("Finding document...")
    doc_quad = find_document(imgdata, g, quads, show_quad)
    print("Doc:", doc_quad)
    return doc_quad


def scan_document(imgdata, doc_quad):
    print("Scanning...")
    scanned = four_point_transform(imgdata, numpy.array(doc_quad))
    thresh = skimage.filters.threshold_local(scanned, 251, offset=.1)
    scanned = scanned > thresh
    scanned = scanned.astype("uint8")
    scanned = skimage.filters.median(scanned)
    scanned = skimage.morphology.opening(scanned)
    return scanned


def get_image_text(imgdata,
                   show_img=False, show_edges=False, show_quad=False,
                   show_chars=False, show_blobs=False,
                   doc_area_factor=4,
                   blob_dist_factor=20,
                   blob_area_factor=10000,
                   blob_num_pixels=40,
                   blob_aspect=0.1,
                   line_thresh=10.):
    if show_img:
        display_image(imgdata)

    imgdata /= 255
    doc_quad = get_document_rect(imgdata, doc_area_factor,
                                 show_edges, show_quad)

    if doc_quad is None:
        return

    scanned = scan_document(imgdata, doc_quad)

    print("Detecting characters...")
    h, w = scanned.shape
    blobs = detect_blobs(1 - scanned, w / blob_dist_factor,
                         (h * w) / blob_area_factor, blob_num_pixels,
                         blob_aspect)
    groups = group_blobs_ltor_utod(blobs, line_thresh)

    # Create a neural network for character recognition.
    nn = get_nn()

    read_str = ""

    for group in groups:
        for blob in group:
            # Grap the image of each character.
            char_img = scanned[blob.miny:blob.maxy + 1, blob.minx:blob.maxx + 1]
            char_img = resize(char_img, w=28, h=28)

            # Add some padding to the character image.
            new_img = pad_img(char_img, pad_size=20)

            # Perform the same preprocessing steps as used with the AI's
            # training data.
            new_img = preprocess_nist_img(new_img * 255,
                resize_w=create_mat.COL_SIZE, resize_h=create_mat.COL_SIZE)

            # Predict the character.
            v_index = nn.predict_new(new_img.reshape(
                (create_mat.COL_SIZE ** 2, 1)))
            read_str += chr(create_mat.LETTERS[v_index])

            if show_blobs:
                draw_quad_over_image(scanned, blob.rect)
            if show_chars:
                print("Predicted:", read_str[-1])
                display_image(new_img)

    return read_str

if __name__ == '__main__':
    filename = input("Image path: ")
    imgdata = read_image_grayscale(filename)
    h, _ = imgdata.shape
    read_str = get_image_text(imgdata,
                              show_img=True, show_edges=True, show_chars=False,
                              show_quad=False, show_blobs=True,
                              doc_area_factor=4,
                              blob_dist_factor=20,
                              blob_area_factor=10000,
                              blob_num_pixels=40,
                              blob_aspect=0.1,
                              line_thresh=h / 20)
    if read_str is not None:
        print("Text:", read_str)
