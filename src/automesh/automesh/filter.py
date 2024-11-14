import numpy as np
import scipy.ndimage.measurements
import skimage.filter
import skimage.filter.rank
import sklearn.cluster


def crop(original, ranges):
    cropped = original.copy(copy_values=False)
    cropped.values = original.values[ranges[0][0]:ranges[0][1], ranges[1][0]:ranges[1][1], ranges[2][0]:ranges[2][1]]
    cropped.origin = [original.origin[i] + ranges[i][0] * original.spacing[i] for i in [0, 1, 2]]
    return cropped


def denoise_bilateral(original, params, inplace=True):
    if inplace:
        max_intensity = original.values.max()
        original.values = skimage.filter.denoise_bilateral(original.values / max_intensity,
                                                           sigma_range=params.sigma_range,
                                                           sigma_spatial=params.sigma_spatial,
                                                           win_size=params.win_size)
        original.values *= max_intensity
        return original
    else:
        # TODO: copy scan, denoise, then return denoised copy
        pass


# Calculate Image Entropy
def image_entropy(im, params):
    im2 = im / im.max()
    en = np.zeros(im.shape)
    selem = np.ones([params.size, params.size])
    for i in range(im.shape[2]):
        en[:, :, i] = skimage.filter.rank.entropy(im2[:, :, i], selem)
    return en


# K-means segmentation
def kmeans_segmentation(scan, params, entropy=None):
    if entropy is None:
        entropy = image_entropy(scan.values, params.entropy)

    x1 = np.array([entropy.reshape([-1]), scan.values.reshape([-1])])
    kmeans = sklearn.cluster.KMeans(n_clusters=params.num_clusters,
                                    init='k-means++',
                                    max_iter=params.max_iterations,
                                    n_jobs=params.num_jobs)
    kmeans.fit(x1.T)
    labels = scan.copy(copy_values=False)
    labels.values = kmeans.predict(x1.T).reshape(scan.shape)
    return labels


def order_labels(labelled_image, image, mode):
    if mode == 'mean_intensity':
        means = []
        labels = []
        for i in range(labelled_image.values.min(), labelled_image.values.max() + 1):
            labels.append(i)
            means.append(np.mean(image.values[labelled_image.values == i]))
        order = np.argsort(means)
        return order


def extract_label(labelled_image, label):
    extracted = labelled_image.copy(copy_values=False)
    extracted.values = np.zeros(labelled_image.shape)
    extracted.values[labelled_image.values == label] = 1
    return extracted


def extract_lungs_image(air_image):
    kres4 = np.ones(air_image.shape)
    for i in range(0, air_image.shape[2]):
        nm1 = np.zeros([air_image.shape[0], air_image.shape[1]])
        lbl, nlbl = scipy.ndimage.measurements.label(air_image.values[:, :, i])
        lbls = np.arange(1, nlbl + 1)
        pl = scipy.ndimage.measurements.labeled_comprehension(air_image.values[:, :, i], lbl, lbls, np.sum, float, 0)
        if len(pl) >= 2:
            li1 = np.argsort(pl)[-2] + 1
            if len(pl) >= 3:
                li2 = np.argsort(pl)[-3] + 1
                nm1[np.logical_or(lbl == li1, lbl == li2)] = 1
            else:
                nm1[lbl == li1] = 1
        kres4[:, :, i] = nm1
    lungs_image = air_image.copy(copy_values=False)
    lungs_image.values = kres4
    return lungs_image


def extract_skin_image(air_image):
    kres3 = np.ones(air_image.shape)
    for i in range(0, air_image.shape[2]):
        nm = np.ones([air_image.shape[0], air_image.shape[1]])
        lbl, nlbl = scipy.ndimage.measurements.label(air_image.values[:, :, i])
        lbls = np.arange(1, nlbl + 1)
        pl = scipy.ndimage.measurements.labeled_comprehension(air_image.values[:, :, i], lbl, lbls, np.sum, float, 0)
        li = np.argsort(pl)[-1] + 1
        nm[lbl == li] = 0
        kres3[:, :, i] = nm
    skin_image = air_image.copy(copy_values=False)
    skin_image.values = kres3
    return skin_image


# Remove small irregular structures
def remove_small_structures(labelled_image, threshold, inplace=True):
    if not inplace:
        print 'replace not implemented'
        return labelled_image
    labels, num_labels = scipy.ndimage.measurements.label(labelled_image.values)
    lbls = np.arange(1, num_labels + 1)
    pl = scipy.ndimage.measurements.labeled_comprehension(labelled_image.values, labels, lbls, np.sum, float, 0)
    pl = (np.float32(pl) / np.float32(np.sum(pl))) * 100
    res5 = np.zeros(labelled_image.shape)
    # Compare the three largest structures
    pq = np.minimum(3, len(pl))
    for i in range(-pq, 0):
        # Remove irrelevant structures smaller
        # than 20% of the total segmentation
        if pl[np.argsort(pl)[i]] > threshold:
            li = np.argsort(pl)[i] + 1
            res5[labels == li] = 1
    labelled_image.values = res5
    return labelled_image


def image_to_points(image, params):
    pts = None
    return pts
