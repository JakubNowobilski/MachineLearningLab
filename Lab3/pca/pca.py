from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import imageio
from sklearn.decomposition import PCA

image_shape = (128, 128)
classes = ["Codzienne", "Klapki", "Pantofle", "Trampki", "Zimowe"]


def load_images(path):
    images = []
    for shoe_class in classes:
        for idx in range(1, 7):
            img = imageio.imread(path + "/" + shoe_class + " (" + str(idx) + ").png")
            img = img.ravel()
            images.append(img)

    return np.array(images)


def plot_gallery(title, images, n_col=6, n_row=5, labels=None):
    plt.figure(figsize=(3.0 * n_col, 3.0 * n_row))
    plt.suptitle(title, size=12 + n_col * n_row)
    for i, img in enumerate(images):
        plt.subplot(n_row, n_col, i + 1, title=None if labels is None else labels[i])
        plt.imshow(img.reshape(image_shape), cmap=plt.cm.gray)
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.05, 0.05, 0.95, 0.90)
    plt.show()


def pca_reconstruction(n, images_centered, mean_image):
    pca_n = PCA(n_components=n)
    pca_n.fit(images_centered)
    pca_n_images = pca_n.transform(images_centered)
    pca_n_images = pca_n.inverse_transform(pca_n_images) + mean_image
    plot_gallery("Image reconstruction with " + str(n) + " principal components", pca_n_images)


def scatter_projection(pca_images):
    pca_images = pca_images.reshape(5, 6, 2)
    plt.subplot()
    for i, shoe_class in enumerate(pca_images):
        plt.scatter(shoe_class[:, 0], shoe_class[:, 1], label=classes[i])
    plt.title("PCA 2D projection")
    plt.legend()
    plt.show()


# load images
images = load_images("128x128")
n_samples, n_features = images.shape

# plot original images
plot_gallery("Original images", images)

# mean image
mean_image = images.mean(axis=0)
plot_gallery("Mean image", [mean_image], 1, 1)

# images centering
images_centered = images - mean_image
plot_gallery("Centered images", images_centered)

# PCA
pca = PCA(n_components=30)
pca.fit(images_centered)
plot_gallery("Principal components with explained variance ratio",
             pca.components_,
             labels=pca.explained_variance_ratio_)

# PCA reconstruction n=4
pca_reconstruction(4, images_centered, mean_image)
# PCA reconstruction n=16
pca_reconstruction(16, images_centered, mean_image)
# PCA reconstruction n=2
pca_reconstruction(2, images_centered, mean_image)

# PCA n=2 projection
pca2 = PCA(n_components=2)
pca2.fit(images_centered)
pca2_images = pca2.transform(images_centered)
scatter_projection(pca2_images)

