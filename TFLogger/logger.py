# -*-coding:utf-8-*-
import numpy as np
from io import BytesIO
import tensorflow as tf
class TFLogger(object):
    def __init__(self, log_dir=None):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.compat.v1.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def images_summary(self, tag, images, step):
        """Log a list of images."""
        for i, imgs in enumerate(images):
            img_summaries = []
            for bidx, img in enumerate(imgs):
                # Write the image to a string
                s = BytesIO()
                img.save(s, format="png")

                # Create an Image object
                img_sum = tf.compat.v1.Summary.Image(encoded_image_string=s.getvalue(),
                                           height=img.size[0],
                                           width=img.size[1])
                # Create a Summary value
                img_summaries.append(tf.compat.v1.Summary.Value(tag='%d/%d' % (bidx, tag[bidx]), image=img_sum))

            # Create and write Summary
            summary = tf.compat.v1.Summary(value=img_summaries)
            self.writer.add_summary(summary, step[i])
            self.writer.flush()

    def image_summary(self, tag, image, step):
        # Write the image to a string
        s = BytesIO()
        image.save(s, format="png")

        # Create an Image object
        img_sum = tf.compat.v1.Summary.Image(encoded_image_string=s.getvalue(),
                                   height=image.size[0],
                                   width=image.size[1])

        # Create and write Summary
        summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag='%s' % tag, image=img_sum)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def close(self):
        self.writer.close()
