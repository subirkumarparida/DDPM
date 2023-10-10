import argparse
import matplotlib.pyplot as plot

import os
import chainer

from chainercv.links import FasterRCNNVGG16
from chainercv import utils
from chainercv.visualizations import vis_bbox

#import download_model "D:\Face\Wider_Face\trained_model"
TRAINED_MODEL_DEFAULT = 'D:/Face/Wider_Face/trained_model/snapshot_model.npz'
#MODEL_URL = 'http://nixeneko.2-d.jp/hatenablog/20170724_facedetection_model/snapshot_model.npz'

# A pretrained model for the higher version of chainercv.
TRAINED_MODEL_NEW = 'D:/Face/Wider_Face/trained_model/snapshot_model_20180404.npz'
#MODEL_URL_NEW = 'http://nixeneko.2-d.jp/hatenablog/20170724_facedetection_model/snapshot_model_20180404.npz'

def main():
    chainer.config.train = False

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--pretrained_model', default=TRAINED_MODEL_DEFAULT)
    parser.add_argument('image')
    args = parser.parse_args()
    print(args.pretrained_model)
    print(args.image)
    print("here 1--------")

    try:
        print("here 2--------")
        model = FasterRCNNVGG16(
            n_fg_class=1,
            pretrained_model=args.pretrained_model)
        print("here 3--------")
    
    # Workaround for the newer version of chainercv (maybe v0.7.0 and higher)
    except KeyError:
        print("here 4--------")
        model = FasterRCNNVGG16(
            n_fg_class=1,
            pretrained_model=TRAINED_MODEL_NEW)
        print("here 5--------")

    if args.gpu >= 0:
        model.to_gpu(args.gpu)
        chainer.cuda.get_device(args.gpu).use()

    img = utils.read_image(args.image, color=True)
    bboxes, labels, scores = model.predict([img])
    bbox, label, score = bboxes[0], labels[0], scores[0]

    vis_bbox(
        img, bbox, label, score, label_names=('face',))
    plot.show()


if __name__ == '__main__':
    main()
