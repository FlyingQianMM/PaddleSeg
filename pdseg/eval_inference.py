# coding: utf8
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
# GPU memory garbage collection optimization flags
os.environ['FLAGS_eager_delete_tensor_gb'] = "0.0"

import sys
import argparse
import pprint
import numpy as np
import math
import tqdm
import paddle.fluid as fluid

from utils.config import cfg
from utils.timer import Timer, calculate_eta
from models.model_builder import build_model
from models.model_builder import ModelPhase
from reader import SegDataset
from metrics import ConfusionMatrix


def parse_args():
    parser = argparse.ArgumentParser(description='PaddleSeg model evalution')
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='Config file for training (and optionally testing)',
        default=None,
        type=str)
    parser.add_argument(
        '--use_gpu',
        dest='use_gpu',
        help='Use gpu or cpu',
        action='store_true',
        default=False)
    parser.add_argument(
        '--use_mpio',
        dest='use_mpio',
        help='Use multiprocess IO or not',
        action='store_true',
        default=False)
    parser.add_argument(
        'opts',
        help='See utils/config.py for all options',
        default=None,
        nargs=argparse.REMAINDER)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def evaluate(cfg, ckpt_dir=None, use_gpu=False, use_mpio=False, **kwargs):
    np.set_printoptions(precision=5, suppress=True)

    startup_prog = fluid.Program()
    test_prog = fluid.Program()
    dataset = SegDataset(
        file_list=cfg.DATASET.VAL_FILE_LIST,
        mode=ModelPhase.EVAL,
        data_dir=cfg.DATASET.DATA_DIR)

    pred, logit = build_model(test_prog, startup_prog, phase=ModelPhase.VISUAL)

    data_generator = dataset.batch(dataset.generator, batch_size=cfg.BATCH_SIZE)

    # Get device environment
    places = fluid.cuda_places() if use_gpu else fluid.cpu_places()
    place = places[0]
    dev_count = len(places)
    print("#Device count: {}".format(dev_count))

    exe = fluid.Executor(place)
    exe.run(startup_prog)

    test_prog = test_prog.clone(for_test=True)

    ckpt_dir = cfg.TEST.TEST_MODEL if not ckpt_dir else ckpt_dir

    if not os.path.exists(ckpt_dir):
        raise ValueError('The TEST.TEST_MODEL {} is not found'.format(ckpt_dir))

    if ckpt_dir is not None:
        print('load test model:', ckpt_dir)
        [prog, input_names, outputs] = fluid.io.load_inference_model(
            ckpt_dir, exe, params_filename='__params__')
        test_prog = prog


    # Use streaming confusion matrix to calculate mean_iou
    np.set_printoptions(
        precision=4, suppress=True, linewidth=160, floatmode="fixed")
    conf_mat = ConfusionMatrix(cfg.DATASET.NUM_CLASSES, streaming=True)
    fetch_list = [pred.name]
    num_images = 0
    timer = Timer()
    timer.start()
    step = 0
   
    for data in data_generator:
        images, labels, masks = data
        num_samples = images.shape[0]
        print(images.shape)
        feed_data = {'image': images}
        results = exe.run(test_prog,
                          feed=feed_data,
                          fetch_list=outputs,
                          return_numpy=True)
        pred = results[0]
        print(pred.shape)
        pred = np.argmax(pred, axis=1)
        pred = np.expand_dims(pred, axis=-1)
        mask = labels != cfg.DATASET.IGNORE_INDEX
        num_images += pred.shape[0]
        conf_mat.calculate(pred, labels, mask)
        _, iou = conf_mat.mean_iou()
        _, acc = conf_mat.accuracy()

        speed = 1.0 / timer.elapsed_time()

        step += 1
        print(
            "[EVAL]step={} acc={:.4f} IoU={:.4f}"
            .format(step, acc, iou))
        timer.restart()
        sys.stdout.flush()

    category_iou, avg_iou = conf_mat.mean_iou()
    category_acc, avg_acc = conf_mat.accuracy()
    print("[EVAL]#image={} acc={:.4f} IoU={:.4f}".format(
        num_images, avg_acc, avg_iou))
    print("[EVAL]Category IoU:", category_iou)
    print("[EVAL]Category Acc:", category_acc)
    print("[EVAL]Kappa:{:.4f}".format(conf_mat.kappa()))

    return category_iou, avg_iou, category_acc, avg_acc


def main():
    args = parse_args()
    if args.cfg_file is not None:
        cfg.update_from_file(args.cfg_file)
    if args.opts:
        cfg.update_from_list(args.opts)
    cfg.check_and_infer()
    print(pprint.pformat(cfg))
    evaluate(cfg, **args.__dict__)


if __name__ == '__main__':
    main()
