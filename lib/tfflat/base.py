import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from collections import OrderedDict as dict
import setproctitle
import os
import os.path as osp
import glob
import abc
import math
import random
import json
from crowdposetools.coco import COCO
import time
import copy

from .net_utils import average_gradients, aggregate_batch, get_optimizer, get_tower_summary_dict
from .saver import load_model, Saver
from .timer import Timer
from .logger import colorlogger
from .utils import approx_equal

class ModelDesc(object):
    __metaclass__ = abc.ABCMeta
    def __init__(self):
        self._loss = None
        self._inputs = []
        self._outputs = []
        self._tower_summary = []

    def set_inputs(self, *vars):
        self._inputs = vars

    def set_outputs(self, *vars):
        self._outputs = vars

    def set_loss(self, var):
        if not isinstance(var, tf.Tensor):
            raise ValueError("Loss must be an single tensor.")
        # assert var.get_shape() == [], 'Loss tensor must be a scalar shape but got {} shape'.format(var.get_shape())
        self._loss = var

    def get_loss(self, include_wd=False):
        if self._loss is None:
            raise ValueError("Network doesn't define the final loss")

        if include_wd:
            weight_decay = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            weight_decay = tf.add_n(weight_decay)
            return self._loss + weight_decay
        else:
            return self._loss

    def get_inputs(self):
        if len(self._inputs) == 0:
            raise ValueError("Network doesn't define the inputs")
        return self._inputs

    def get_outputs(self):
        if len(self._outputs) == 0:
            raise ValueError("Network doesn't define the outputs")
        return self._outputs

    def add_tower_summary(self, name, vars, reduced_method='mean'):
        assert reduced_method == 'mean' or reduced_method == 'sum', \
            "Summary tensor only supports sum- or mean- reduced method"
        if isinstance(vars, list):
            for v in vars:
                if vars.get_shape() == None:
                    print('Summary tensor {} got an unknown shape.'.format(name))
                else:
                    assert v.get_shape().as_list() == [], \
                        "Summary tensor only supports scalar but got {}".format(v.get_shape().as_list())
                tf.add_to_collection(name, v)

                # add tensorboard summary
                tf.summary.scalar(name,v)
        else:
            if vars.get_shape() == None:
                print('Summary tensor {} got an unknown shape.'.format(name))
            else:
                assert vars.get_shape().as_list() == [], \
                    "Summary tensor only supports scalar but got {}".format(vars.get_shape().as_list())

                # add tensorboard summary
                tf.summary.scalar(name,vars)
            tf.add_to_collection(name, vars)
        self._tower_summary.append([name, reduced_method])

    @abc.abstractmethod
    def make_network(self, is_train):
        pass


class Base(object):
    __metaclass__ = abc.ABCMeta
    """
    build graph:
        _make_graph
            make_inputs
            make_network
                add_tower_summary
        get_summary
    
    train/test
    """

    def __init__(self, net, cfg, data_iter=None, log_name='logs.txt'):
        self._input_list = []
        self._output_list = []
        self._outputs = []
        self.graph_ops = None

        self.net = net
        self.cfg = cfg

        self._optimizer = None

        self.cur_epoch = 0

        self.summary_dict = {}

        # timer
        self.tot_timer = Timer()
        self.gpu_timer = Timer()
        self.read_timer = Timer()

        # logger
        self.logger = colorlogger(cfg.log_dir, log_name=log_name)

        # initialize tensorflow
        self.tfconfig = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        self.tfconfig.gpu_options.allow_growth = True

        # only use this session for test
        if isinstance(self,Tester):
            self.sess = tf.Session(config=self.tfconfig)

            # build_graph
            self.build_graph()

        # get data iter
        self._data_iter = data_iter

    @abc.abstractmethod
    def _make_data(self):
        return

    @abc.abstractmethod
    def _make_graph(self):
        return

    def build_graph(self):
        # all variables should be in the same graph and stored in cpu.
        with tf.device('/device:CPU:0'):
            tf.set_random_seed(2333)
            self.graph_ops = self._make_graph()
            if not isinstance(self.graph_ops, list) and not isinstance(self.graph_ops, tuple):
                self.graph_ops = [self.graph_ops]
        self.summary_dict.update( get_tower_summary_dict(self.net._tower_summary) )

    def load_weights(self,model=None,model_dump_dir=None,sess=None):

        if isinstance(self,Trainer):
            assert sess is not None
        else:
            assert sess is None

        load_dir = self.cfg.model_dump_dir if model_dump_dir is None else model_dump_dir
        used_sess = self.sess if sess is None else sess

        if model == 'last_epoch':
            sfiles = os.path.join(load_dir, 'snapshot_*.ckpt.meta')
            sfiles = glob.glob(sfiles)
            if len(sfiles) > 0:
                sfiles.sort(key=os.path.getmtime)
                sfiles = [i[:-5] for i in sfiles if i.endswith('.meta')]
                model = sfiles[-1]
            else:
                self.logger.critical('No snapshot model exists.')
                return

        if isinstance(model, int):
            model = os.path.join(load_dir, 'snapshot_%d.ckpt' % model)

        if isinstance(model, str) and (osp.exists(model + '.meta') or osp.exists(model)):
            self.logger.info('Initialized model weights from {} ...'.format(model))
            load_model(used_sess, model)
            if model.split('/')[-1].startswith('snapshot_'):
                self.cur_epoch = int(model[model.find('snapshot_')+9:model.find('.ckpt')])
                self.logger.info('Current epoch is %d.' % self.cur_epoch)
        else:
            self.logger.critical('Load nothing. There is no model in path {}.'.format(model))

    def next_feed(self,data_iter):
        if self._data_iter is None and data_iter is None:
            raise ValueError('No input data.')
        feed_dict = dict()
        for inputs in self._input_list:
            blobs = next(data_iter)
            for i, inp in enumerate(inputs):
                inp_shape = inp.get_shape().as_list()
                if None in inp_shape:
                    feed_dict[inp] = blobs[i]
                else:
                    feed_dict[inp] = blobs[i].reshape(*inp_shape)
        return feed_dict

class Trainer(Base):
    def __init__(self, net, cfg, data_iter=None):
        from dataset import Dataset
        # self.lr_eval = cfg.lr
        # self.save_summary_steps = cfg.save_summary_steps
        # self.summary_dir = cfg.summary_dir
        # self.lr = tf.Variable(cfg.lr, trainable=False)
        # self._optimizer = get_optimizer(self.lr, cfg.optimizer)

        super(Trainer, self).__init__(net, cfg, data_iter, log_name='train_logs.txt')

        # make data
        # fixme this has to be applied from inside thhe training loop
        self.d = Dataset()
        # self._data_iter, self.itr_per_epoch = self._make_data()
        if self.cfg.cnt_val_itr >= self.d.num_val_split:
            raise ValueError("The validation iteration to continue is larger than overall number of cross-validation runs!")

    def _make_graph(self):

        assert self._optimizer is not None

        self.logger.info("Generating training graph on {} GPUs ...".format(self.cfg.num_gpus))

        weights_initializer = slim.xavier_initializer()
        biases_initializer = tf.constant_initializer(0.)
        biases_regularizer = tf.no_regularizer
        weights_regularizer = tf.contrib.layers.l2_regularizer(self.cfg.weight_decay)

        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(self.cfg.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('tower_%d' % i) as name_scope:
                        # Force all Variables to reside on the CPU.
                        with slim.arg_scope([slim.model_variable, slim.variable], device='/device:CPU:0'):
                            with slim.arg_scope([slim.conv2d, slim.conv2d_in_plane, \
                                                 slim.conv2d_transpose, slim.separable_conv2d,
                                                 slim.fully_connected],
                                                weights_regularizer=weights_regularizer,
                                                biases_regularizer=biases_regularizer,
                                                weights_initializer=weights_initializer,
                                                biases_initializer=biases_initializer):
                                # loss over single GPU
                                self.net.make_network(is_train=True)
                                if i == self.cfg.num_gpus - 1:
                                    loss = self.net.get_loss(include_wd=True)
                                else:
                                    loss = self.net.get_loss()
                                self._input_list.append( self.net.get_inputs() )

                        tf.get_variable_scope().reuse_variables()

                        if i == 0:
                            if self.cfg.num_gpus > 1 and self.cfg.bn_train is True:
                                self.logger.warning("BN is calculated only on single GPU.")
                            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, name_scope)
                            with tf.control_dependencies(extra_update_ops):
                                grads = self._optimizer.compute_gradients(loss)
                        else:
                            grads = self._optimizer.compute_gradients(loss)
                        final_grads = []
                        with tf.variable_scope('Gradient_Mult') as scope:
                            for grad, var in grads:
                                final_grads.append((grad, var))
                        tower_grads.append(final_grads)

        if len(tower_grads) > 1:
            grads = average_gradients(tower_grads)
        else:
            grads = tower_grads[0]

        apply_gradient_op = self._optimizer.apply_gradients(grads)
        train_op = tf.group(apply_gradient_op, *extra_update_ops)

        return train_op

    def train(self):
        from gen_batch import generate_batch
        from tfflat.data_provider import  DataFromList, MultiProcessMapDataZMQ, BatchData, MapData
        from test import test

        start_val_itr = self.cfg.cnt_val_itr if self.cfg.cnt_val_itr >= 0 else 0

        for out_itr in range(start_val_itr,self.d.num_val_split):
            # reset input and output lists
            self._input_list = []
            self._output_list = []
            self._outputs = []
            self.graph_ops = None

            # reset current epoch
            self.cur_epoch = 0


            #reset summary dict
            self.summary_dict = {}

            # timer
            self.tot_timer = Timer()
            self.gpu_timer = Timer()
            self.read_timer = Timer()

            run_pref = "run_{}".format(out_itr + 1)
            lr_eval = self.cfg.lr
            save_summary_steps = self.cfg.save_summary_steps
            summary_dir = os.path.join(self.cfg.summary_dir, run_pref)
            train_data, val_data = self.d.load_train_data(out_itr)



            with tf.Session(config=self.tfconfig) as sess:

                lr = tf.Variable(self.cfg.lr, trainable=False)
                self._optimizer = get_optimizer(lr, self.cfg.optimizer)


                if self.cfg.equal_random_seed:
                    # set random seed for the python pseudo random number generator in order to obtain comparable results
                    tf.set_random_seed(2223)
                    random.seed(2223)

                # build_graph
                self.build_graph()

                data_load_thread = DataFromList(train_data)
                if self.cfg.multi_thread_enable:
                    data_thread = MultiProcessMapDataZMQ(data_load_thread, self.cfg.num_thread, generate_batch, strict=True)
                else:
                    data_thread = MapData(data_load_thread, generate_batch)
                data_load_thread = BatchData(data_thread, self.cfg.batch_size)

                if self.cfg.equal_random_seed:
                    data_load_thread.reset_state()

                dataiter = data_load_thread.get_data()
                itr_per_epoch = math.ceil(len(train_data)/self.cfg.batch_size/self.cfg.num_gpus)



                # summaries
                # merge all summaries, run this operation later in order to retain the added summaries
                merged_sums = tf.summary.merge_all()
                writer = tf.summary.FileWriter(summary_dir,sess.graph)


                # saver
                self.logger.info('Initialize saver ...')
                model_dump_dir = os.path.join(self.cfg.model_dump_dir,run_pref)
                train_saver = Saver(sess, tf.global_variables(), model_dump_dir)

                best_model_dir = os.path.join(model_dump_dir,"best_model")
                val_dir = os.path.join(self.cfg.val_dir,run_pref)
                if not os.path.isdir(best_model_dir):
                    os.makedirs(best_model_dir)

                if not os.path.isdir(val_dir):
                    os.makedirs(val_dir)

                best_saver = Saver(sess,tf.global_variables(),best_model_dir,max_to_keep=1)

                # initialize weights
                self.logger.info('Initialize all variables ...')
                sess.run(tf.variables_initializer(tf.global_variables(), name='init'))
                self.load_weights('last_epoch' if self.cfg.continue_train else self.cfg.init_model,model_dump_dir,sess=sess)

                self.logger.info('Start training; validation iteration #{}...'.format(out_itr))
                start_itr = self.cur_epoch * itr_per_epoch + 1
                end_itr = itr_per_epoch * self.cfg.end_epoch + 1
                best_loss = self.cfg.min_save_loss
                for itr in range(start_itr, end_itr):
                    self.tot_timer.tic()

                    self.cur_epoch = itr // itr_per_epoch
                    setproctitle.setproctitle('val_it {};train epoch{}:'.format(out_itr,self.cur_epoch))

                    # apply current learning policy
                    cur_lr = self.cfg.get_lr(self.cur_epoch)
                    if not approx_equal(cur_lr, lr_eval):
                        print(lr_eval, cur_lr)
                        sess.run(tf.assign(lr, cur_lr))

                    # input data
                    self.read_timer.tic()
                    feed_dict = self.next_feed(dataiter)
                    self.read_timer.toc()

                    # train one step
                    self.gpu_timer.tic()
                    _, lr_eval, *summary_res, tb_summaries = sess.run(
                        [self.graph_ops[0], lr, *self.summary_dict.values(),merged_sums], feed_dict=feed_dict)
                    self.gpu_timer.toc()

                    # write summary values to event file at disk
                    if itr % save_summary_steps == 0:
                        writer.add_summary(tb_summaries,itr)

                    itr_summary = dict()
                    for i, k in enumerate(self.summary_dict.keys()):
                        itr_summary[k] = summary_res[i]

                    screen = [
                        'Validation itr %d' % (out_itr),
                        'Epoch %d itr %d/%d:' % (self.cur_epoch, itr, itr_per_epoch),
                        'lr: %g' % (lr_eval),
                        'speed: %.2f(%.2fs r%.2f)s/itr' % (
                            self.tot_timer.average_time, self.gpu_timer.average_time, self.read_timer.average_time),
                        '%.2fh/epoch' % (self.tot_timer.average_time / 3600. * itr_per_epoch),
                        ' '.join(map(lambda x: '%s: %.4f' % (x[0], x[1]), itr_summary.items())),
                    ]


                    #TODO(display stall?)
                    if itr % self.cfg.display == 0:
                        self.logger.info(' '.join(screen))

                    # save best model
                    loss = itr_summary['loss']
                    if loss < best_loss:
                        best_loss = loss
                        print("Saving model because best loss was undergone; Value is {}.".format(loss))
                        best_saver.save_model(self.cfg.end_epoch + 1)

                    if itr % itr_per_epoch == 0:
                        train_saver.save_model(self.cur_epoch)

                    self.tot_timer.toc()


            #clean up
            sess.close()
            tf.reset_default_graph()
            if self.cfg.multi_thread_enable:
                data_thread.__del__()
            print("Finish training for val run #{}; Apply validation".format(out_itr + 1))
            if self.cfg.additional_name=="CrowdPose":
                print("Training on CrowdPose, no additional validation required!")
            else:
                self.cross_val(val_data,self.cfg.end_epoch + 1,val_dir,best_model_dir)


    def cross_val(self,val_data,val_model,val_dir,model_dir):
        """
        This function applies cross validation to a trained model after one complete trainind and stores the result
        :param val_data: The validation split of the overall data set that has not been seen before
        :param val_model: The checkpoint number that's to be used for the evaluation
        :param val_dir: The directory to store the validation results to
        :param model_dir: The directory where the loaded model for eval can be found
        :return:
        """

        #validation data contains also bounding boxes
        dets = val_data
        dets.sort(key=lambda x: (x['image_id']))

        # store val data as json in order to be able to read it with COCO
        val_gt_path = osp.join(val_dir,"val_kp_gt.json")
        with open(val_gt_path,"w") as f:
            json.dump(val_data,f)

        coco = COCO(self.d.train_annot_path)
        # construct coco object for evaluation
        val_gt = self.construct_coco(val_data,coco,val_gt_path,val_dir)

        # from tfflat.mp_utils import MultiProc
        from test import test_net
        from model import Model
        img_start = 0
        ranges = [0]
        img_num = len(np.unique([i['image_id'] for i in dets]))
        images_per_gpu = int(img_num / len(self.cfg.gpu_ids.split(','))) + 1
        for run_img in range(img_num):
            img_end = img_start + 1
            while img_end < len(dets) and dets[img_end]['image_id'] == dets[img_start]['image_id']:
                img_end += 1
            if (run_img + 1) % images_per_gpu == 0 or (run_img + 1) == img_num:
                ranges.append(img_end)
            img_start = img_end

        def func(gpu_id):
            tester = Tester(Model(), self.cfg)
            tester.load_weights(val_model,model_dump_dir=model_dir)
            range = [ranges[gpu_id], ranges[gpu_id + 1]]
            return test_net(tester, dets, range, gpu_id, self.d.sigmas,False)

        # MultiGPUFunc = MultiProc(len(self.cfg.gpu_ids.split(',')), func)
        # result = MultiGPUFunc.work()
        result = func(0)
        # evaluation
        self.d.evaluation(result, val_gt, val_dir, self.cfg.testset)

        # remove tensorflow graph to be able to start next training
        tf.reset_default_graph()

        # clean up hard disk
        os.remove(val_gt.anno_file[0])
        os.remove(val_gt_path)

    def construct_coco(self,data,base_coco,res_file,val_dir):
        """
        Construct a COCO object from a data list
        :param data:
        :return:
        """

        res = COCO()
        img_file_names = [p["imgpath"] for p in data]
        res.dataset['images'] = [img for img in base_coco.dataset['images'] if img['file_name'] in img_file_names]
        print('Loading and preparing results...')
        with open(res_file) as f:
            anns = json.load(f)

        assert type(anns) == list, 'results in not an array of objects'

        res.dataset['categories'] = copy.deepcopy(
            base_coco.dataset['categories'])

        ann_ids = [ann["id"] for ann in anns]
        gt_anno_file_path = osp.join(val_dir, "gt_anno.json")
        res.dataset['annotations'] = [ann for ann in base_coco.dataset["annotations"] if ann["id"] in ann_ids]
        res.createIndex()

        with open(gt_anno_file_path,"w") as f:
            json.dump(res.dataset,f)

        res.anno_file=[gt_anno_file_path]
        return res

class Tester(Base):
    def __init__(self, net, cfg, data_iter=None):
        super(Tester, self).__init__(net, cfg, data_iter, log_name='test_logs.txt')

    def next_feed(self, batch_data=None):
        if self._data_iter is None and batch_data is None:
            raise ValueError('No input data.')
        feed_dict = dict()
        if batch_data is None:
            for inputs in self._input_list:
                blobs = next(self._data_iter)
                for i, inp in enumerate(inputs):
                    inp_shape = inp.get_shape().as_list()
                    if None in inp_shape:
                        feed_dict[inp] = blobs[i]
                    else:
                        feed_dict[inp] = blobs[i].reshape(*inp_shape)
        else:
            assert isinstance(batch_data, list) or isinstance(batch_data, tuple), "Input data should be list-type."
            assert len(batch_data) == len(self._input_list[0]), "Input data is incomplete."

            batch_size = self.cfg.batch_size
            if self._input_list[0][0].get_shape().as_list()[0] is None:
                # fill batch
                for i in range(len(batch_data)):
                    batch_size = (len(batch_data[i]) + self.cfg.num_gpus - 1) // self.cfg.num_gpus
                    total_batches = batch_size * self.cfg.num_gpus
                    left_batches = total_batches - len(batch_data[i])
                    if left_batches > 0:
                        batch_data[i] = np.append(batch_data[i], np.zeros((left_batches, *batch_data[i].shape[1:])), axis=0)
                        self.logger.warning("Fill some blanks to fit batch_size which wastes %d%% computation" % (
                            left_batches * 100. / total_batches))
            else:
                assert self.cfg.batch_size * self.cfg.num_gpus == len(batch_data[0]), \
                    "Input batch doesn't fit placeholder batch."

            for j, inputs in enumerate(self._input_list):
                for i, inp in enumerate(inputs):
                    feed_dict[ inp ] = batch_data[i][j * batch_size: (j+1) * batch_size]

            #@TODO(delete)
            assert (j+1) * batch_size == len(batch_data[0]), 'check batch'
        return feed_dict, batch_size

    def _make_graph(self):
        # optimizer has to bne None for test mode
        assert self._optimizer is None
        self.logger.info("Generating testing graph on {} GPUs ...".format(self.cfg.num_gpus))

        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(self.cfg.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('tower_%d' % i) as name_scope:
                        with slim.arg_scope([slim.model_variable, slim.variable], device='/device:CPU:0'):
                            self.net.make_network(is_train=False)
                            self._input_list.append(self.net.get_inputs())
                            self._output_list.append(self.net.get_outputs())

                        tf.get_variable_scope().reuse_variables()

        self._outputs = aggregate_batch(self._output_list)

        # run_meta = tf.RunMetadata()
        # opts = tf.profiler.ProfileOptionBuilder.float_operation()
        # flops = tf.profiler.profile(self.sess.graph, run_meta=run_meta, cmd='op', options=opts)
        #
        # opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
        # params = tf.profiler.profile(self.sess.graph, run_meta=run_meta, cmd='op', options=opts)

        # print("{:,} --- {:,}".format(flops.total_float_ops, params.total_parameters))
        # from IPython import embed; embed()

        return self._outputs

    def predict_one(self, data=None):
        # TODO(reduce data in limited batch)
        assert len(self.summary_dict) == 0, "still not support scalar summary in testing stage"
        setproctitle.setproctitle('test epoch:' + str(self.cur_epoch))

        self.read_timer.tic()
        feed_dict, batch_size = self.next_feed(data)
        self.read_timer.toc()

        self.gpu_timer.tic()
        res = self.sess.run([*self.graph_ops, *self.summary_dict.values()], feed_dict=feed_dict)
        self.gpu_timer.toc()

        if data is not None and len(data[0]) < self.cfg.num_gpus * batch_size:
            for i in range(len(res)):
                res[i] = res[i][:len(data[0])]

        return res

    def test(self):
        pass

