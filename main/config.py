import os
import os.path as osp
import sys
import numpy as np

class Config:
    
    ## dataset
    dataset = 'JTA' # 'COCO', 'PoseTrack', 'MPII'
    testset = 'test' # train, test, val (there is no validation set for MPII)
    # additional_name = 'SyMPose_urban2worldexpo_standard_cyclegan_bs1_gfdim32_dfdim64_start2019-06-13-19-54-19'
    # additional_name = 'SyMPose_urban2iosb_standard_cyclegan_bs1_gfdim32_dfdim64_start2019-06-17-09-11-54'
    # additional_name = "SyMPose_urban2iosb_standard_cyclegan_bs1_gfdim32_dfdim64_start2019-06-17-09-11-54"
    additional_name = 'SyMPose_urban2iosb_spade_cyclegan_bs1_gfdim32_dfdim64_start2019-06-13-12-16-01'
    # additional_name = "SyMPose_urban2worldexpo_lr_sched_standard_cyclegan_bs1_gfdim32_dfdim64_start2019-06-13-19-54-19"
    # additional_name = "SyMPose_urban2worldexpo_spade_cyclegan_bs1_gfdim32_dfdim64_start2019-06-13-09-47-28"
    # additional_name = 'SyMPose_urban2cityscapes_standard_cyclegan_bs1_gfdim32_dfdim64_start2019-06-14-13-18-27'
    # additional_name = 'SyMPose'
    # additional_name = "SyMPose_green"

    ## directory
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = osp.join(cur_dir, '..')
    data_dir = osp.join(root_dir, 'data')
    output_dir = '/net/merkur/storage/deeplearning/users/blaand/data/jta_to_real/detection'
    # output_dir = osp.join(root_dir, 'output')
    model_dump_dir = osp.join(output_dir, 'model_dump', dataset,additional_name)
    vis_dir = osp.join(output_dir, 'vis', dataset,additional_name)
    log_dir = osp.join(output_dir, 'log', dataset,additional_name)
    result_dir = osp.join(output_dir, 'result', dataset,additional_name)
    summary_dir = osp.join(output_dir, 'tensorboard', dataset,additional_name)

 
    ## model setting
    backbone = 'resnet50' # 'resnet50', 'resnet101', 'resnet152'
    init_model = osp.join(data_dir, 'imagenet_weights', 'resnet_v1_' + backbone[6:] + '.ckpt')
    
    ## input, output
    input_shape = (256, 192) # (256,192), (384,288)
    output_shape = (input_shape[0]//4, input_shape[1]//4)
    if output_shape[0] == 64:
        sigma = 2
    elif output_shape[0] == 96:
        sigma = 3
    pixel_means = np.array([[[123.68, 116.78, 103.94]]])

    ## training config
    lr_dec_epoch = [3, 12]
    end_epoch = 15
    lr = 5e-4
    lr_dec_factor = 10
    optimizer = 'adam'
    weight_decay = 1e-5
    bn_train = True
    batch_size = 32
    scale_factor = 0.3
    rotation_factor = 40
    min_save_loss = 70

    # user defined
    save_summary_steps = 300


    ## testing config
    useGTbbox = True
    flip_test = True
    oks_nms_thr = 0.9
    score_thr = 0.2
    test_batch_size = 1

    ## others
    multi_thread_enable = True
    num_thread = 10
    gpu_ids = '0'
    num_gpus = 1
    continue_train = False
    display = 1
    
    ## helper functions
    def get_lr(self, epoch):
        for e in self.lr_dec_epoch:
            if epoch < e:
                break
        if epoch < self.lr_dec_epoch[-1]:
            i = self.lr_dec_epoch.index(e)
            return self.lr / (self.lr_dec_factor ** i)
        else:
            return self.lr / (self.lr_dec_factor ** len(self.lr_dec_epoch))
    
    def normalize_input(self, img):
        return img - self.pixel_means
    def denormalize_input(self, img):
        return img + self.pixel_means

    def set_args(self, gpu_ids, continue_train=False):
        self.gpu_ids = gpu_ids
        self.num_gpus = len(self.gpu_ids.split(','))
        self.continue_train = continue_train
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        print('>>> Using /gpu:{}'.format(self.gpu_ids))

cfg = Config()

sys.path.insert(0, osp.join(cfg.root_dir, 'lib'))
from tfflat.utils import add_pypath, make_dir
add_pypath(osp.join(cfg.data_dir))
add_pypath(osp.join(cfg.data_dir, cfg.dataset))
make_dir(cfg.model_dump_dir)
make_dir(cfg.vis_dir)
make_dir(cfg.log_dir)
make_dir(cfg.result_dir)
make_dir(cfg.summary_dir)

from dataset import dbcfg
cfg.num_kps = dbcfg.num_kps
cfg.kps_names = dbcfg.kps_names
cfg.kps_lines = dbcfg.kps_lines
cfg.kps_symmetry = dbcfg.kps_symmetry
cfg.img_path = dbcfg.img_path
cfg.human_det_path = dbcfg.human_det_path
cfg.vis_keypoints = dbcfg.vis_keypoints

