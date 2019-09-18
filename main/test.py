import os
import os.path as osp
import numpy as np
import argparse
from config import cfg
import cv2
import time
import json
from tqdm import tqdm
import math
from glob import glob
import pandas as pd

import tensorflow as tf

from tfflat.base import Tester
from tfflat.utils import mem_info
from model import Model

from gen_batch import generate_batch
from dataset import Dataset
from nms.nms import oks_nms

def test_net(tester, dets, det_range, gpu_id,sigmas,vis_kps):

    dump_results = []

    start_time = time.time()

    img_start = det_range[0]
    img_id = 0
    img_id2 = 0
    pbar = tqdm(total=det_range[1] - img_start - 1, position=gpu_id)
    pbar.set_description("GPU %s" % str(gpu_id))
    while img_start < det_range[1]:
        img_end = img_start + 1
        im_info = dets[img_start]
        while img_end < det_range[1] and dets[img_end]['image_id'] == im_info['image_id']:
            img_end += 1
        
        # all human detection results of a certain image
        cropped_data = dets[img_start:img_end]

        pbar.update(img_end - img_start)
        img_start = img_end

        kps_result = np.zeros((len(cropped_data), cfg.num_kps, 3))
        area_save = np.zeros(len(cropped_data))

        # cluster human detection results with test_batch_size
        for batch_id in range(0, len(cropped_data), cfg.test_batch_size):
            start_id = batch_id
            end_id = min(len(cropped_data), batch_id + cfg.test_batch_size)
             
            imgs = []
            crop_infos = []
            for i in range(start_id, end_id):
                img, crop_info = generate_batch(cropped_data[i], stage='test')
                imgs.append(img)
                crop_infos.append(crop_info)
            imgs = np.array(imgs)
            crop_infos = np.array(crop_infos)
            
            # forward
            heatmap = tester.predict_one([imgs])[0]
            
            if cfg.flip_test:
                flip_imgs = imgs[:, :, ::-1, :]
                flip_heatmap = tester.predict_one([flip_imgs])[0]
               
                flip_heatmap = flip_heatmap[:, :, ::-1, :]
                for (q, w) in cfg.kps_symmetry:
                    flip_heatmap_w, flip_heatmap_q = flip_heatmap[:,:,:,w].copy(), flip_heatmap[:,:,:,q].copy()
                    flip_heatmap[:,:,:,q], flip_heatmap[:,:,:,w] = flip_heatmap_w, flip_heatmap_q
                flip_heatmap[:,:,1:,:] = flip_heatmap.copy()[:,:,0:-1,:]
                heatmap += flip_heatmap
                heatmap /= 2
            
            # for each human detection from clustered batch
            for image_id in range(start_id, end_id):
               
                for j in range(cfg.num_kps):
                    hm_j = heatmap[image_id - start_id, :, :, j]
                    idx = hm_j.argmax()
                    y, x = np.unravel_index(idx, hm_j.shape)
                    
                    px = int(math.floor(x + 0.5))
                    py = int(math.floor(y + 0.5))
                    if 1 < px < cfg.output_shape[1]-1 and 1 < py < cfg.output_shape[0]-1:
                        diff = np.array([hm_j[py][px+1] - hm_j[py][px-1],
                                         hm_j[py+1][px]-hm_j[py-1][px]])
                        diff = np.sign(diff)
                        x += diff[0] * .25
                        y += diff[1] * .25
                    kps_result[image_id, j, :2] = (x * cfg.input_shape[1] / cfg.output_shape[1], y * cfg.input_shape[0] / cfg.output_shape[0])
                    kps_result[image_id, j, 2] = hm_j.max() / 255 

                vis=False
                crop_info = crop_infos[image_id - start_id,:]
                area = (crop_info[2] - crop_info[0]) * (crop_info[3] - crop_info[1])
                if vis and np.any(kps_result[image_id,:,2]) > 0.9 and area > 96**2:
                    tmpimg = imgs[image_id-start_id].copy()
                    tmpimg = cfg.denormalize_input(tmpimg)
                    tmpimg = tmpimg.astype('uint8')
                    tmpkps = np.zeros((3,cfg.num_kps))
                    tmpkps[:2,:] = kps_result[image_id,:,:2].transpose(1,0)
                    tmpkps[2,:] = kps_result[image_id,:,2]
                    _tmpimg = tmpimg.copy()
                    _tmpimg = cfg.vis_keypoints(_tmpimg, tmpkps)
                    cv2.imwrite(osp.join(cfg.vis_dir, str(img_id) + '_output.jpg'), _tmpimg)
                    img_id += 1

                # map back to original images
                for j in range(cfg.num_kps):
                    kps_result[image_id, j, 0] = kps_result[image_id, j, 0] / cfg.input_shape[1] * (\
                    crop_infos[image_id - start_id][2] - crop_infos[image_id - start_id][0]) + crop_infos[image_id - start_id][0]
                    kps_result[image_id, j, 1] = kps_result[image_id, j, 1] / cfg.input_shape[0] * (\
                    crop_infos[image_id - start_id][3] - crop_infos[image_id - start_id][1]) + crop_infos[image_id - start_id][1]
                
                area_save[image_id] = (crop_infos[image_id - start_id][2] - crop_infos[image_id - start_id][0]) * (crop_infos[image_id - start_id][3] - crop_infos[image_id - start_id][1])
                
        #vis
        if vis_kps and np.any(kps_result[:,:,2] > 0.8):
            tmpimg = cv2.imread(os.path.join(cfg.img_path, cropped_data[0]['imgpath']))
            tmpimg = tmpimg.astype('uint8')
            for i in range(len(kps_result)):
                tmpkps = np.zeros((3,cfg.num_kps))
                tmpkps[:2,:] = kps_result[i, :, :2].transpose(1,0)
                tmpkps[2,:] = kps_result[i, :, 2]
                tmpimg = cfg.vis_keypoints(tmpimg, tmpkps)
            cv2.imwrite(osp.join(cfg.vis_dir, str(img_id2) + '.jpg'), tmpimg)
            img_id2 += 1
        
        score_result = np.copy(kps_result[:, :, 2])
        kps_result[:, :, 2] = 1
        kps_result = kps_result.reshape(-1,cfg.num_kps*3)
       
        # rescoring and oks nms
        if cfg.dataset == 'COCO' or cfg.dataset == 'JTA':
            rescored_score = np.zeros((len(score_result)))
            for i in range(len(score_result)):
                score_mask = score_result[i] > cfg.score_thr
                if np.sum(score_mask) > 0:
                    rescored_score[i] = np.mean(score_result[i][score_mask]) * cropped_data[i]['score']
            score_result = rescored_score
            keep = oks_nms(kps_result, score_result, area_save, cfg.oks_nms_thr,sigmas)
            if len(keep) > 0 :
                kps_result = kps_result[keep,:]
                score_result = score_result[keep]
                area_save = area_save[keep]
        elif cfg.dataset == 'PoseTrack':
            keep = oks_nms(kps_result, np.mean(score_result,axis=1), area_save, cfg.oks_nms_thr)
            if len(keep) > 0 :
                kps_result = kps_result[keep,:]
                score_result = score_result[keep,:]
                area_save = area_save[keep]
        
        # save result
        for i in range(len(kps_result)):
            if cfg.dataset == 'COCO' or cfg.dataset == 'JTA':
                result = dict(image_id=im_info['image_id'], category_id=1, score=float(round(score_result[i], 4)),
                             keypoints=kps_result[i].round(3).tolist())
            elif cfg.dataset == 'PoseTrack':
                result = dict(image_id=im_info['image_id'], category_id=1, track_id=0, scores=score_result[i].round(4).tolist(),
                              keypoints=kps_result[i].round(3).tolist())
            elif cfg.dataset == 'MPII':
                result = dict(image_id=im_info['image_id'], scores=score_result[i].round(4).tolist(),
                              keypoints=kps_result[i].round(3).tolist())
            # elif cfg.dataset == 'JTA':
            #     result = dict(image_id=im_info['image_id'],category_id=1, scores=score_result[i].round(4).tolist(),
            #                   keypoints=kps_result[i].round(3).tolist())
            dump_results.append(result)


    return dump_results


def test(test_model,vis_kps,run_nr,make_stats,gpu_id):
    cfg.set_args(gpu_id)
    if make_stats:

        base_log_dir = cfg.log_dir
        # due to cocoeval format for keypoints
        index = np.asarray([0,1,2,5,6,7,8,9,10],dtype=np.int32)
        header = ["AP@[IoU=0.5:0.95]",
                  "AP@[IoU=0.5]",
                  "AP@[IoU=0.75]",
                  "AR@[IoU=0.5:0.95]",
                  "AR@[IoU=0.5]",
                  "AR@[IoU=0.75]",
                  "AP@[IoU=0.5:0.95,easy]",
                  "AP@[IoU=0.5:0.95,medium]",
                  "AP@[IoU=0.5:0.95,hard]"]


        available_runs = glob(os.path.join(cfg.model_dump_dir,"run_*"))

        if not all([os.path.isdir(run) for run in available_runs]):
            print("Not all subdirs are valid dirs, exit.")
            exit(0)

        print("Found {} runs. Appply testing on all.".format(len(available_runs)))

        results = np.zeros([len(available_runs),index.shape[0]])
        for nr,run in enumerate(available_runs):

            eval_dir = osp.join(cfg.result_dir, "run_{}".format(nr+1))
            if not osp.isdir(eval_dir):
                os.makedirs(eval_dir)

            snapshots = glob(osp.join(run,"snapshot_*.ckpt.*"))

            nr_snapshots = len(snapshots) // 3

            stats = np.zeros([nr_snapshots,index.shape[0]])

            sn_index = ["@ckpt#{}".format(n+1) for n in range(nr_snapshots)]

            for sn_nr in range(nr_snapshots):
                # annotation load
                d = Dataset(train=False)
                annot = d.load_annot(cfg.testset)
                gt_img_id = d.load_imgid(annot)
                cfg.log_dir = osp.join(base_log_dir,"eval","run_{}".format(nr+1),"sn_{}".format(sn_nr+1))

                if not osp.isdir(cfg.log_dir):
                    os.makedirs(cfg.log_dir)

                # human bbox load
                if cfg.useGTbbox and cfg.testset in ['train', 'val']:
                    if cfg.testset == 'train':
                        dets = d.load_test_data(score=True)
                    else:
                        dets = d.load_val_data_with_annot()
                    dets.sort(key=lambda x: (x['image_id']))
                else:
                    with open(cfg.human_det_path, 'r') as f:
                        dets = json.load(f)
                    dets = [i for i in dets if i['image_id'] in gt_img_id]
                    dets = [i for i in dets if i['category_id'] == 1]
                    dets = [i for i in dets if i['score'] > 0]
                    dets.sort(key=lambda x: (x['image_id'], x['score']), reverse=True)

                    img_id = []
                    for i in dets:
                        img_id.append(i['image_id'])
                    imgname = d.imgid_to_imgname(annot, img_id, cfg.testset)
                    for i in range(len(dets)):
                        dets[i]['imgpath'] = imgname[i]

                det_range = [0, len(dets)]


                tester = Tester(Model(), cfg)
                tester.load_weights(sn_nr+1, model_dump_dir=run)
                result = test_net(tester, dets, det_range, int(gpu_id), d.sigmas, vis_kps)


                tester.sess.close()
                tf.reset_default_graph()

                del tester

                # result = func(gpu_id)
                # evaluation
                cocoeval = d.evaluation_stats(result, annot, eval_dir)

                stats[sn_nr,:] = cocoeval.stats[index]

                print("Finished! Actual stats are:")
                print(cocoeval.stats[index])

                del d


            # get best result with respect to ap 0.5:0.95
            best_id = np.argmax(stats[:,0])
            results[nr,:] = stats[best_id,:]

            # store actual results to csv
            df = pd.DataFrame(data=stats,columns=header,index=sn_index)
            df.to_csv(osp.join(eval_dir,"stats_all.csv"),index=True)

        # add f1-score computation
        header +=["F1@[IoU=0.5:0.95]"]
        #compute f1-scores for every run
        f1_scores = 2*(results[:,0]*results[:,3])/(results[:,0]+results[:,3])
        f1_scores = np.expand_dims(f1_scores,axis=1)
        results = np.concatenate([results,f1_scores],axis=1)

        # compute stats and store to csv file
        mean = np.mean(results,axis=0)
        if len(available_runs)<2:
            std = np.zeros_like(mean,dtype=np.float)
        else:
            # compute unbiased variant of std, as the number of samples is not large
            std = np.std(results,axis=0,ddof=1)
        best = results[np.argmax(results[:,0]),:]
        out = np.stack([best,mean,std],axis=1)
        out=out.transpose()
        out_df = pd.DataFrame(data=out,columns=header,index=["best","mean","stddev"])
        out_df.to_csv(osp.join(cfg.result_dir,"results_final.csv"),index=True)


        #store all results
        run_idx = ["run_{}".format(r+1) for r in range(len(available_runs))]
        all_df = pd.DataFrame(data=results,columns=header,index=run_idx)
        all_df.to_csv(osp.join(cfg.result_dir,"results_list.csv"),index=True)

        # store final results
        header.append(["mean",""])




    else:

        # annotation load
        d = Dataset(train=False)
        annot = d.load_annot(cfg.testset)
        gt_img_id = d.load_imgid(annot)

        # human bbox load
        if cfg.useGTbbox and cfg.testset in ['train', 'val']:
            if cfg.testset == 'train':
                dets = d.load_test_data(score=True)
            else:
                dets = d.load_val_data_with_annot()
            dets.sort(key=lambda x: (x['image_id']))
        else:
            with open(cfg.human_det_path, 'r') as f:
                dets = json.load(f)
            dets = [i for i in dets if i['image_id'] in gt_img_id]
            dets = [i for i in dets if i['category_id'] == 1]
            dets = [i for i in dets if i['score'] > 0]
            dets.sort(key=lambda x: (x['image_id'], x['score']), reverse=True)

            img_id = []
            for i in dets:
                img_id.append(i['image_id'])
            imgname = d.imgid_to_imgname(annot, img_id, cfg.testset)
            for i in range(len(dets)):
                dets[i]['imgpath'] = imgname[i]

        det_range = [0, len(dets)]

        model_dump_dir = osp.join(cfg.model_dump_dir, "run_{}".format(run_nr))

        assert osp.isdir(model_dump_dir)
        # evaluation
        cfg.set_args(gpu_id)
        tester = Tester(Model(), cfg)
        tester.load_weights(test_model,model_dump_dir=model_dump_dir)
        result = test_net(tester, dets, det_range, int(gpu_id),d.sigmas,vis_kps)

        eval_dir = osp.join(cfg.result_dir,"run_{}".format(run_nr))
        if not osp.isdir(eval_dir):
            os.makedirs(eval_dir)

        d.evaluation(result, annot, eval_dir, cfg.testset)

if __name__ == '__main__':
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--gpu', type=str, dest='gpu_ids')
        parser.add_argument('--test_epoch', type=str, dest='test_epoch',default="15")
        parser.add_argument("--vis",dest='vis_kps', action='store_true')
        parser.add_argument("--run_nr",type=int,default=1,dest="run_nr")
        parser.add_argument("--stats",dest="stats",action="store_true")
        args = parser.parse_args()

        # test gpus
        if not args.gpu_ids:
            args.gpu_ids = str(np.argmin(mem_info()))

        if '-' in args.gpu_ids:
            gpus = args.gpu_ids.split('-')
            gpus[0] = 0 if not gpus[0].isdigit() else int(gpus[0])
            gpus[1] = len(mem_info()) if not gpus[1].isdigit() else int(gpus[1]) + 1
            args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))
        
        assert args.test_epoch, 'Test epoch is required.'
        return args

    global args
    args = parse_args()
    test(int(args.test_epoch),args.vis_kps,args.run_nr,args.stats,args.gpu_ids)
