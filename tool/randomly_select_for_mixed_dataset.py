import os
import glob
import random
import shutil
import json

dataset_path_1 = '/home/ies/kong/sympose_image_reduced'
# dataset_path_2 = '/home/ies/kong/adapted_images/cityscapes_standard_cyclegan_bs1_gfdim32_dfdim64_start2019-06-17-09-16-34/images'
dataset_path_3 = '/home/ies/kong/adapted_images/iosb_standard_cyclegan_bs1_gfdim32_dfdim64_start2019-06-17-09-11-54/images'
# dataset_path_4 = '/home/ies/kong/adapted_images/w10_standard_cyclegan_bs1_gfdim32_dfdim64_start2019-06-13-19-54-19/images'
save_path = '/home/ies/kong/TF-SimpleHumanPose/data/JTA/SyMPose_IOSB_CrowdPose/images'
seq_list = ['seq_1', 'seq_3', 'seq_7', 'seq_9', 'seq_12', 'seq_14', 'seq_23']

def select_from_images():
    for dir in os.listdir(dataset_path_1):
        if not os.path.isdir(os.path.join(save_path, dir)):
            os.makedirs(os.path.join(save_path, dir))

    # for each in glob.glob(os.path.join(dataset_path_1, '*', '*.jpeg')):
    #     print(each)
    #     count += 1
    #     if count >= 225:
    #         break

    for seq in seq_list:

        image_list = [f for f in os.listdir(os.path.join(dataset_path_1, seq)) if f.endswith('.jpeg')]
        file_nr = 600

        for i in range(file_nr // 2):
            random_file = random.choice(image_list)
            shutil.copy(os.path.join(dataset_path_1, seq, random_file), os.path.join(save_path, seq, random_file))
            image_list.remove(random_file)

        # for i in range(file_nr // 4):
        #     random_file = random.choice(image_list)
        #     shutil.copy(os.path.join(dataset_path_2, seq, random_file), os.path.join(save_path, seq, random_file))
        #     image_list.remove(random_file)

        for i in range(file_nr // 2):
            random_file = random.choice(image_list)
            shutil.copy(os.path.join(dataset_path_3, seq, random_file), os.path.join(save_path, seq, random_file))
            image_list.remove(random_file)

        # for i in range(len(image_list)):
        #     random_file = random.choice(image_list)
        #     shutil.copy(os.path.join(dataset_path_4, seq, random_file), os.path.join(save_path, seq, random_file))
        #     image_list.remove(random_file)

def select_from_annotations():
    in_file_path = '/home/ies/kong/TF-SimpleHumanPose/data/JTA/SyMPose/annotations/train_jta.json'
    save_path = '/home/ies/kong/TF-SimpleHumanPose/data/JTA/SyMPose_IOSB_CrowdPose/annotations/'
    with open(in_file_path, 'r') as in_json_file:
        json_obj_list = json.load(in_json_file)
        # print(json_obj_list.keys())
        # for obj in json_obj_list:
        #     print(json_obj_list[obj][0])
        image_id = []
        dic = {'images': [], 'annotations': []}
        for i in range(4200):
            random_file = random.choice(json_obj_list['images'])
            dic['images'].append(random_file)
            image_id.append(random_file['id'])
            json_obj_list['images'].remove(random_file)

        for item in json_obj_list['annotations']:
            if item['image_id'] in image_id:
                dic['annotations'].append(item)

        print(len(dic['images']))
        with open(os.path.join(save_path, 'train_jta.json'), 'w') as out_json_file:
            json.dump({'images': dic['images'], 'annotations': dic['annotations'], 'categories': json_obj_list['categories']},
                      out_json_file)
        out_json_file.close()
    in_json_file.close()

def after_select_from_annotations():
    in_file_path = '/home/ies/kong/TF-SimpleHumanPose/data/JTA/SyMPose_IOSB_CrowdPose/annotations/train_jta.json'
    sympose_path = '/home/ies/kong/sympose_image_reduced'
    iosb_path = '/home/ies/kong/adapted_images/iosb_standard_cyclegan_bs1_gfdim32_dfdim64_start2019-06-17-09-11-54'
    destination = '/home/ies/kong/TF-SimpleHumanPose/data/JTA/SyMPose_IOSB_CrowdPose/'
    counter = 0
    with open(in_file_path, 'r') as in_json_file:
        json_obj_list = json.load(in_json_file)
        for item in json_obj_list['images']:
            if counter % 2 == 0:
                shutil.copy(os.path.join(sympose_path, item['file_name'][7:]), os.path.join(destination, item['file_name']))
            else:
                shutil.copy(os.path.join(iosb_path, item['file_name']), os.path.join(destination, item['file_name']))
    in_json_file.close()


def combine_2_annotations():
    anno_1 = '/home/ies/kong/TF-SimpleHumanPose/data/JTA/SyMPose_IOSB_CrowdPose/annotations/train_jta_sympose_iosb.json'
    anno_2 = '/home/ies/kong/CrowdPose/crowdpose_train_reduced.json'
    save_path = '/home/ies/kong/TF-SimpleHumanPose/data/JTA/SyMPose_IOSB_CrowdPose/annotations/train_jta_combine.json'
    dic = {'images': [], 'annotations': []}
    with open(anno_1, 'r') as in_json_file:
        json_obj_list = json.load(in_json_file)
        dic['images'].extend(json_obj_list['images'])
        dic['annotations'].extend(json_obj_list['annotations'])
        with open(anno_2, 'r') as anno_2_file:
            anno_2_dic = json.load(anno_2_file)
            dic['images'].extend(anno_2_dic['images'])
            dic['annotations'].extend(anno_2_dic['annotations'])
            print(len(dic['images']))
            with open(save_path, 'w') as out_json_file:
                json.dump({'images': dic['images'], 'annotations': dic['annotations'],
                           'categories': json_obj_list['categories']},
                          out_json_file)
            out_json_file.close()
        anno_2_file.close()
    in_json_file.close()


if __name__ == '__main__':
    # select_from_annotations()
    # after_select_from_annotations()
    combine_2_annotations()
