import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

from config import get_config

from datasets.dataset import CenterCropGenerator
from datasets.dataset import USdatasetCls, USdatasetSeg
from PIL import Image

from utils import omni_seg_test
from sklearn.metrics import accuracy_score
# 在文件开头添加
USE_SAMUS = True  # 与训练时保持一致

if USE_SAMUS:
    from networks.samus_adapter import SAMUSAdapter as ModelClass
else:
    from networks.omni_vision_transformer import OmniVisionTransformer as ModelClass


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='./data', help='root dir for data')
parser.add_argument('--output_dir', type=str, help='output dir')
parser.add_argument('--max_epochs', type=int, default=200, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_saveout', action="store_true", help='whether to save results during inference')
parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--cfg', type=str, default="configs/tiny_samus_config.yaml",
                    metavar="FILE", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                    'full: cache all data, '
                    'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

parser.add_argument('--prompt', action='store_true', help='using prompt')

args = parser.parse_args()
config = get_config(args)



def inference(args, model, test_save_path=None):
    import csv
    import time

    if not os.path.exists("exp_out/result.csv"):
        with open("exp_out/result.csv", 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['dataset', 'task', 'metric', 'time'])

    seg_test_set = [
        "BUS-BRA",
        "BUSIS",
        "BUSI",
        "CAMUS",
        "DDTI",
        "Fetal_HC",
        "KidneyUS",
        "private_Thyroid",
        "private_Kidney",
        "private_Fetal_Head",
        "private_Cardiac",
        "private_Breast_luminal",
        "private_Breast",
    ]

    for dataset_name in seg_test_set:
        num_classes = 2
        db_test = USdatasetSeg(
            base_dir=os.path.join(args.root_path, "segmentation", dataset_name),
            split="test",
            list_dir=os.path.join(args.root_path, "segmentation", dataset_name),
            transform=CenterCropGenerator(output_size=[args.img_size, args.img_size]),
            prompt=args.prompt
        )
        testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
        logging.info("{} test iterations per epoch".format(len(testloader)))
        model.eval()

        metric_list = 0.0
        
        count_matrix = np.ones((len(db_test), num_classes-1))
        for i_batch, sampled_batch in tqdm(enumerate(testloader)):
            image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
            if args.prompt:
                position_prompt = torch.tensor(np.array(sampled_batch['position_prompt'])).permute([1, 0]).float()
                task_prompt = torch.tensor(np.array([[1], [0]])).permute([1, 0]).float()
                type_prompt = torch.tensor(np.array(sampled_batch['type_prompt'])).permute([1, 0]).float()
                nature_prompt = torch.tensor(np.array(sampled_batch['nature_prompt'])).permute([1, 0]).float()
                metric_i = omni_seg_test(image, label, model,
                                         classes=num_classes,
                                         test_save_path=test_save_path,
                                         case=case_name,
                                         prompt=args.prompt,
                                         type_prompt=type_prompt,
                                         nature_prompt=nature_prompt,
                                         position_prompt=position_prompt,
                                         task_prompt=task_prompt
                                         )
            else:
                image_path=db_test.data_dir +"/imgs/"+ case_name
                mask_path=db_test.data_dir +"/masks/"+ case_name
                metric_i,mask_img = omni_seg_test(image, label, model,
                                         classes=num_classes,
                                         test_save_path=test_save_path,
                                         case=case_name,img_path=image_path)
                # 合成并保存：原图 | 真值掩码 | 预测掩码
                if test_save_path is not None:
                    try:
                        os.makedirs(test_save_path, exist_ok=True)
                        base = os.path.splitext(os.path.basename(image_path))[0]

                        # 读取原图
                        if os.path.exists(image_path):
                            orig = Image.open(image_path).convert("RGB")
                        else:
                            orig = Image.new("RGB", mask_img.size, (0, 0, 0))

                        # 读取真值掩码（若存在）
                        if mask_path and os.path.exists(mask_path):
                            gt = Image.open(mask_path).convert("L")
                        else:
                            gt = None

                        # 目标尺寸以原图为准
                        target_size = (224,224)  # (width, height)

                        def to_rgb_and_resize(im, size):
                            im = im.resize(size, Image.NEAREST)
                            if im.mode != "RGB":
                                im = im.convert("RGB")
                            return im

                        orig_r = to_rgb_and_resize(orig, target_size)
                        gt_r = to_rgb_and_resize(gt, target_size) if gt is not None else Image.new("RGB", target_size, (0, 0, 0))
                        pred_r = to_rgb_and_resize(mask_img, target_size)

                         # 给三张图上方加标签
                        try:
                            font_size = 18
                            try:
                                font = ImageFont.truetype("DejaVuSans.ttf", font_size)
                            except Exception:
                                font = ImageFont.load_default()

                            def add_label_top(im: Image.Image, text: str, font: ImageFont.ImageFont, pad: int = 6):
                                # 兼容不同 Pillow 版本的文本测量方法
                                try:
                                    dummy = Image.new("RGB", (1, 1))
                                    draw = ImageDraw.Draw(dummy)
                                    # 优先使用 textbbox（更精确）
                                    bbox = draw.textbbox((0, 0), text, font=font)
                                    text_w = bbox[2] - bbox[0]
                                    text_h = bbox[3] - bbox[1]
                                except Exception:
                                    try:
                                        # 备用方法
                                        text_w, text_h = draw.textsize(text, font=font)
                                    except Exception:
                                        # 最后回退到 getmask（几乎总可用）
                                        mask = font.getmask(text)
                                        text_w, text_h = mask.size

                                new_h = im.height + text_h + pad
                                new_im = Image.new("RGB", (im.width, new_h), (255, 255, 255))
                                new_im.paste(im, (0, text_h + pad))
                                draw2 = ImageDraw.Draw(new_im)
                                draw2.text(((im.width - text_w) // 2, pad // 2), text, font=font, fill=(0, 0, 0))
                                return new_im
                            orig_l = add_label_top(orig_r, "orig", font)
                            gt_l = add_label_top(gt_r, "gt_mask", font)
                            pred_l = add_label_top(pred_r, "pred_mask", font)
                        except Exception as e:
                            # 回退到不带标签的图
                            print(f"绘制标签失败: {e}")
                            orig_l, gt_l, pred_l = orig_r, gt_r, pred_r

                        # 横向拼接三张图（带标签）
                        total_w = target_size[0] * 3
                        h = orig_l.height
                        composite = Image.new("RGB", (total_w, h), (255, 255, 255))
                        composite.paste(orig_l, (0, 0))
                        composite.paste(gt_l, (target_size[0], 0))
                        composite.paste(pred_l, (target_size[0] * 2, 0))

                        out_dir = os.path.join(test_save_path, dataset_name)
                        os.makedirs(out_dir, exist_ok=True)
                        save_path = os.path.join(out_dir, f"{base}_orig_gt_pred.png")
                        composite.save(save_path)
                    except Exception as e:
                        print(f"保存合成图失败: {e}")

                                         


            zero_label_flag = False
            for i in range(1, num_classes):
                if not metric_i[i-1][1]:
                    count_matrix[i_batch, i-1] = 0
                    zero_label_flag = True
            metric_i = [element[0] for element in metric_i]
            metric_list += np.array(metric_i)
            logging.info('idx %d case %s mean_dice %f' %
                         (i_batch, case_name, np.mean(metric_i, axis=0)))
            logging.info("This case has zero label: %s" % zero_label_flag)

        metric_list = metric_list / (count_matrix.sum(axis=0) + 1e-6)
        for i in range(1, num_classes):
            logging.info('Mean class %d mean_dice %f' % (i, metric_list[i-1]))
        performance = np.mean(metric_list, axis=0)
        logging.info('Testing performance in best val model: mean_dice : %f' % (performance))

        with open("exp_out/result.csv", 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if args.prompt:
                writer.writerow([dataset_name, 'omni_seg_prompt@'+args.output_dir, performance,
                                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())])
            else:
                writer.writerow([dataset_name, 'omni_seg@'+args.output_dir, performance,
                                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())])

    cls_test_set = [
        "Appendix",
        "BUS-BRA",
        "BUSI",
        "Fatty-Liver",
        "private_Liver",
        "private_Breast_luminal",
        "private_Breast",
        "private_Appendix",
    ]

    for dataset_name in cls_test_set:
        if dataset_name == "private_Breast_luminal":
            num_classes = 4
        else:
            num_classes = 2
        db_test = USdatasetCls(
            base_dir=os.path.join(args.root_path, "classification", dataset_name),
            split="test",
            list_dir=os.path.join(args.root_path, "classification", dataset_name),
            transform=CenterCropGenerator(output_size=[args.img_size, args.img_size]),
            prompt=args.prompt
        )

        testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
        logging.info("{} test iterations per epoch".format(len(testloader)))
        model.eval()

        label_list = []
        prediction_list = []
        for i_batch, sampled_batch in tqdm(enumerate(testloader)):
            image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
            if args.prompt:
                position_prompt = torch.tensor(np.array(sampled_batch['position_prompt'])).permute([1, 0]).float()
                task_prompt = torch.tensor(np.array([[0], [1]])).permute([1, 0]).float()
                type_prompt = torch.tensor(np.array(sampled_batch['type_prompt'])).permute([1, 0]).float()
                nature_prompt = torch.tensor(np.array(sampled_batch['nature_prompt'])).permute([1, 0]).float()
                with torch.no_grad():
                    output = model((image.cuda(), position_prompt.cuda(), task_prompt.cuda(),
                                   type_prompt.cuda(), nature_prompt.cuda()))
            else:
                with torch.no_grad():
                    output = model(image.cuda())

            if num_classes == 4:
                logits = output[2]
            else:
                logits = output[1]

            prediction = np.argmax(torch.softmax(logits, dim=1).data.cpu().numpy())
            logging.info('idx %d case %s label: %d predict: %d' % (i_batch, case_name, label, prediction))

            label_list.append(label.numpy())
            prediction_list.append(prediction)

        label_list = np.array(label_list)
        prediction_list = np.array(prediction_list)
        for i in range(num_classes):
            logging.info('class %d acc %f' % (i, accuracy_score(
                (label_list == i).astype(int), (prediction_list == i).astype(int))))
        performance = accuracy_score(label_list, prediction_list)
        logging.info('Testing performance in best val model: acc : %f' % (performance))

        with open("exp_out/result.csv", 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if args.prompt:
                writer.writerow([dataset_name, 'omni_cls_prompt@'+args.output_dir, performance,
                                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())])
            else:
                writer.writerow([dataset_name, 'omni_cls@'+args.output_dir, performance,
                                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())])


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
 
    net = ModelClass(
        config,
        prompt=args.prompt,
    ).cuda()
    net.load_from(config)

    snapshot = os.path.join(args.output_dir, 'best_model.pth')
    if not os.path.exists(snapshot):
        snapshot = snapshot.replace('best_model', 'epoch_'+str(args.max_epochs-1))

    device = torch.device("cuda")
    model = net.to(device=device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    torch.distributed.init_process_group(backend="nccl", init_method='env://', world_size=1, rank=0)
    model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)

    import copy
    pretrained_dict = torch.load(snapshot, map_location=device)
    full_dict = copy.deepcopy(pretrained_dict)
    for k, v in pretrained_dict.items():
        if "module." not in k:
            full_dict["module."+k] = v
            del full_dict[k]

    msg = model.load_state_dict(full_dict)

    print("self trained swin unet", msg)
    snapshot_name = snapshot.split('/')[-1]

    logging.basicConfig(filename=args.output_dir+"/"+"test_result.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_saveout:
        args.test_save_dir = os.path.join(args.output_dir, "predictions")
        test_save_path = args.test_save_dir
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, net, test_save_path)
