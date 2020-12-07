import cv2
import glob
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class YOLOv4_Dataset(Dataset):
    @torch.no_grad()
    def __init__(self, root_dir, model, params, train=False):
        """
        Args:
            root_dir (string): Directory with all the images.
            model (callable): Model to be applied on a sample to get label.
        """
        self.root_dir = root_dir
        self.model = model
        self.params = params
        self.class_names = self._get_class("experiments/yolov4_tiny_distill/yolov4_teacher/coco.names")
        self.classes = None
        self.train = train

        self.model.eval()

        # put all images into np array
        files = glob.glob(self.root_dir + "/*")
        self.imgs = np.array([cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB) for file in files], dtype=object)
        self.img_shapes = np.array([img.shape for img in self.imgs]) # HWC
        print(f'imgs shape: {self.imgs.shape}')
        # print(f'img sizes: {self.img_shapes}')

        # preprocess data for input to teacher model
        images = [np.array(cv2.resize(img, (params.width, params.height))) for img in self.imgs]
        images = np.stack(images, axis=0)
        images = np.divide(images, 255, dtype=np.float32)
        images = torch.from_numpy(images.transpose(0, 3, 1, 2))

        # get all teacher labels
        self.targets = []
        imgs_split = torch.split(images, params.teacher_bs)
        sizes_split = torch.split(torch.from_numpy(self.img_shapes), params.teacher_bs)
        boxes = []
        confs = []
        for i, imgs_smol in enumerate(imgs_split):
            imgs_smol = imgs_smol.cuda()
            teacher_output = self.model(imgs_smol)
            boxes.append(teacher_output[0].cpu().numpy())
            confs.append(teacher_output[1].cpu().numpy())
        boxes = np.concatenate(boxes, axis=0)
        confs = np.concatenate(confs, axis=0)
        output = [boxes, confs]

        self.targets = self._post_processing(output)
        # print(f'targets len: {len(self.targets)}')

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index

        Returns:
            tuple: (image, target)
        """

        if not self.train:
            return self._get_val_item(idx)

        img, target = self.imgs[idx], self.targets[idx]

        dhue, dsat, dexp, flip, blur = 0, 0, 0, 0, 0
        gaussian_noise = 0

        oh, ow, oc = img.shape
        dh, dw, dc = np.array(np.array([oh, ow, oc]) * self.params.jitter, dtype=np.int)

        dhue = self._rand_uniform_strong(-self.params.hue, self.params.hue)
        dsat = self._rand_scale(self.params.saturation)
        dexp = self._rand_scale(self.params.exposure)

        pleft = random.randint(-dw, dw)
        pright = random.randint(-dw, dw)
        ptop = random.randint(-dh, dh)
        pbot = random.randint(-dh, dh)

        flip = random.randint(0, 1) if self.params.flip else 0

        if self.params.blur:
            tmp_blur = random.randint(0, 2)  # 0 - disable, 1 - blur background, 2 - blur the whole image
            if tmp_blur == 0:
                blur = 0
            elif tmp_blur == 1:
                blur = 1
            else:
                blur = self.params.blur

        if self.params.gaussian and random.randint(0, 1):
            gaussian_noise = self.params.gaussian
        else:
            gaussian_noise = 0

        if self.params.letter_box:
            img_ar = ow / oh
            net_ar = self.params.width / self.params.height
            result_ar = img_ar / net_ar
            # print(" ow = %d, oh = %d, w = %d, h = %d, img_ar = %f, net_ar = %f, result_ar = %f \n", ow, oh, w, h, img_ar, net_ar, result_ar);
            if result_ar > 1:  # sheight - should be increased
                oh_tmp = ow / net_ar
                delta_h = (oh_tmp - oh) / 2
                ptop = ptop - delta_h
                pbot = pbot - delta_h
                # print(" result_ar = %f, oh_tmp = %f, delta_h = %d, ptop = %f, pbot = %f \n", result_ar, oh_tmp, delta_h, ptop, pbot);
            else:  # swidth - should be increased
                ow_tmp = oh * net_ar
                delta_w = (ow_tmp - ow) / 2
                pleft = pleft - delta_w
                pright = pright - delta_w
                # printf(" result_ar = %f, ow_tmp = %f, delta_w = %d, pleft = %f, pright = %f \n", result_ar, ow_tmp, delta_w, pleft, pright);

        swidth = ow - pleft - pright
        sheight = oh - ptop - pbot

        truth, min_w_h = self._fill_truth_detection(np.array(target), self.params.boxes, self.params.num_classes, flip, pleft, ptop, swidth, sheight, self.params.width, self.params.height)
        if (min_w_h / 8) < blur and blur > 1:  # disable blur if one of the objects is too small
            blur = min_w_h / 8

        ai = self._image_data_augmentation(img, self.params.width, self.params.height, pleft, ptop, swidth, sheight, flip, dhue, dsat, dexp, gaussian_noise, blur, truth)

        out_img = ai
        out_bboxes = truth

        out_bboxes1 = np.zeros([self.params.boxes, 5])
        out_bboxes1[:min(out_bboxes.shape[0], self.params.boxes)] = out_bboxes[:min(out_bboxes.shape[0], self.params.boxes)]

        out_img = np.divide(out_img, 255, dtype=np.float32)
        out_img = torch.from_numpy(out_img.transpose(2, 0, 1))

        out_bboxes1 = torch.from_numpy(out_bboxes1)

        return out_img, out_bboxes1

    def _get_val_item(self, idx):
        img = self.imgs[idx]
        bboxes_with_cls_id = np.array(self.targets[idx])
        num_objs = len(bboxes_with_cls_id)

        target = {}
        # boxes to coco format
        boxes = bboxes_with_cls_id[..., :4]
        boxes[..., 2:] = boxes[..., 2:] - boxes[..., :2]  # box width, box height

        target['image_id'] = idx
        target['boxes'] = boxes

        if num_objs > 0:
            target['labels'] = bboxes_with_cls_id[..., 4].flatten()
            target['area'] = (target['boxes'][:,3])*(target['boxes'][:,2])
            target['iscrowd'] = np.zeros((num_objs,))

        return img, target

    @staticmethod
    def _get_class(classes_path):
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _post_processing(self, output):
        # [batch, num, 1, 4]
        box_array = output[0]
        # [batch, num, num_classes]
        confs = output[1]

        if type(box_array).__name__ != 'ndarray':
            box_array = box_array.cpu().numpy()
            confs = confs.cpu().numpy()

        num_classes = confs.shape[2]

        # [batch, num, 4]
        box_array = box_array[:, :, 0]

        # [batch, num, num_classes] --> [batch, num]
        max_conf = np.max(confs, axis=2)
        max_id = np.argmax(confs, axis=2)

        bboxes_batch = []
        for i in range(box_array.shape[0]):
           
            argwhere = max_conf[i] > self.params.teacher_bbox_thresh
            l_box_array = box_array[i, argwhere, :]
            l_max_conf = max_conf[i, argwhere]
            l_max_id = max_id[i, argwhere]

            bboxes = []
            # nms for each class
            for j in range(num_classes):

                cls_argwhere = l_max_id == j
                ll_box_array = l_box_array[cls_argwhere, :]
                ll_max_conf = l_max_conf[cls_argwhere]
                ll_max_id = l_max_id[cls_argwhere]

                keep = self._nms_cpu(ll_box_array, ll_max_conf)
                
                if (keep.size > 0):
                    ll_box_array = ll_box_array[keep, :]
                    ll_max_conf = ll_max_conf[keep]
                    ll_max_id = ll_max_id[keep]

                    for k in range(ll_box_array.shape[0]):
                        bboxes.append([ll_box_array[k, 0], ll_box_array[k, 1], ll_box_array[k, 2], ll_box_array[k, 3], ll_max_conf[k], ll_max_conf[k], ll_max_id[k]])
            
            bboxes_batch.append(bboxes)

        return self._postprocess(bboxes_batch, box_format='ltrbc', classes=self.classes)

    def _nms_cpu(self, boxes, confs, min_mode=False):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = confs.argsort()[::-1]

        keep = []
        while order.size > 0:
            idx_self = order[0]
            idx_other = order[1:]

            keep.append(idx_self)

            xx1 = np.maximum(x1[idx_self], x1[idx_other])
            yy1 = np.maximum(y1[idx_self], y1[idx_other])
            xx2 = np.minimum(x2[idx_self], x2[idx_other])
            yy2 = np.minimum(y2[idx_self], y2[idx_other])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            if min_mode:
                over = inter / np.minimum(areas[order[0]], areas[order[1:]])
            else:
                over = inter / (areas[order[0]] + areas[order[1:]] - inter)

            inds = np.where(over <= self.params.teacher_nms_thresh)[0]
            order = order[inds + 1]
        
        return np.array(keep)

    def _postprocess(self, boxes, box_format='ltrb', classes=None, buffer_ratio=0.0):
        detections = []

        for i, frame_bbs in enumerate(boxes):
            im_height, im_width, _ = self.img_shapes[i]
            frame_dets = []
            for box in frame_bbs:
                cls_conf = box[5]
                cls_id = box[6]
                cls_name = self.class_names[cls_id]

                if classes is not None and cls_name not in classes:
                    continue

                left = int(box[0] * im_width)
                top = int(box[1] * im_height)
                right = int(box[2] * im_width)
                bottom = int(box[3] * im_height)
                
                width = right - left + 1
                height = bottom - top + 1
                width_buffer = width * buffer_ratio
                height_buffer = height * buffer_ratio

                top = max( 0.0, top - 0.5*height_buffer )
                left = max( 0.0, left - 0.5*width_buffer )
                bottom = min( im_height - 1.0, bottom + 0.5*height_buffer )
                right = min( im_width - 1.0, right + 0.5*width_buffer )

                box_infos = []
                for f in box_format:
                    if f == 't':
                        box_infos.append( int(round(top)) ) 
                    elif f == 'l':
                        box_infos.append( int(round(left)) )
                    elif f == 'b':
                        box_infos.append( int(round(bottom)) )
                    elif f == 'r':
                        box_infos.append( int(round(right)) )
                    elif f == 'w':
                        box_infos.append( int(round(width+width_buffer)) )
                    elif f == 'h':
                        box_infos.append( int(round(height+height_buffer)) )
                    elif f == 'c':
                        box_infos.append( int(cls_id) )
                    else:
                        assert False,'box_format given in detect unrecognised!'
                assert len(box_infos) > 0 ,'box infos is blank'

                frame_dets.append(box_infos)
            detections.append(frame_dets)

        return detections

    @staticmethod
    def _rand_uniform_strong(min, max):
        if min > max:
            swap = min
            min = max
            max = swap
        return random.random() * (max - min) + min

    def _rand_scale(self, s):
        scale = self._rand_uniform_strong(1, s)
        if random.randint(0, 1) % 2:
            return scale
        return 1. / scale

    @staticmethod
    def _fill_truth_detection(bboxes, num_boxes, classes, flip, dx, dy, sx, sy, net_w, net_h):
        if bboxes.shape[0] == 0:
            # return np.array(bboxes), 10000
            return np.zeros([0, 5]), 10000
        np.random.shuffle(bboxes)
        bboxes[:, 0] -= dx
        bboxes[:, 2] -= dx
        bboxes[:, 1] -= dy
        bboxes[:, 3] -= dy

        bboxes[:, 0] = np.clip(bboxes[:, 0], 0, sx)
        bboxes[:, 2] = np.clip(bboxes[:, 2], 0, sx)

        bboxes[:, 1] = np.clip(bboxes[:, 1], 0, sy)
        bboxes[:, 3] = np.clip(bboxes[:, 3], 0, sy)

        out_box = list(np.where(((bboxes[:, 1] == sy) & (bboxes[:, 3] == sy)) |
                                ((bboxes[:, 0] == sx) & (bboxes[:, 2] == sx)) |
                                ((bboxes[:, 1] == 0) & (bboxes[:, 3] == 0)) |
                                ((bboxes[:, 0] == 0) & (bboxes[:, 2] == 0)))[0])
        list_box = list(range(bboxes.shape[0]))
        for i in out_box:
            list_box.remove(i)
        bboxes = bboxes[list_box]

        if bboxes.shape[0] == 0:
            return bboxes, 10000

        bboxes = bboxes[np.where((bboxes[:, 4] < classes) & (bboxes[:, 4] >= 0))[0]]

        if bboxes.shape[0] > num_boxes:
            bboxes = bboxes[:num_boxes]

        min_w_h = np.array([bboxes[:, 2] - bboxes[:, 0], bboxes[:, 3] - bboxes[:, 1]]).min()

        bboxes[:, 0] = bboxes[:, 0] * (net_w / sx)
        bboxes[:, 2] = bboxes[:, 2] * (net_w / sx)
        bboxes[:, 1] = bboxes[:, 1] * (net_h / sy)
        bboxes[:, 3] = bboxes[:, 3] * (net_h / sy)

        if flip:
            temp = net_w - bboxes[:, 0]
            bboxes[:, 0] = net_w - bboxes[:, 2]
            bboxes[:, 2] = temp

        return bboxes, min_w_h

    @staticmethod
    def _rect_intersection(a, b):
        minx = max(a[0], b[0])
        miny = max(a[1], b[1])

        maxx = min(a[2], b[2])
        maxy = min(a[3], b[3])
        return [minx, miny, maxx, maxy]

    def _image_data_augmentation(self, mat, w, h, pleft, ptop, swidth, sheight, flip, dhue, dsat, dexp, gaussian_noise, blur, truth):
        try:
            img = mat
            oh, ow, _ = img.shape
            pleft, ptop, swidth, sheight = int(pleft), int(ptop), int(swidth), int(sheight)
            # crop
            src_rect = [pleft, ptop, swidth + pleft, sheight + ptop]  # x1,y1,x2,y2
            img_rect = [0, 0, ow, oh]
            new_src_rect = self._rect_intersection(src_rect, img_rect)  # 交集

            dst_rect = [max(0, -pleft), max(0, -ptop), max(0, -pleft) + new_src_rect[2] - new_src_rect[0],
                        max(0, -ptop) + new_src_rect[3] - new_src_rect[1]]
            # cv2.Mat sized

            if (src_rect[0] == 0 and src_rect[1] == 0 and src_rect[2] == img.shape[0] and src_rect[3] == img.shape[1]):
                sized = cv2.resize(img, (w, h), cv2.INTER_LINEAR)
            else:
                cropped = np.zeros([sheight, swidth, 3])
                cropped[:, :, ] = np.mean(img, axis=(0, 1))

                cropped[dst_rect[1]:dst_rect[3], dst_rect[0]:dst_rect[2]] = img[new_src_rect[1]:new_src_rect[3], new_src_rect[0]:new_src_rect[2]]

                # resize
                sized = cv2.resize(cropped, (w, h), cv2.INTER_LINEAR)

            # flip
            if flip:
                # cv2.Mat cropped
                sized = cv2.flip(sized, 1)  # 0 - x-axis, 1 - y-axis, -1 - both axes (x & y)

            # HSV augmentation
            # cv2.COLOR_BGR2HSV, cv2.COLOR_RGB2HSV, cv2.COLOR_HSV2BGR, cv2.COLOR_HSV2RGB
            if dsat != 1 or dexp != 1 or dhue != 0:
                if img.shape[2] >= 3:
                    hsv_src = cv2.cvtColor(sized.astype(np.float32), cv2.COLOR_RGB2HSV)  # RGB to HSV
                    hsv = cv2.split(hsv_src)
                    hsv[1] *= dsat
                    hsv[2] *= dexp
                    hsv[0] += 179 * dhue
                    hsv_src = cv2.merge(hsv)
                    sized = np.clip(cv2.cvtColor(hsv_src, cv2.COLOR_HSV2RGB), 0, 255)  # HSV to RGB (the same as previous)
                else:
                    sized *= dexp

            if blur:
                if blur == 1:
                    dst = cv2.GaussianBlur(sized, (17, 17), 0)
                    # cv2.bilateralFilter(sized, dst, 17, 75, 75)
                else:
                    ksize = (blur / 2) * 2 + 1
                    dst = cv2.GaussianBlur(sized, (ksize, ksize), 0)

                if blur == 1:
                    img_rect = [0, 0, sized.cols, sized.rows]
                    for b in truth:
                        left = (b.x - b.w / 2.) * sized.shape[1]
                        width = b.w * sized.shape[1]
                        top = (b.y - b.h / 2.) * sized.shape[0]
                        height = b.h * sized.shape[0]
                        roi(left, top, width, height)
                        roi = roi & img_rect
                        dst[roi[0]:roi[0] + roi[2], roi[1]:roi[1] + roi[3]] = sized[roi[0]:roi[0] + roi[2], roi[1]:roi[1] + roi[3]]

                sized = dst

            if gaussian_noise:
                noise = np.array(sized.shape)
                gaussian_noise = min(gaussian_noise, 127)
                gaussian_noise = max(gaussian_noise, 0)
                cv2.randn(noise, 0, gaussian_noise)  # mean and variance
                sized = sized + noise
        except:
            print("OpenCV can't augment image: " + str(w) + " x " + str(h))
            sized = mat

        return sized
