from einops import rearrange, repeat
import torch
from torchvision import transforms
from PIL import Image, ImageFilter, ImageOps
import random
from torchvision.ops.boxes import box_area
from pipeline.data_utils.randaugment import RandomAugment
from .builder import PROCESSORS
from torchvision.transforms.transforms import InterpolationMode
from torchvision.transforms import functional as F
 

def box_iou(boxes1, area1, boxes2, eps=1e-5):
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / (union+eps)
    return iou, union

def anchor_rank(anchors, anchors_areas, input_image_size, eps=1e-5):
    # anchors x1 y1 x2 y2

    # image_size: (h, w)
    # xyxy
    input_image_bbox = torch.tensor([0, 0, input_image_size[1], input_image_size[0]]).unsqueeze(0)

    boxes1 = anchors
    boxes2 = input_image_bbox
    boxes3 = anchors.clone()
    # y2
    boxes3[:,3] = input_image_size[0]/input_image_size[1]*anchors[:,2] # 用于算分辨率无关的iou
    
    area1 = anchors_areas
    
    iou, _ = box_iou(boxes1, area1, boxes2)
    iou = iou.squeeze(1)
    shape_iou, _ = box_iou(boxes1, area1, boxes3)
    shape_iou = shape_iou.diag()
    index = torch.argmax(shape_iou*100+iou,dim=0)
    return index

class AnchorResize(torch.nn.Module):
  
    def __init__(self, image_size, anchors, interpolation=InterpolationMode.BILINEAR, antialias=None):
        super().__init__()
        # xyxy
        self.anchors = torch.tensor(
            [[0, 0, _[1]*image_size[1], _[0]*image_size[0]] 
            for _ in anchors], requires_grad=False
        )
        
        self.anchor_areas = box_area(self.anchors)

        self.interpolation = interpolation
        self.antialias = antialias

    def forward(self, img, skip_resize=False):
        """
        Args:
            img (PIL Image or Tensor): Image to be scaled.

        Returns:
            PIL Image or Tensor: Rescaled image.
        """
        selected_anchor = anchor_rank(self.anchors, self.anchor_areas, (img.size[1], img.size[0]))
        target_size = self.anchors[selected_anchor][2:].tolist() # w,h
        if skip_resize:
            # for debug
            return selected_anchor
        return F.resize(img, [target_size[1],target_size[0]], self.interpolation, max_size=None, antialias=self.antialias), selected_anchor

    def __repr__(self) -> str:
        detail = f"(size={self.image_size}, anchor={self.anchors}, interpolation={self.interpolation.value}, antialias={self.antialias})"
        return f"{self.__class__.__name__}{detail}"

@PROCESSORS.register_module()
class DocPretrainProcessor:
    def __init__(self, image_size=224, anchors=[
        (1,1),
        (1,2),(2,1),
        (1,3),(3,1),
        (2,2),(1,4),(4,1),
        (1,5),(5,1),
        (1,6),(6,1),(2,3),(3,2),
        (1,7),(7,1),
        (4,2),(2,4),(1,8),(8,1),
        (3,3),(1,9),(9,1)]):
  
        # h,w
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_size = image_size
        # h,w
        self.anchors = [tuple(_) for _ in anchors]
        self.anchor_max = max([max(_) for _ in self.anchors])
        # xywh -> xyxy
        self.resizer = AnchorResize(image_size=image_size, anchors=anchors, interpolation=InterpolationMode.BICUBIC)
        self.old_resizer = transforms.Resize(image_size,interpolation=Image.BICUBIC)
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        self.text_transform = None
        self.ocr_instructions = ['The picture reads %s.', 
                                    'The image says %s.',
                                    'there are words %s in the image.',
                                    'Words %s are in the picture.',
                                    'The texts in this image read %s.', 
                                    'The words on this picture are %s.',
                                    'The script depicted in this image reads %s.',
                                    'The writing on this visual representation states %s.',
                                    'The content presented in this diagram states %s.',
                                    'The language used in this photograph says %s.',
                                    'The inscription on this picture explains %s.',
                                    'The verbiage included in this snapshot describes %s.']
    
    def _process_image(self, image):
        image, selected_anchor = self.resizer(image)
        image_input = self.image_transform(image) # h,w,3 -> 3,h,w
        # rearrange(x,'B C (n1 h) (n2 w) -> (B n1 n2) C h w', n1=self.down_sample[0], n2=self.down_sample[1])
        image_input = rearrange(image_input, 'C (num_h h) (num_w w) -> (num_h num_w) C h w', h=self.image_size[0], w=self.image_size[1])

        anchor = self.anchors[selected_anchor] # w,h
        patch_position = torch.cat([
            repeat(torch.arange(anchor[0]), 'num_h -> num_h num_w 1', num_w=anchor[1]),
            repeat(torch.arange(anchor[1]), 'num_w -> num_h num_w 1', num_h=anchor[0])],dim=2)
        patch_position = rearrange(patch_position, 'num_h num_w p-> (num_h num_w) p', p=2) # num_patch, (ph,pw)
        return image_input, patch_position

    def _process_text(self, text):
        if isinstance(text["prompt"], list):
            prompt = random.choice(text["prompt"])
        else:
            prompt = text["prompt"]

        # 分离<image>和文本
        image_token_str = text["text"][:text["text"].rfind('<image>')+len('<image>')]
        area_text =  text["text"][text["text"].rfind('<image>')+len('<image>'):]
        text["text"] = '\''+ text["text"] +'\''
        ocr_instruct=random.choice(self.ocr_instructions)
        text_input = dict(
            prompt=text["prompt"],
            completion=image_token_str + ocr_instruct % area_text,
        )
        return text_input

    def __call__(self, image, text):
        assert image or text
        patch_position = None
        if image:
            image_input, patch_position = self._process_image(image)
        else:
            image_input = None

        if text:
            text_input = self._process_text(text)
        else:
            text_input = None
        return image_input, text_input, patch_position

@PROCESSORS.register_module()
class DocSFTProcessor(DocPretrainProcessor):
   
    def _process_text(self, text):
        if isinstance(text["prompt"], list):
            prompt = random.choice(text["prompt"])
        else:
            prompt = text["prompt"]

        text_input = dict(
            prompt=text["prompt"],
            completion=text["text"],
        )
        return text_input


@PROCESSORS.register_module()
class DocNoCutProcessor:
    def __init__(self, image_size=224, anchors=None):
        self.image_size = image_size

        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size),interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        self.text_transform = None

    def __call__(self, image, text):
        assert image or text
        patch_position = None
        if image:
            image_input = self.image_transform(image).unsqueeze(0)
            patch_position = torch.zeros(1,2).long()
        else:
            image_input = None

        if text:
            if isinstance(text["prompt"], list):
                prompt = random.choice(text["prompt"])
            else:
                prompt = text["prompt"]
            text_input = dict(
                prompt=prompt,
                completion=text["text"],
            )
        else:
            text_input = None
        return image_input, text_input, patch_position

@PROCESSORS.register_module()
class DocNewSFTProcessor(DocSFTProcessor):
    '''
    新Processor用make_text预组织文本 下游task如果需要定制 可以继承这个类进行改进
    {
        "image": ["xxx"],
        "prompt": "", 
        "text": "", 
        "system_instruction": "", 
        "conversations": [
            {"from": "user", "value": "<image>"}, 
            {"from": "user", "value": "Which country has longest bar?"}, 
            {"from": "assistant", "value": "Nigeria"}
        ], 
        "task_type": "qa_sft"
    }
    '''

    def build_text(self, data):
        fin_text = ''
        if 'system_instruction' in data:
            if isinstance(data['system_instruction'], str):
                fin_text+=data['system_instruction']
            elif isinstance(data['system_instruction'], list):
                fin_text+=random.choice(data['system_instruction'])
            else:
                pass
            if not fin_text.endswith('\n'):
                fin_text += '\n'
        
        for cv in data['conversations']:
            if cv['from'] == 'user':
                fin_text+='Human: '+cv['value']
            elif cv['from'] == 'assistant':
                fin_text+='AI: '+cv['value']
            if not fin_text.endswith('\n'):
                fin_text += '\n'
        return fin_text


@PROCESSORS.register_module()
class DocNewMultiScaleSFTProcessor(DocNewSFTProcessor):
    def _process_image(self, image):
        nocut_image = self.image_transform(self.old_resizer(image)).unsqueeze(0)

        image, selected_anchor = self.resizer(image)
        image_input = self.image_transform(image) # h,w,3 -> 3,h,w
        # rearrange(x,'B C (n1 h) (n2 w) -> (B n1 n2) C h w', n1=self.down_sample[0], n2=self.down_sample[1])
        image_input = rearrange(image_input, 'C (num_h h) (num_w w) -> (num_h num_w) C h w', h=self.image_size[0], w=self.image_size[1])

        anchor = self.anchors[selected_anchor] # w,h
        patch_position = torch.cat([
            repeat(torch.arange(anchor[0]), 'num_h -> num_h num_w 1', num_w=anchor[1]),
            repeat(torch.arange(anchor[1]), 'num_w -> num_h num_w 1', num_h=anchor[0])],dim=2)
        patch_position = rearrange(patch_position, 'num_h num_w p-> (num_h num_w) p', p=2) # num_patch, (ph,pw)
        
        image_input = torch.cat([nocut_image, image_input], dim=0)
        patch_position = torch.cat([torch.ones(1,2).long()*self.anchor_max, patch_position], dim=0) # 切片id为0~8
        return image_input, patch_position


@PROCESSORS.register_module()
class DocAdaptiveMultiScaleSFTProcessor(DocNewMultiScaleSFTProcessor):
    """
    形状自适应裁剪模块（内容密度 + ROI）：
    - 保留全图patch（global view）
    - 基于内容密度图提议多个ROI，保持长宽比后缩放填充到统一尺寸
    - 输出 (N, C, H, W) 和对应 patch_position 供后续位置编码使用
    """

    def __init__(
        self,
        image_size=224,
        anchors=[(1, 1)],
        roi_grid_size=12,
        roi_window_size=3,
        roi_stride=2,
        max_roi_patches=8,
        roi_score_threshold=0.15,
    ):
        super().__init__(image_size=image_size, anchors=anchors)
        self.roi_grid_size = int(roi_grid_size)
        self.roi_window_size = int(roi_window_size)
        self.roi_stride = int(roi_stride)
        self.max_roi_patches = int(max_roi_patches)
        self.roi_score_threshold = float(roi_score_threshold)

    def _content_density_map(self, image):
        gray = transforms.ToTensor()(image.convert('L')).squeeze(0)
        gx = torch.zeros_like(gray)
        gy = torch.zeros_like(gray)
        gx[:, 1:] = (gray[:, 1:] - gray[:, :-1]).abs()
        gy[1:, :] = (gray[1:, :] - gray[:-1, :]).abs()
        edge = gx + gy

        mean = F.gaussian_blur(gray.unsqueeze(0), [3, 3], [1.0, 1.0]).squeeze(0)
        var = (gray - mean).pow(2)
        density = 0.7 * edge + 0.3 * var
        density = density - density.min()
        density = density / (density.max() + 1e-6)
        return density

    def _grid_rois(self, density, width, height):
        grid = self.roi_grid_size
        ws = min(self.roi_window_size, grid)
        pooled = torch.nn.functional.adaptive_avg_pool2d(density.unsqueeze(0).unsqueeze(0), (grid, grid)).squeeze(0).squeeze(0)

        candidates = []
        for y in range(0, grid - ws + 1, self.roi_stride):
            for x in range(0, grid - ws + 1, self.roi_stride):
                score = pooled[y:y + ws, x:x + ws].mean().item()
                if score < self.roi_score_threshold:
                    continue
                candidates.append((score, x, y, x + ws, y + ws))

        candidates.sort(key=lambda x: x[0], reverse=True)

        selected = []
        for cand in candidates:
            if len(selected) >= self.max_roi_patches:
                break
            _, x1, y1, x2, y2 = cand
            ok = True
            for _, sx1, sy1, sx2, sy2 in selected:
                inter_w = max(0, min(x2, sx2) - max(x1, sx1))
                inter_h = max(0, min(y2, sy2) - max(y1, sy1))
                inter = inter_w * inter_h
                area1 = (x2 - x1) * (y2 - y1)
                area2 = (sx2 - sx1) * (sy2 - sy1)
                iou = inter / (area1 + area2 - inter + 1e-6)
                if iou > 0.4:
                    ok = False
                    break
            if ok:
                selected.append(cand)

        rois = []
        for _, x1, y1, x2, y2 in selected:
            px1 = int(round(x1 / grid * width))
            py1 = int(round(y1 / grid * height))
            px2 = int(round(x2 / grid * width))
            py2 = int(round(y2 / grid * height))
            if px2 - px1 < 2 or py2 - py1 < 2:
                continue
            rois.append((px1, py1, px2, py2))

        if len(rois) == 0:
            rois = [(0, 0, width, height)]
        return rois

    def _crop_with_aspect_preserve(self, image, box):
        x1, y1, x2, y2 = box
        roi = image.crop((x1, y1, x2, y2))
        # 保持长宽比，padding到固定输入尺寸
        return ImageOps.pad(roi, (self.image_size[1], self.image_size[0]), method=Image.BICUBIC, color=(255, 255, 255), centering=(0.5, 0.5))

    def _roi_position(self, box, width, height):
        x1, y1, x2, y2 = box
        cx = ((x1 + x2) * 0.5) / max(width, 1)
        cy = ((y1 + y2) * 0.5) / max(height, 1)
        gx = min(self.anchor_max - 1, max(0, int(cx * (self.anchor_max - 1))))
        gy = min(self.anchor_max - 1, max(0, int(cy * (self.anchor_max - 1))))
        return torch.tensor([gy, gx]).long()

    def _process_image(self, image):
        width, height = image.size

        # global patch，防止ROI漏检关键语义
        global_patch = self.image_transform(self.old_resizer(image)).unsqueeze(0)
        global_pos = torch.ones(1, 2).long() * self.anchor_max

        density = self._content_density_map(image)
        rois = self._grid_rois(density, width, height)

        roi_tensors = []
        roi_positions = []
        for box in rois[: self.max_roi_patches]:
            roi_img = self._crop_with_aspect_preserve(image, box)
            roi_tensors.append(self.image_transform(roi_img).unsqueeze(0))
            roi_positions.append(self._roi_position(box, width, height).unsqueeze(0))

        if len(roi_tensors) == 0:
            return global_patch, global_pos

        # 兼容 pre 模式中对 (0,0) 作为切片起始标记的逻辑
        roi_positions[0] = torch.zeros_like(roi_positions[0])

        image_input = torch.cat([global_patch] + roi_tensors, dim=0)
        patch_position = torch.cat([global_pos] + roi_positions, dim=0)
        return image_input, patch_position

if __name__ == '__main__':
    pre_pc = DocPretrainProcessor()
    pass
