import os
from torchvision import transforms
from .transforms import *
from .masking_generator import TubeMaskingGenerator, RandomMaskingGenerator
from .kvq_datasets import KVQ_VideoClsDataset
import yaml
import copy




class DataAugmentationForVideoMAE(object):
    def __init__(self, args):
        self.input_mean = [0.485, 0.456, 0.406]  # IMAGENET_DEFAULT_MEAN
        self.input_std = [0.229, 0.224, 0.225]  # IMAGENET_DEFAULT_STD
        normalize = GroupNormalize(self.input_mean, self.input_std)
        self.train_augmentation = GroupMultiScaleCrop(args.input_size, [1, .875, .75, .66])
        if args.color_jitter > 0:
            self.transform = transforms.Compose([                            
                self.train_augmentation,
                GroupColorJitter(args.color_jitter),
                GroupRandomHorizontalFlip(flip=args.flip),
                Stack(roll=False),
                ToTorchFormatTensor(div=True),
                normalize,
            ])
        else:
            self.transform = transforms.Compose([                            
                self.train_augmentation,
                GroupRandomHorizontalFlip(flip=args.flip),
                Stack(roll=False),
                ToTorchFormatTensor(div=True),
                normalize,
            ])
        if args.mask_type == 'tube':
            self.masked_position_generator = TubeMaskingGenerator(
                args.window_size, args.mask_ratio
            )
        elif args.mask_type == 'random':
            self.masked_position_generator = RandomMaskingGenerator(
                args.window_size, args.mask_ratio
            )
        elif args.mask_type in 'attention':
            self.masked_position_generator = None

    def __call__(self, images):
        process_data, _ = self.transform(images)
        if self.masked_position_generator is None:
            return process_data, -1
        else:
            return process_data, self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForVideoMAE,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr


def build_dataset(is_train, test_mode, args):
    print(f'Use Dataset: {args.data_set}')
    
    if args.data_set == 'KVQ':
        mode = None
        data_path = args.data_path
        if is_train is True:
            mode = 'train'
            # data_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            # data_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'val'
            # data_path = os.path.join(args.data_path, 'val.csv') 

     
        dataset = KVQ_VideoClsDataset(
            data_path=args.data_path,
            prefix='',
            split=args.split,
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            keep_aspect_ratio=True,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            args=args
        )


        
        nb_classes = args.nb_classes

    else:
        print(f'Wrong: {args.data_set}')
        raise NotImplementedError()
    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes

