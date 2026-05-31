from __future__ import annotations

from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _normalize_op() -> T.Normalize:
    return T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def get_train_transforms(img_size: int = 224) -> T.Compose:
    return T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.Pad(10, padding_mode="reflect"),
            T.RandomCrop((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=30),
            T.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1,
                hue=0.05,
            ),
            T.ToTensor(),
            _normalize_op(),
        ]
    )


def get_val_transforms(img_size: int = 224) -> T.Compose:
    return T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            _normalize_op(),
        ]
    )


def get_simclr_transforms(img_size: int = 224) -> T.Compose:
    return T.Compose(
        [
            T.RandomResizedCrop((img_size, img_size), scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(
                brightness=0.5,
                contrast=0.5,
                saturation=0.4,
                hue=0.1,
            ),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0)),
            T.ToTensor(),
            _normalize_op(),
        ]
    )


def get_teacher_transforms(img_size: int = 224) -> T.Compose:
    return T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.05,
                hue=0.02,
            ),
            T.ToTensor(),
            _normalize_op(),
        ]
    )


def get_student_transforms(img_size: int = 224) -> T.Compose:
    return get_train_transforms(img_size=img_size)
