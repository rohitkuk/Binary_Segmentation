from torchvision import transforms

IMG_Trasforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0])

    ])

MASK_Trasforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.0),
        (1.0))
    ])