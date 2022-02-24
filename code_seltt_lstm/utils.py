import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split


def data_generator(root, batch_size):
    # typical LSTM과 동일하게 데이터셋으로 불러오는 과정. transform.Compose~ 부분도 이미지를 텐서로 만들고 노말라이즈하는 전형적인 값.
    train_set = datasets.MNIST(root=root, train=True, download=True,
                               transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]))
    train_set, val_set = random_split(train_set, [50000, 10000],)
            #generator=torch.Generator().manual_seed(42))

    test_set = datasets.MNIST(root=root, train=False, download=True,
                              transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,)) ]))
    print("<Util.py><def data_generator>")
    print(f"train: {len(train_set)}\tval: {len(val_set)}\ttest: {len(test_set)}")

    num_workers=12
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=num_workers) # 데이터로더는 50,000/128=390.xx 390번 데이터를 로드해준다
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,num_workers=num_workers)

    return train_loader, val_loader, test_loader


def count_model_params(model):
    """
    Returns number of trainable and non-trainable parameters in a model.
    :param model: A PyTorch nn.Module object.
    :return: A tuple (train_params_count, non_train_params_count)
    """
    print("<START><utils.py><def count_model_params>")
    print("model:{}".format(model))
    train_params_count = 0
    non_train_params_count = 0
    for p in model.parameters():                    #parameters(): 모든 파라미터를 하나씩 반환
        if p.requires_grad:
            train_params_count += p.numel()         # numel(): 원소 갯수
        else:
            non_train_params_count += p.numel()
    print("<END><utils.py><def count_model_params>")
    return train_params_count, non_train_params_count
