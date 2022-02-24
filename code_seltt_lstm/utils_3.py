import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split


def data_generator(root, batch_size):
    # batch_size = 4
    # transform = transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform = transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    train_set = datasets.CIFAR10(root=root, train=True,
                                 download=True, transform=transform)
    print( " len(train_set) : ", len(train_set))            # 50,000                    [ 50,000, ([3, 32, 32] , 6) ]
    print( " len(train_set[0]) : ", len(train_set[0]))      # 2
    print( " train_set[0][0].shape : ", train_set[0][0].shape)  # torch.Size([3, 32, 32])
    print( " train_set[0][1] : ", train_set[0][1])  # int / 6 /


    train_set, val_set = random_split(train_set, [40000, 10000],)
            #generator=torch.Generator().manual_seed(42))

    test_set = datasets.CIFAR10(root=root, train=False,
                               download=True, transform=transform)
    print("<Util.py><def data_generator>")
    print(f"train: {len(train_set)}\tval: {len(val_set)}\ttest: {len(test_set)}")

    num_workers=2
    # train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    # test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)

    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

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
