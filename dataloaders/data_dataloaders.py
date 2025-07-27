import torch
from torch.utils.data import DataLoader
from dataloaders.dataloader_MGSV_EC_feature import MGSV_EC_DataLoader


def dataloader_MGSV_EC_train(args):
    MGSV_EC_trainset = MGSV_EC_DataLoader(
        csv_path=args.train_csv,
        args=args,
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(MGSV_EC_trainset, num_replicas=args.world_size, rank=args.rank)
    dataloader = DataLoader(
        MGSV_EC_trainset,
        batch_size=args.batch_size_train // args.gpu_num,
        num_workers=args.num_workers,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
        pin_memory=True,
    )
    return dataloader, len(MGSV_EC_trainset), train_sampler

def dataloader_MGSV_EC_val(args):
    MGSV_EC_valset = MGSV_EC_DataLoader(
        csv_path=args.val_csv,
        args=args,
    )
    val_sampler = torch.utils.data.distributed.DistributedSampler(MGSV_EC_valset, num_replicas=args.world_size, rank=args.rank)
    dataloader = DataLoader(
        MGSV_EC_valset,
        batch_size=args.batch_size_val // args.gpu_num,
        num_workers=args.num_workers,
        shuffle=(val_sampler is None),
        sampler=val_sampler,
        drop_last=False,
    )
    return dataloader, len(MGSV_EC_valset), val_sampler

def dataloader_MGSV_EC_test(args):
    MGSV_EC_testset = MGSV_EC_DataLoader(
        csv_path=args.test_csv,
        args=args,
    )
    test_sampler = torch.utils.data.distributed.DistributedSampler(MGSV_EC_testset, num_replicas=args.world_size, rank=args.rank)
    dataloader = DataLoader(
        MGSV_EC_testset,
        batch_size=args.batch_size_val // args.gpu_num,
        num_workers=args.num_workers,
        shuffle=(test_sampler is None),
        sampler=test_sampler,
        drop_last=False,
    )
    return dataloader, len(MGSV_EC_testset), test_sampler



DATALOADER_DICT = {}
DATALOADER_DICT["kuai50k_uni"] = {
    "train": dataloader_MGSV_EC_train,
    "val": dataloader_MGSV_EC_val,
    "test": dataloader_MGSV_EC_test
}
# DATALOADER_DICT["kuai50k_vmr"] = {
#     "train": dataloader_MGSV_EC_train,
#     "val": dataloader_MGSV_EC_val,
#     "test": dataloader_MGSV_EC_test
# }
# DATALOADER_DICT["kuai50k_mr"] = {
#     "train": dataloader_MGSV_EC_train,
#     "val": dataloader_MGSV_EC_val,
#     "test": dataloader_MGSV_EC_test
# }